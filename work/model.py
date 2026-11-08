"""U-Net, input/target building, post-processing and metrics (shared by train.py and predict.py)."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pages import BOXES, boxes_to_orig, boxes_to_small, target_size

CHANNELS = (16, 32, 64, 128, 256)  # 4 poolings -> needs multiple of 16; we use multiples of 32


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Plain U-Net. Input: RGB + 2 coordinate channels (x, y in [0, 1]).
    The bottleneck adds dilated convs so the net sees more of the page at once."""

    def __init__(self, cin=5, n_out=3, ch=CHANNELS):
        super().__init__()
        self.downs = nn.ModuleList()
        c = cin
        for co in ch[:-1]:
            self.downs.append(block(c, co))
            c = co
        self.mid = block(c, ch[-1])
        self.context = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(ch[-1], ch[-1], 3, padding=d, dilation=d, bias=False), nn.GroupNorm(8, ch[-1]),
            nn.ReLU(inplace=True)) for d in (2, 4, 8, 16)])
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        c = ch[-1]
        for co in reversed(ch[:-1]):
            self.ups.append(nn.ConvTranspose2d(c, co, 2, stride=2))
            self.dec.append(block(co * 2, co))
            c = co
        self.head = nn.Conv2d(c, n_out, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        x = x + self.context(x)
        for up, dec, s in zip(self.ups, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], 1))
        return self.head(x)


def make_input(page_u8, dev=None):
    """uint8 (H, W, 3) -> float tensor (5, H, W): normalised RGB + x/y coordinates.
    Built on `dev` (the GPU) so the CPU is not the bottleneck."""
    h, w = page_u8.shape[:2]
    rgb = torch.from_numpy(page_u8).to(dev).permute(2, 0, 1).float() / 255.0
    yy = torch.linspace(0, 1, h, device=dev).view(1, h, 1).expand(1, h, w)
    xx = torch.linspace(0, 1, w, device=dev).view(1, 1, w).expand(1, h, w)
    return torch.cat([(rgb - 0.5) / 0.5, xx, yy], 0)


def make_target(boxes, w, h):
    """Filled supplied boxes at model size -> float32 (3, out_h, out_w)."""
    out_w, out_h = target_size(w, h)
    t = np.zeros((3, out_h, out_w), np.float32)
    for i, k in enumerate(BOXES):
        x1, y1, x2, y2 = boxes_to_small(boxes, w, h)[k]
        x1, y1 = int(np.floor(x1)), int(np.floor(y1))
        x2, y2 = max(x1 + 1, int(np.ceil(x2))), max(y1 + 1, int(np.ceil(y2)))
        t[i, y1:y2, x1:x2] = 1
    return t


def loss_fn(logits, target, class_w=(1.0, 1.0, 1.0)):
    """BCE + Dice, each averaged per class and weighted by class_w (depth, lithology, legend)."""
    w = torch.tensor(class_w, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((0, 2, 3))
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    dice = (1 - (2 * inter + 1) / (p.sum((2, 3)) + target.sum((2, 3)) + 1)).mean(0)
    return ((bce + dice) * w).sum() / w.sum()


COLUMN_CLASSES = ("depth_column", "interpreted_lithology_column")


def probs_to_boxes(prob, w, h, thr=0.5, merge_columns=True):
    """prob (3, out_h, out_w) -> {class: [x1,y1,x2,y2]} in ORIGINAL pixels.
    Largest connected component per class, then its bounding rectangle.
    Columns: the model often marks only the filled parts of a column, so the mask
    breaks into vertical pieces. With merge_columns, every component that overlaps
    the largest one by >= 50% in x is kept, and the box spans all of them in y."""
    small = {}
    for i, k in enumerate(BOXES):
        m = prob[i] > thr
        if not m.any():  # nothing above threshold: use half of this class's max
            m = prob[i] > prob[i].max() * 0.5
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh = stats[j, :4]
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        if merge_columns and k in COLUMN_CLASSES:
            for c in range(1, n):
                cx, cy, cw, ch = stats[c, :4]
                ov = max(0, min(x + bw, cx + cw) - max(x, cx))
                if ov >= 0.5 * min(bw, cw):
                    y1, y2 = min(y1, cy), max(y2, cy + ch)
        small[k] = [x1, y1, x2, y2]
    out = boxes_to_orig(small, w, h)
    return {k: [min(max(0, v), lim) for v, lim in zip(b, (w, h, w, h))] for k, b in out.items()}


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def metrics(pred, gt, w, h):
    """Per class: box IoU, mean |edge error| in original px, and pixel error =
    fraction of page pixels where the predicted box and the supplied box disagree."""
    out = {}
    for k in BOXES:
        p, g = pred[k], gt[k]
        inter = max(0, min(p[2], g[2]) - max(p[0], g[0])) * max(0, min(p[3], g[3]) - max(p[1], g[1]))
        area_p, area_g = (p[2] - p[0]) * (p[3] - p[1]), (g[2] - g[0]) * (g[3] - g[1])
        out[k] = {"iou": box_iou(p, g),
                  "edge_px": float(np.mean([abs(a - b) for a, b in zip(p, g)])),
                  "pixel_err": (area_p + area_g - 2 * inter) / (w * h)}
    return out
