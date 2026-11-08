"""Train the U-Net on whole resized pages, pick the best by val box IoU.

Usage:
  work/.venv/Scripts/python work/train.py --memtest
  work/.venv/Scripts/python work/train.py --minutes 60 --tag v0.2
  work/.venv/Scripts/python work/train.py --splits train val --epochs 120 --tag v0.2-final   # no val selection
Outputs: work/checkpoints/<tag>/{best,last}.pt, work/checkpoints/<tag>/train_log.csv

v0.2 changes vs v0.1:
  - legend class weighted 3x in the loss (it is the smallest region)
  - "blank column" augmentation: random vertical stretches inside the depth and
    lithology boxes are painted white (border lines kept), target unchanged, so the
    model learns that an empty stretch of a column is still the column
  - input building and augmentation run on the GPU (v0.1 was CPU-bound)
"""
import argparse
import csv
import math
import random
import time

import numpy as np
import torch

from model import UNet, loss_fn, make_input, make_target, metrics, probs_to_boxes
from pages import BOXES, MAX_H, OUT_W, ROOT, WORK, boxes_to_small, load_resized, read_splits

SEED = 42
LR = 1e-3
CLASS_W = (1.0, 1.0, 3.0)
SHORT = {"depth_column": "depth", "interpreted_lithology_column": "lithology", "lithology_legend": "legend"}


def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def augment(x, rng, small_boxes):
    """Light brightness/contrast + noise on RGB, plus blank-column stretches. No flips."""
    rgb = x[:3]
    # blank column: whiten 1-3 random vertical stretches inside each column box (p = 0.5 per column)
    for k in ("depth_column", "interpreted_lithology_column"):
        if rng.random() < 0.5:
            x1, y1, x2, y2 = [int(round(v)) for v in small_boxes[k]]
            if x2 - x1 > 6 and y2 - y1 > 40:
                for _ in range(rng.randint(1, 3)):
                    L = int((y2 - y1) * rng.uniform(0.05, 0.3))
                    s = rng.randint(y1, max(y1, y2 - L))
                    rgb[:, s:s + L, x1 + 2:x2 - 2] = 1.0  # white, keep the 2 px border lines
    rgb = rgb * rng.uniform(0.85, 1.15) + rng.uniform(-0.15, 0.15)
    rgb = rgb + torch.randn_like(rgb) * rng.uniform(0.0, 0.08)
    return torch.cat([rgb.clamp(-1, 1), x[3:]], 0)


def load_split(rows, splits):
    data = []
    for r in rows:
        if r["split"] in splits:
            page = load_resized(ROOT / r["image"], r["stem"], r["width"], r["height"])
            data.append((r, page, make_target(r["boxes"], r["width"], r["height"])))
    return data


@torch.no_grad()
def predict_probs(model, page, dev):
    x = make_input(page, dev)[None]
    if dev.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
    else:  # CPU fallback: plain float32, slower but works without an NVIDIA GPU
        logits = model(x)
    return torch.sigmoid(logits.float())[0].cpu().numpy()


def evaluate(model, data, dev):
    model.eval()
    per = {k: [] for k in BOXES}
    for r, page, _ in data:
        pred = probs_to_boxes(predict_probs(model, page, dev), r["width"], r["height"])
        m = metrics(pred, r["boxes"], r["width"], r["height"])
        for k in BOXES:
            per[k].append(m[k]["iou"])
    model.train()
    return {k: float(np.mean(v)) for k, v in per.items()}


def memtest(dev):
    model = UNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    x = torch.randn(1, 5, MAX_H, OUT_W, device=dev)
    t = torch.zeros(1, 3, MAX_H, OUT_W, device=dev)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(x).float(), t, CLASS_W)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    print(f"input {OUT_W}x{MAX_H}: peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
          f"{(time.time() - t0) / 3:.2f} s/step (incl. warm-up), params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60, help="time budget (ignored if --epochs is set)")
    ap.add_argument("--epochs", type=int, default=0, help="fixed epoch count (for train+val runs)")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--eval-every", type=float, default=5, help="minutes between val evaluations")
    ap.add_argument("--tag", default="v0.2")
    ap.add_argument("--memtest", action="store_true")
    a = ap.parse_args()
    seed_all()
    dev = torch.device("cuda")
    if a.memtest:
        return memtest(dev)

    rows = read_splits()
    assert "test" not in a.splits
    train = load_split(rows, a.splits)
    do_eval = "val" not in a.splits
    val = load_split(rows, ["val"]) if do_eval else []
    print(f"tag {a.tag}: training on {a.splits} ({len(train)} pages), val {len(val)} pages", flush=True)
    model = UNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = random.Random(SEED)
    out = WORK / "checkpoints" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.csv", "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["minutes", "epoch", "train_loss"] + [f"val_iou_{k}" for k in BOXES] + ["val_iou_mean"])

    small = [boxes_to_small(r["boxes"], r["width"], r["height"]) for r, _, _ in train]
    budget = a.minutes * 60
    t0 = time.time()
    next_eval = a.eval_every * 60
    best, epoch, losses = -1.0, 0, []
    while True:
        epoch += 1
        order = list(range(len(train)))
        rng.shuffle(order)
        for i in order:
            el = time.time() - t0
            frac = min(1.0, (epoch - 1 + order.index(i) / len(order)) / a.epochs) if a.epochs else min(1.0, el / budget)
            lr = LR * min(1.0, el / 60) * 0.5 * (1 + math.cos(math.pi * frac))  # 1 min warm-up, cosine decay
            for g in opt.param_groups:
                g["lr"] = max(lr, 1e-6)
            r, page, tgt = train[i]
            x = augment(make_input(page, dev), rng, small[i])[None]
            t = torch.from_numpy(tgt)[None].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = loss_fn(logits.float(), t, CLASS_W)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        el = time.time() - t0
        done = epoch >= a.epochs if a.epochs else el >= budget
        if el >= next_eval or done:
            next_eval += a.eval_every * 60
            tl = float(np.mean(losses))
            losses = []
            if do_eval:
                v = evaluate(model, val, dev)
                mean = float(np.mean(list(v.values())))
                lw.writerow([f"{el / 60:.1f}", epoch, f"{tl:.4f}"] + [f"{v[k]:.4f}" for k in BOXES] + [f"{mean:.4f}"])
                print(f"{el / 60:5.1f} min  epoch {epoch:4d}  loss {tl:.4f}  val IoU "
                      + "  ".join(f"{SHORT[k]} {v[k]:.3f}" for k in BOXES) + f"  mean {mean:.3f}", flush=True)
                if mean > best:
                    best = mean
                    torch.save({"model": model.state_dict(), "epoch": epoch, "val": v}, out / "best.pt")
            else:
                lw.writerow([f"{el / 60:.1f}", epoch, f"{tl:.4f}"] + [""] * 4)
                print(f"{el / 60:5.1f} min  epoch {epoch:4d}  loss {tl:.4f}", flush=True)
            log.flush()
        if done:
            break
    torch.save({"model": model.state_dict(), "epoch": epoch}, out / "last.pt")
    print(f"done: {epoch} epochs" + (f", best val mean IoU {best:.3f}" if do_eval else ""), flush=True)


if __name__ == "__main__":
    main()
