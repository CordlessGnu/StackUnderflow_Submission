import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Initialize RapidOCR
engine = RapidOCR()

# 1. Load image in grayscale
img = cv2.imread('larpteryx.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Could not load image. Check file path.")

h, w = img.shape
edges = cv2.Canny(img, 50, 150)

# --- A. DETECT VERTICAL TRACKS ---
vertical_profile = np.sum(edges, axis=0)
kernel_w = max(5, int(w * 0.005))
smoothed_v_profile = np.convolve(vertical_profile, np.ones(kernel_w)/kernel_w, mode='same')
from scipy.signal import find_peaks
col_peaks, _ = find_peaks(smoothed_v_profile, distance=int(w * 0.03), prominence=np.ptp(smoothed_v_profile) * 0.1)

separators = [0] + [int(p) for p in col_peaks if 10 < p < w - 10] + [w]
separators = sorted(list(set(separators)))

col_bounds = []
for i in range(len(separators) - 1):
    col_bounds.append((separators[i], separators[i + 1]))


# --- B. USE RAPIDOCR TO FIND THE LITHOLOGY COLUMN TRACK & SELECT THE PREVIOUS ONE ---
# Cap the crop height: RapidOCR downscales large inputs, which loses the small header text
header_h = min(int(h * 0.15), 3000)
header_crop = img[0:header_h, :]
ocr_result, _ = engine(header_crop)

lithology_x_center = None
lithology_text_bottom_y = None
if ocr_result:
    candidates = [item for item in ocr_result if "LITH" in item[1].strip().upper()]
    # Prefer the exact track header over legend entries such as "LITHOLOGY SYMBOL"
    exact = [item for item in candidates if item[1].strip().upper() == "LITHOLOGY"]
    for item in (exact or candidates):
        box, text, score = item
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lithology_x_center = int(np.mean(xs))
        lithology_text_bottom_y = int(max(ys))
        break

# Find which track index contains this x-coordinate
lithology_track_idx = None
if lithology_x_center is not None:
    for idx, (start_x, end_x) in enumerate(col_bounds):
        if start_x <= lithology_x_center <= end_x:
            lithology_track_idx = idx
            break

# Fallback if text isn't matched clearly
if lithology_track_idx is None:
    lithology_track_idx = len(col_bounds) - 1

# Use the lithology track itself
target_track_idx = lithology_track_idx
start_x, end_x = col_bounds[target_track_idx]


# --- C. DETECT HORIZONTAL LINES SPECIFICALLY FOR THAT TRACK ---
start_x, end_x = col_bounds[target_track_idx]
lithology_strip = img[:, start_x:end_x]

_, bin_strip = cv2.threshold(lithology_strip, 200, 255, cv2.THRESH_BINARY_INV)
horiz_kernel_len = int((end_x - start_x) * 0.5)
horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
full_horiz_lines = cv2.morphologyEx(bin_strip, cv2.MORPH_OPEN, horiz_kernel)

row_sums = np.sum(full_horiz_lines, axis=1)
# Require near-full-width coverage so text strokes are not mistaken for track lines
row_indices = np.where(row_sums > 0.8 * (end_x - start_x) * 255)[0]

lithology_lines_y = []
if len(row_indices) > 0:
    current_cluster = [row_indices[0]]
    for r in row_indices[1:]:
        if r - current_cluster[-1] <= 3:
            current_cluster.append(r)
        else:
            lithology_lines_y.append(int(np.mean(current_cluster)))
            current_cluster = [r]
    lithology_lines_y.append(int(np.mean(current_cluster)))

# Filter to keep ONLY the first detected line below the header text and the final line
below_header = lithology_lines_y
if lithology_text_bottom_y is not None:
    below_header = [y for y in lithology_lines_y if y > lithology_text_bottom_y]

filtered_lines_y = []
if len(below_header) >= 1:
    filtered_lines_y.append(below_header[0])   # First line below the header text
if len(below_header) >= 2:
    filtered_lines_y.append(below_header[-1])  # Final bottom line of the track


# --- D. VISUALIZATION ---
output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.rectangle(output_img, (start_x, 0), (end_x, h), (0, 255, 0), 2)
cv2.putText(output_img, "TARGET TRACK", (start_x + 5, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

for y_pos in filtered_lines_y:
    cv2.line(output_img, (start_x, y_pos), (end_x, y_pos), (0, 0, 255), 1)

cv2.imwrite('accurate_tracks_visualized.png', output_img)
print(f"Target track found at X: {start_x} to {end_x} with {len(filtered_lines_y)} horizontal lines rendered.")