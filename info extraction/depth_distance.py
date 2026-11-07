import os
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "real_test1.png")
output_path = os.path.join(script_dir, "test1_annotated.png")

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not open image at {image_path}")
else:
    result, _ = engine(image)

    if not result:
        print("No text detected by RapidOCR.")
    else:
        depth_points = []
        
        print("--- RapidOCR Detected Text (Early Stop Enabled) ---")
        for line in result:
            box, text, confidence = line
            text = text.strip()
            
            try:
                conf_val = float(confidence)
            except ValueError:
                conf_val = 0.0
                
            # Check if text is a valid 4-digit depth number
            if text.isdigit() and len(text) == 4:
                depth_val = int(text)
                y_coords = [point[1] for point in box]
                y_center = sum(y_coords) / len(y_coords)
                depth_points.append((depth_val, y_center))
                print(f"Valid depth found: '{depth_val}' at y={y_center:.1f} (Conf: {conf_val:.2f})")

                # Stop immediately once we have found 2 valid depth numbers
                if len(depth_points) >= 2:
                    print("--> Found 2 valid numbers. Stopping early to save resources.")
                    break

            # Draw bounding box for visualization
            pts = np.array(box, dtype=np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            org = (int(box[0][0]), int(box[0][1]) - 5)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imwrite(output_path, image)

        # Sort detected depths numerically in case they weren't read in top-down order
        depth_points = sorted(depth_points, key=lambda x: x[0])

        if len(depth_points) >= 2:
            d_start, y_start = depth_points[0]
            d_end, y_end = depth_points[-1]
            
            pixel_diff = abs(y_end - y_start)
            depth_diff = abs(d_end - d_start)
            
            depth_per_pixel = depth_diff / pixel_diff
            
            print("\n------------------------")
            print(f"Detected Depth Range: {d_start} to {d_end}")
            print(f"Pixel Distance: {pixel_diff} pixels")
            print(f"Scale: 1 pixel = {depth_per_pixel:.4f} depth units")
        else:
            print("\nInsufficient valid depth numbers found for scale calculation.")