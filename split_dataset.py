import os
import shutil
from pathlib import Path
import random

# Define base dataset directories
base_dir = Path("Dataset/dataset")
image_dir = base_dir / "images"
annot_dir = base_dir / "bbox_annotation"

# Define output directory for split dataset
output_dir = Path("Dataset/split_dataset")

# Split ratios (must sum to 1.0)
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Set random seed for reproducibility
random.seed(42)

# Create output directory structure (excluding bbox_vis)
splits = ['train', 'val', 'test']
subfolders = ['images', 'bbox_annotation']

for split in splits:
    for subfolder in subfolders:
        (output_dir / split / subfolder).mkdir(parents=True, exist_ok=True)

print("Directory structure created successfully.")

# Get all valid image filenames that have an exact matching JSON in bbox_annotation[cite: 1, 4]
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
valid_images = []

for f in os.listdir(image_dir):
    if f.lower().endswith(tuple(image_extensions)):
        stem = Path(f).stem
        json_annot = annot_dir / f"{stem}.json"
        
        # Ensure the image has an exact matching JSON file[cite: 4]
        if json_annot.exists():
            valid_images.append(f)

# Shuffle the paired file list randomly
random.shuffle(valid_images)

total_images = len(valid_images)
train_end = int(total_images * train_ratio)
val_end = train_end + int(total_images * val_ratio)

train_files = valid_images[:train_end]
val_files = valid_images[train_end:val_end]
test_files = valid_images[val_end:]

print(f"Total valid paired images: {total_images}")
print(f"Train split: {len(train_files)}")
print(f"Validation split: {len(val_files)}")
print(f"Test split: {len(test_files)}")

def copy_files(file_list, split_name):
    for img_name in file_list:
        stem = Path(img_name).stem  # filename without extension
        
        # Source paths
        src_img = image_dir / img_name
        src_annot = annot_dir / f"{stem}.json"
        
        # Destination paths
        dst_img = output_dir / split_name / "images" / img_name
        dst_annot = output_dir / split_name / "bbox_annotation" / f"{stem}.json"
        
        # Copy image
        if src_img.exists():
            shutil.copy(src_img, dst_img)
            
        # Copy exact-matching JSON annotation[cite: 4]
        if src_annot.exists():
            shutil.copy(src_annot, dst_annot)

# Execute copying for each split
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("Dataset successfully split and files copied!")

# Verify the split counts
for split in splits:
    img_count = len(os.listdir(output_dir / split / "images"))
    annot_count = len(os.listdir(output_dir / split / "bbox_annotation"))
    print(f"[{split.upper()}] Images: {img_count}, Annotations: {annot_count}")