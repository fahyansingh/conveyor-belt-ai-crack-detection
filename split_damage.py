import os
import shutil
import random

BASE = r"damage_dataset"

IMAGE_DIR = os.path.join(BASE, "images", "train")
VAL_DIR = os.path.join(BASE, "images", "val")
LABEL_DIR = os.path.join(BASE, "labels")

os.makedirs(VAL_DIR, exist_ok=True)

# Get all crack images
images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

images.sort()

print("Total images found:", len(images))

# Random but reproducible
random.seed(42)
random.shuffle(images)

# 20 images for validation
val_images = images[:20]

for image_name in val_images:

    # Move image
    src_image = os.path.join(IMAGE_DIR, image_name)
    dst_image = os.path.join(VAL_DIR, image_name)

    shutil.move(src_image, dst_image)

    # Corresponding JSON
    base_name = os.path.splitext(image_name)[0]
    json_name = base_name + ".json"

    src_json = os.path.join(LABEL_DIR, json_name)

    if os.path.exists(src_json):
        # Move JSON into a temporary validation folder
        val_label_dir = os.path.join(LABEL_DIR, "val")
        os.makedirs(val_label_dir, exist_ok=True)

        dst_json = os.path.join(val_label_dir, json_name)

        shutil.move(src_json, dst_json)

print()
print("================================")
print("SPLIT COMPLETE")
print("================================")
print("Training images :", len(images) - 20)
print("Validation images:", len(val_images))
print("================================")