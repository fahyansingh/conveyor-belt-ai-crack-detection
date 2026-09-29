import os
import random
import shutil

BASE = r"damage_dataset\images\hole"

TRAIN_DIR = r"damage_dataset\images\hole_train"
VAL_DIR = r"damage_dataset\images\hole_val"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

images = [
    f for f in os.listdir(BASE)
    if f.lower().endswith(".jpg")
]

images.sort()

random.seed(42)
random.shuffle(images)

train_images = images[:64]
val_images = images[64:]

for img in train_images:
    shutil.copy2(
        os.path.join(BASE, img),
        os.path.join(TRAIN_DIR, img)
    )

for img in val_images:
    shutil.copy2(
        os.path.join(BASE, img),
        os.path.join(VAL_DIR, img)
    )

print(f"Total images: {len(images)}")
print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")