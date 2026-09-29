import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Existing YOLO training data
SOURCE_IMAGES = os.path.join(
    BASE_DIR, "yolo_dataset", "images", "train"
)
SOURCE_LABELS = os.path.join(
    BASE_DIR, "yolo_dataset", "labels", "train"
)

# Validation folders
VAL_IMAGES = os.path.join(
    BASE_DIR, "yolo_dataset", "images", "val"
)
VAL_LABELS = os.path.join(
    BASE_DIR, "yolo_dataset", "labels", "val"
)

os.makedirs(VAL_IMAGES, exist_ok=True)
os.makedirs(VAL_LABELS, exist_ok=True)

# Reproducible random split
random.seed(42)

# Get all training images
images = [
    f for f in os.listdir(SOURCE_IMAGES)
    if f.lower().endswith(".jpg")
]

# Shuffle
random.shuffle(images)

# 20% validation
val_count = round(len(images) * 0.20)

val_images = images[:val_count]

# Copy validation images + labels
for image_name in val_images:

    label_name = os.path.splitext(image_name)[0] + ".txt"

    image_source = os.path.join(
        SOURCE_IMAGES, image_name
    )

    label_source = os.path.join(
        SOURCE_LABELS, label_name
    )

    image_dest = os.path.join(
        VAL_IMAGES, image_name
    )

    label_dest = os.path.join(
        VAL_LABELS, label_name
    )

    shutil.copy2(image_source, image_dest)

    if os.path.exists(label_source):
        shutil.copy2(label_source, label_dest)

print("==============================")
print("VALIDATION SPLIT CREATED")
print("==============================")
print(f"Training images : {len(images)}")
print(f"Validation images : {len(val_images)}")
print("==============================")