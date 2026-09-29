import os
import json
import shutil

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images")
ANNOTATION_DIR = os.path.join(BASE_DIR, "dataset", "annotations")

YOLO_DIR = os.path.join(BASE_DIR, "yolo_dataset")

# Class
CLASS_ID = 0
CLASS_NAME = "joint"


# ==============================
# CREATE FOLDERS
# ==============================
train_images = os.path.join(YOLO_DIR, "images", "train")
train_labels = os.path.join(YOLO_DIR, "labels", "train")

test_images = os.path.join(YOLO_DIR, "images", "test")
test_labels = os.path.join(YOLO_DIR, "labels", "test")

for folder in [train_images, train_labels, test_images, test_labels]:
    os.makedirs(folder, exist_ok=True)


# ==============================
# PROCESS JSON FILES
# ==============================
train_count = 0
test_count = 0

json_files = [
    f for f in os.listdir(ANNOTATION_DIR)
    if f.lower().endswith(".json")
]

for json_file in json_files:

    json_path = os.path.join(ANNOTATION_DIR, json_file)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data["imagePath"]
    image_name = os.path.basename(image_name)

    image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"WARNING: Image missing -> {image_name}")
        continue

    # ==============================
    # TRAIN / TEST DECISION
    # 0-4 mm = TRAIN
    # 5-7 mm = TEST
    # ==============================
    if image_name.startswith(("5mm_", "6mm_", "7mm_")):
        split = "test"
        dest_image_dir = test_images
        dest_label_dir = test_labels
        test_count += 1
    else:
        split = "train"
        dest_image_dir = train_images
        dest_label_dir = train_labels
        train_count += 1

    # Copy image
    shutil.copy2(
        image_path,
        os.path.join(dest_image_dir, image_name)
    )

    # Image dimensions
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    yolo_lines = []

    for shape in data["shapes"]:

        if shape["label"] != CLASS_NAME:
            continue

        if shape["shape_type"] != "rectangle":
            continue

        points = shape["points"]

        x1 = min(points[0][0], points[1][0])
        y1 = min(points[0][1], points[1][1])

        x2 = max(points[0][0], points[1][0])
        y2 = max(points[0][1], points[1][1])

        # YOLO format
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height

        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        yolo_lines.append(
            f"{CLASS_ID} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    # Save YOLO label
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(dest_label_dir, label_name)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))


# ==============================
# CREATE data.yaml
# ==============================
yaml_path = os.path.join(YOLO_DIR, "data.yaml")

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(
        "path: " + YOLO_DIR.replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/test\n\n"
        "names:\n"
        "  0: joint\n"
    )


print("\n==============================")
print("YOLO DATASET CREATED")
print("==============================")
print(f"Train images : {train_count}")
print(f"Test images  : {test_count}")
print(f"Dataset path : {YOLO_DIR}")
print("==============================")