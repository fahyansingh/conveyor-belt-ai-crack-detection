import os
import json
import cv2

BASE = r"damage_dataset"

TRAIN_DIR = os.path.join(BASE, "images", "hole_train")
VAL_DIR = os.path.join(BASE, "images", "hole_val")

LABEL_DIR = os.path.join(BASE, "images", "hole_labels")

OUT_TRAIN = os.path.join(BASE, "hole_yolo_labels", "train")
OUT_VAL = os.path.join(BASE, "hole_yolo_labels", "val")

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_VAL, exist_ok=True)


def convert_split(image_dir, output_dir):
    converted = 0
    missing = 0
    empty = 0

    for image_name in os.listdir(image_dir):

        if not image_name.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(image_dir, image_name)

        # Corresponding LabelMe JSON
        json_name = os.path.splitext(image_name)[0] + ".json"
        json_path = os.path.join(LABEL_DIR, json_name)

        if not os.path.exists(json_path):
            print(f"JSON missing: {json_name}")
            missing += 1
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"Image error: {image_name}")
            continue

        height, width = image.shape[:2]

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        yolo_lines = []

        for shape in data.get("shapes", []):

            label = shape.get("label", "").strip().lower()

            if label != "hole":
                continue

            points = shape.get("points", [])

            if len(points) < 2:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            xmin = max(0, min(xs))
            xmax = min(width, max(xs))
            ymin = max(0, min(ys))
            ymax = min(height, max(ys))

            box_width = xmax - xmin
            box_height = ymax - ymin

            if box_width <= 0 or box_height <= 0:
                continue

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # YOLO normalized coordinates
            x_center /= width
            y_center /= height
            box_width /= width
            box_height /= height

            # Class 0 = hole
            yolo_lines.append(
                f"0 {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )

        output_name = os.path.splitext(image_name)[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)

        if yolo_lines:
            with open(output_path, "w") as f:
                f.write("\n".join(yolo_lines))

            converted += 1
        else:
            empty += 1
            print(f"No hole annotation: {image_name}")

    print(f"Converted: {converted}")
    print(f"JSON missing: {missing}")
    print(f"Empty annotations: {empty}")


print("\n--- TRAIN ---")
convert_split(TRAIN_DIR, OUT_TRAIN)

print("\n--- VALIDATION ---")
convert_split(VAL_DIR, OUT_VAL)

print("\nYOLO conversion complete.")
print("Output:")
print(r"damage_dataset\hole_yolo_labels")