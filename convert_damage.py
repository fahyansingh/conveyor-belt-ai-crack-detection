import os
import json

BASE = r"damage_dataset"

# LabelMe JSON files are here
LABEL_DIR = os.path.join(BASE, "images", "labels")

SPLITS = ["train", "val"]

for split in SPLITS:

    image_dir = os.path.join(BASE, "images", split)

    # YOLO TXT labels will be created here
    yolo_label_dir = os.path.join(BASE, "yolo_labels", split)
    os.makedirs(yolo_label_dir, exist_ok=True)

    print(f"\nProcessing {split}...")

    count = 0
    missing_json = 0

    for filename in os.listdir(image_dir):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_name = os.path.splitext(filename)[0]

        # Corresponding LabelMe JSON
        json_path = os.path.join(
            LABEL_DIR,
            image_name + ".json"
        )

        if not os.path.exists(json_path):
            print("JSON missing:", filename)
            missing_json += 1
            continue

        # Read JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        width = data["imageWidth"]
        height = data["imageHeight"]

        yolo_lines = []

        for shape in data["shapes"]:

            label = shape["label"].strip().lower()

            # Only crack
            if label != "crack":
                continue

            points = shape["points"]

            if len(points) < 2:
                continue

            # Polygon → bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            xmin = max(0, min(xs))
            xmax = min(width, max(xs))

            ymin = max(0, min(ys))
            ymax = min(height, max(ys))

            # Avoid invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            # Convert to YOLO normalized format
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            # Class 0 = crack
            yolo_lines.append(
                f"0 {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )

        # Save TXT
        txt_path = os.path.join(
            yolo_label_dir,
            image_name + ".txt"
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        count += 1

    print(f"{split}: {count} labels converted.")
    print(f"{split}: {missing_json} JSON files missing.")

print("\n================================")
print("CONVERSION COMPLETE")
print("================================")

print("\nYOLO labels created in:")
print(r"damage_dataset\yolo_labels")

print("================================")