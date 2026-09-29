import json
from pathlib import Path
import shutil

BASE = Path(r"C:\Users\Fahyan Singh\Desktop\SIH_Gap_Measurement")

ANN_DIR = BASE / "gap_annotations"
DATASET = BASE / "gap_dataset"

# Train / validation files already copied
splits = {
    "train": DATASET / "images" / "train",
    "val": DATASET / "images" / "val",
}

for split, image_dir in splits.items():

    label_dir = DATASET / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_dir.glob("*.jpg"):

        json_path = ANN_DIR / f"{image_path.stem}.json"

        if not json_path.exists():
            print(f"JSON missing: {json_path.name}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        width = data["imageWidth"]
        height = data["imageHeight"]

        yolo_lines = []

        for shape in data["shapes"]:

            if shape["label"] != "gap":
                continue

            points = shape["points"]

            normalized = []

            for x, y in points:
                normalized.append(x / width)
                normalized.append(y / height)

            # class 0 = gap
            line = "0 " + " ".join(f"{v:.6f}" for v in normalized)
            yolo_lines.append(line)

        output = label_dir / f"{image_path.stem}.txt"

        with open(output, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        print(f"Converted: {split}/{image_path.name}")

print("\nConversion complete!")