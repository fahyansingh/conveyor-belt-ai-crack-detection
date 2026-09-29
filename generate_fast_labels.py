import os
import cv2
from ultralytics import YOLO

# =========================================================
# PATHS
# =========================================================
INPUT_DIR = r"gap_fast_dataset"

OUTPUT_DIR = r"gap_fast_dataset_pseudo"

MODEL_PATH = r"runs\segment\train\weights\best.pt"

# =========================================================
# SETTINGS
# =========================================================
CONF = 0.30
IMG_SIZE = 640
DEVICE = 0

# =========================================================
# CREATE MODEL
# =========================================================
print("Loading gap segmentation model...")

model = YOLO(MODEL_PATH)

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# PROCESS ALL GAP FOLDERS
# =========================================================
total_images = 0
total_masks = 0

classes = ["0mm", "1mm", "2mm", "3mm",
           "4mm", "5mm", "6mm", "7mm"]

for gap_class in classes:

    input_folder = os.path.join(
        INPUT_DIR,
        gap_class
    )

    output_folder = os.path.join(
        OUTPUT_DIR,
        gap_class
    )

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"\nSkipping missing folder: {input_folder}")
        continue

    images = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    images.sort()

    print()
    print("========================================")
    print(f"Processing {gap_class}")
    print(f"Images: {len(images)}")
    print("========================================")

    for i, filename in enumerate(images, 1):

        image_path = os.path.join(
            input_folder,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read: {filename}")
            continue

        results = model.predict(
            source=image,
            conf=CONF,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        output_path = os.path.join(
            output_folder,
            filename
        )

        # -------------------------------------------------
        # SAVE IMAGE
        # -------------------------------------------------
        cv2.imwrite(
            output_path,
            image
        )

        total_images += 1

        # -------------------------------------------------
        # SAVE YOLO SEGMENTATION LABEL
        # -------------------------------------------------
        label_path = os.path.join(
            output_folder,
            os.path.splitext(filename)[0] + ".txt"
        )

        mask_count = 0

        with open(label_path, "w") as f:

            if result.masks is not None:

                polygons = result.masks.xy

                for polygon in polygons:

                    if len(polygon) < 3:
                        continue

                    h, w = image.shape[:2]

                    points = []

                    for x, y in polygon:

                        x_norm = x / w
                        y_norm = y / h

                        points.append(
                            f"{x_norm:.6f}"
                        )
                        points.append(
                            f"{y_norm:.6f}"
                        )

                    line = "0 " + " ".join(points)

                    f.write(line + "\n")

                    mask_count += 1

        total_masks += mask_count

        if i % 10 == 0 or i == len(images):

            print(
                f"{gap_class}: "
                f"{i}/{len(images)} "
                f"| masks: {mask_count}"
            )

print()
print("========================================")
print("PSEUDO-LABEL GENERATION COMPLETE")
print("========================================")
print(f"Total images processed : {total_images}")
print(f"Total masks generated  : {total_masks}")
print(f"Output folder          : {OUTPUT_DIR}")
print("========================================")