from ultralytics import YOLO
import cv2
import numpy as np
import os

# ==========================================
# SETTINGS
# ==========================================

# GAP SEGMENTATION MODEL
GAP_MODEL_PATH = r"runs\segment\train\weights\best.pt"

# JOINT DETECTION MODEL
JOINT_MODEL_PATH = r"runs\detect\train\weights\best.pt"

# Test image
IMAGE_PATH = r"gap_dataset\images\test\7mm_20.jpg"

# Camera calibration
PIXELS_PER_MM = 4.46

# Filtering
MIN_CONF = 0.70
MIN_PIXEL_GAP = 25

# Current prototype has 5 actual gap regions
MAX_GAPS = 5


# ==========================================
# LOAD MODELS
# ==========================================

print("Loading models...")

joint_model = YOLO(JOINT_MODEL_PATH)
gap_model = YOLO(GAP_MODEL_PATH)

print("Models loaded.")


# ==========================================
# READ IMAGE
# ==========================================

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Image nahi mili!")
    exit()


output = img.copy()


# ==========================================
# STEP 1 — JOINT DETECTION
# ==========================================

joint_result = joint_model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.50,
    verbose=False
)[0]


if len(joint_result.boxes) == 0:

    print("Joint detect nahi hua!")

    cv2.imshow("Gap Measurement", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    exit()


# First joint box
box = (
    joint_result
    .boxes
    .xyxy[0]
    .cpu()
    .numpy()
    .astype(int)
)

x1, y1, x2, y2 = box


# Draw joint ROI
cv2.rectangle(
    output,
    (x1, y1),
    (x2, y2),
    (255, 0, 0),
    2
)

cv2.putText(
    output,
    "JOINT",
    (x1, max(20, y1 - 8)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 0, 0),
    2
)


# ==========================================
# STEP 2 — GAP SEGMENTATION
# ==========================================

gap_result = gap_model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.25,
    verbose=False
)[0]


candidates = []


if gap_result.masks is not None:

    polygons = gap_result.masks.xy

    confidences = (
        gap_result.boxes.conf
        .cpu()
        .numpy()
    )


    # ======================================
    # PROCESS EVERY MASK
    # ======================================

    for i, polygon in enumerate(polygons):

        polygon = np.array(
            polygon,
            dtype=np.int32
        )


        # Pixel height
        y_min = polygon[:, 1].min()
        y_max = polygon[:, 1].max()

        pixel_gap = y_max - y_min


        # Confidence
        confidence = float(
            confidences[i]
        )


        # ==================================
        # FILTER
        # ==================================

        if confidence < MIN_CONF:
            continue

        if pixel_gap < MIN_PIXEL_GAP:
            continue


        candidates.append(
            {
                "polygon": polygon,
                "pixel_gap": pixel_gap,
                "confidence": confidence
            }
        )


# ==========================================
# SORT BY GAP SIZE
# ==========================================

candidates.sort(
    key=lambda x: x["pixel_gap"],
    reverse=True
)


# Prototype limit
selected = candidates[:MAX_GAPS]


# ==========================================
# MEASURE GAPS
# ==========================================

measurements = []


for i, item in enumerate(selected):

    polygon = item["polygon"]

    pixel_gap = item["pixel_gap"]

    confidence = item["confidence"]


    gap_mm = (
        pixel_gap / PIXELS_PER_MM
    )


    measurements.append(gap_mm)


    # Draw segmentation polygon
    cv2.polylines(
        output,
        [polygon],
        True,
        (0, 255, 0),
        2
    )


    # Label
    tx = int(polygon[:, 0].min())

    ty = int(polygon[:, 1].min())


    cv2.putText(
        output,
        f"{gap_mm:.2f} mm",
        (tx, max(20, ty - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )


# ==========================================
# FINAL JOINT GAP
# ==========================================

if measurements:

    mean_gap = float(
        np.mean(measurements)
    )

    max_gap = float(
        np.max(measurements)
    )

else:

    mean_gap = 0.0
    max_gap = 0.0


# ==========================================
# GAP STATUS
# ==========================================

if mean_gap <= 3.0:

    gap_status = "NORMAL"

elif mean_gap <= 5.0:

    gap_status = "WARNING"

else:

    gap_status = "CRITICAL"


# ==========================================
# DASHBOARD-STYLE OUTPUT
# ==========================================

print("\n===================================")
print("       CAMERA INSPECTION")
print("===================================")

print(
    f"Joint Gap       {mean_gap:.2f} mm"
)

print(
    f"Maximum Gap     {max_gap:.2f} mm"
)

print(
    f"Detected Gaps   {len(measurements)}"
)

print(
    f"Gap Status      {gap_status}"
)

print(
    "Vision Status   NORMAL"
)

print("===================================")


# ==========================================
# DISPLAY ON IMAGE
# ==========================================

cv2.putText(
    output,
    f"Joint Gap: {mean_gap:.2f} mm",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 0, 0),
    2
)

cv2.putText(
    output,
    f"Gap Status: {gap_status}",
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (0, 0, 255),
    2
)

cv2.putText(
    output,
    "Vision Status: NORMAL",
    (20, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (0, 150, 0),
    2
)


# ==========================================
# SHOW
# ==========================================

cv2.imshow(
    "FINAL CAMERA INSPECTION",
    output
)

print("\nPress any key to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()