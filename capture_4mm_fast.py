import cv2
import os
import time
from ultralytics import YOLO

CAMERA_INDEX = 2

SAVE_DIR = r"gap_fast_dataset\4mm"
JOINT_MODEL_PATH = r"runs\detect\train\weights\best.pt"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60

JOINT_CONF = 0.30
SAVE_INTERVAL = 0.15

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading joint detector...")
joint_model = YOLO(JOINT_MODEL_PATH)

print("Starting camera...")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print()
print("========================================")
print("4mm FAST CONVEYOR DATASET CAPTURE")
print("========================================")
print("Sample       : 4 mm")
print("Save folder  :", SAVE_DIR)
print()
print("SPACE = Start / Stop capturing")
print("Q     = Exit")
print("========================================")

capturing = False
last_save_time = 0

count = len([
    f for f in os.listdir(SAVE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame error.")
        break

    display = frame.copy()

    results = joint_model.predict(
        source=frame,
        conf=JOINT_CONF,
        imgsz=640,
        device=0,
        verbose=False
    )

    joint_detected = False

    if len(results) > 0:

        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:

            joint_detected = True

            for box in result.boxes.xyxy.cpu().numpy():

                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    current_time = time.time()

    if capturing and joint_detected:

        if current_time - last_save_time >= SAVE_INTERVAL:

            count += 1

            filename = os.path.join(
                SAVE_DIR,
                f"4mm_{count:04d}.jpg"
            )

            cv2.imwrite(filename, frame)

            last_save_time = current_time

            print(f"Saved: {filename}")

    status = "CAPTURING" if capturing else "PAUSED"

    cv2.putText(
        display,
        f"4mm | {status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Images: {count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if joint_detected:

        cv2.putText(
            display,
            "JOINT DETECTED",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            display,
            "JOINT NOT DETECTED",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "4mm Fast Conveyor Capture",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):

        capturing = not capturing

        if capturing:
            print("\n>>> CAPTURE STARTED <<<\n")
        else:
            print("\n>>> CAPTURE PAUSED <<<\n")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("========================================")
print("CAPTURE FINISHED")
print(f"Total 4mm images: {count}")
print(f"Folder: {SAVE_DIR}")
print("========================================")