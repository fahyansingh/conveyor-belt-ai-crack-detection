import cv2
import os
import time
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 2

JOINT_MODEL_PATH = r"runs\detect\train\weights\best.pt"

SAVE_DIR = r"gap_fast_dataset\6mm"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

JOINT_CONF = 0.30

# Minimum time between saved frames
SAVE_INTERVAL = 0.15


# ============================================================
# FOLDER
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# LOAD JOINT MODEL
# ============================================================

print()
print("Loading joint detector...")

model = YOLO(JOINT_MODEL_PATH)

print("Joint model loaded.")
print()


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

camera.set(
    cv2.CAP_PROP_FPS,
    60
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not camera.isOpened():

    print("ERROR: DroidCam not found.")
    exit()


# ============================================================
# EXISTING IMAGE COUNT
# ============================================================

count = len([
    f
    for f in os.listdir(SAVE_DIR)
    if f.lower().endswith(".jpg")
])


# ============================================================
# STATE
# ============================================================

capturing = False

last_save_time = 0


# ============================================================
# START MESSAGE
# ============================================================

print("========================================")
print("AI ASSISTED FAST DATASET CAPTURE")
print("========================================")
print("Sample       : 6 mm")
print("Folder       :", SAVE_DIR)
print()
print("SPACE = Start / Stop")
print("Q     = Exit")
print()
print("Only frames containing the JOINT")
print("will be saved.")
print("========================================")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:
        continue


    # ========================================================
    # JOINT DETECTION
    # ========================================================

    result = model.predict(
        source=frame,
        imgsz=640,
        conf=JOINT_CONF,
        device=0,
        verbose=False
    )[0]


    joint_detected = False


    if result.boxes is not None:

        if len(result.boxes) > 0:

            joint_detected = True


    # ========================================================
    # DRAW JOINT BOX
    # ========================================================

    display = frame.copy()


    if joint_detected:

        for box in result.boxes.xyxy.cpu().numpy():

            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


    # ========================================================
    # STATUS
    # ========================================================

    if capturing:

        if joint_detected:

            status = "CAPTURING JOINT"

        else:

            status = "WAITING FOR JOINT"

    else:

        status = "READY"


    cv2.putText(
        display,
        f"6mm | {status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"Images: {count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE ONLY WHEN JOINT IS DETECTED
    # ========================================================

    current_time = time.time()


    if capturing and joint_detected:

        if (
            current_time - last_save_time
            >= SAVE_INTERVAL
        ):

            filename = os.path.join(
                SAVE_DIR,
                f"6mm_{count + 1:04d}.jpg"
            )


            cv2.imwrite(
                filename,
                frame
            )


            count += 1

            last_save_time = current_time


            print(
                f"[SAVED] {filename}"
            )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "6mm Fast Conveyor Capture",
        display
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord(" "):

        capturing = not capturing


        if capturing:

            print()
            print(
                "[START] Automatic joint capture ON"
            )

            print(
                "[INFO] Only detected joint frames "
                "will be saved."
            )

            print()


        else:

            print()
            print(
                "[STOP] Capture paused."
            )

            print()


    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()


print()

print("========================================")
print("CAPTURE FINISHED")
print("Total saved images:", count)
print("Folder:", SAVE_DIR)
print("========================================")