import cv2
import os
from datetime import datetime

# =========================
# SETTINGS
# =========================

CAMERA_INDEX = 2

SAVE_DIR = r"damage_dataset\images\train"

# =========================
# CREATE FOLDER
# =========================

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera open nahi hua!")
    exit()

# Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("===================================")
print(" DAMAGE DATASET CAPTURE")
print("===================================")
print("S = Save Image")
print("Q = Quit")
print("===================================")

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Camera frame nahi mila")
        break

    # Display
    display = frame.copy()

    cv2.putText(
        display,
        "S = CAPTURE    Q = EXIT",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Damage Dataset Capture", display)

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # SAVE IMAGE
    # =========================

    if key == ord('s'):

        filename = f"crack_{count + 1:03d}.jpg"

        filepath = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(filepath, frame)

        count += 1

        print(f"✅ Saved: {filename}")

    # =========================
    # EXIT
    # =========================

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print()
print(f"Total images captured: {count}")
print("Capture finished.")