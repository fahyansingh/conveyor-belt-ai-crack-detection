import cv2
import os

CAMERA_INDEX = 2
SAVE_DIR = r"damage_dataset\images\hole"

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera open nahi hua.")
    exit()

count = len([
    f for f in os.listdir(SAVE_DIR)
    if f.lower().endswith(".jpg")
])

print("Camera ready.")
print("S = Capture image")
print("Q = Exit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame nahi mil raha.")
        break

    cv2.imshow("Hole Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        count += 1

        filename = f"hole_{count:03d}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(filepath, frame)

        print(f"Saved: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()