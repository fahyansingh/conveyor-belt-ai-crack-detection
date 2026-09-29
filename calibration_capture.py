import cv2
import os

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Camera open nahi hua!")
    exit()

os.makedirs("calibration", exist_ok=True)

print("S = photo capture")
print("Q = exit")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Frame nahi mil raha!")
        break

    cv2.imshow("Calibration Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        path = "calibration/calibration_5mm.jpg"
        cv2.imwrite(path, frame)
        print(f"Photo saved: {path}")

    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()