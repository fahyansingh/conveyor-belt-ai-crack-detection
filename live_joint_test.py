import cv2
from ultralytics import YOLO
import time

MODEL_PATH = r"runs\detect\train\weights\best.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("DroidCam open nahi hua!")
    exit()

print("LIVE JOINT TEST")
print("Press Q to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Smaller image = faster CPU inference
    small = cv2.resize(frame, (640, 480))

    results = model.predict(
        source=small,
        imgsz=640,
        conf=0.50,
        verbose=False
    )

    result = results[0]

    if len(result.boxes) > 0:

        for box in result.boxes.xyxy.cpu().numpy():

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                small,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

    cv2.imshow(
        "LIVE JOINT TEST",
        small
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()