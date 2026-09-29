import cv2
from ultralytics import YOLO

CAMERA_INDEX = 2
MODEL_PATH = r"runs\detect\runs\crack_detection_final\weights\best.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: DroidCam open nahi hua.")
    exit()

print("Live Crack Detection Started")
print("Q = Exit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame nahi mila.")
        break

    frame = cv2.resize(frame, (640, 480))

    results = model.predict(
        source=frame,
        imgsz=320,
        conf=0.30,
        device=0,
        verbose=False
    )

    result = results[0]

    annotated = frame.copy()

    # Belt ROI — same concept as hole detection
    ROI_X1 = 100
    ROI_Y1 = 20
    ROI_X2 = 540
    ROI_Y2 = 480

    # Temporary ROI boundary
    cv2.rectangle(
        annotated,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        (255, 255, 0),
        2
    )

    for box in result.boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Accept only detections inside belt ROI
        if (
            ROI_X1 <= center_x <= ROI_X2
            and ROI_Y1 <= center_y <= ROI_Y2
            and conf >= 0.30
        ):

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # RED box for crack
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated,
                f"CRACK {conf:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    cv2.imshow("LIVE CRACK DETECTION", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()