import cv2
import threading
import time
import numpy as np
from flask import Flask, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 2

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 60

# ============================================================
# MODEL PATHS
# ============================================================

JOINT_MODEL_PATH = r"runs\detect\train\weights\best.pt"
GAP_MODEL_PATH = r"runs\segment\train\weights\best.pt"

CRACK_MODEL_PATH = r"runs\detect\runs\crack_detection_final\weights\best.pt"
HOLE_MODEL_PATH = r"runs\detect\runs\hole_detection_final\weights\best.pt"


# ============================================================
# LOAD MODELS
# ============================================================

print("\n========================================")
print("SIH CONVEYOR VISION SERVER")
print("========================================")

print("[INFO] Loading Joint model...")
joint_model = YOLO(JOINT_MODEL_PATH)

print("[INFO] Loading Gap model...")
gap_model = YOLO(GAP_MODEL_PATH)

print("[INFO] Loading Crack model...")
crack_model = YOLO(CRACK_MODEL_PATH)

print("[INFO] Loading Hole model...")
hole_model = YOLO(HOLE_MODEL_PATH)

print("[INFO] Using NVIDIA GPU...")


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
    CAMERA_FPS
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not camera.isOpened():

    print("[ERROR] DroidCam could not be opened")

    raise RuntimeError(
        "DroidCam camera not found"
    )


print("[OK] DroidCam connected")
print("[INFO] Camera index:", CAMERA_INDEX)
print(
    "[INFO] Resolution:",
    FRAME_WIDTH,
    "x",
    FRAME_HEIGHT
)
print("[INFO] Requested FPS:", CAMERA_FPS)


# ============================================================
# CALIBRATION
# ============================================================

CAL_SLOPE = 4.4434
CAL_OFFSET = 1.4513


# ============================================================
# LIVE SETTINGS
# ============================================================

JOINT_CONF = 0.30
GAP_CONF = 0.15

CRACK_CONF = 0.30
HOLE_CONF = 0.30

MIN_PIXEL_GAP = 5
MAX_GAPS = 5
MIN_GAP_CONF = 0.50


JOINT_LOST_FRAMES = 8


# ============================================================
# DAMAGE DETECTION ROI
# ============================================================

ROI_X1 = 100
ROI_Y1 = 20
ROI_X2 = 540
ROI_Y2 = 480


# ============================================================
# SHARED FRAMES
# ============================================================

latest_frame = None
processed_frame = None

frame_lock = threading.Lock()
processed_lock = threading.Lock()

camera_running = True


# ============================================================
# STORED DISPLAY STATE
# ============================================================

last_gap = 0.0

last_status = "WAITING"

gap_detected_for_current_joint = False

joint_was_seen = False

joint_lost_count = 0


# ============================================================
# GAP HISTORY FOR CURRENT JOINT
# ============================================================

gap_history = []

FINAL_GAP_COUNT = 3


# ============================================================
# DAMAGE DISPLAY STATE
# ============================================================

crack_detected = False
hole_detected = False


# ============================================================
# VISION API LOCK
# ============================================================

vision_lock = threading.Lock()


# ============================================================
# GAP STATUS
# ============================================================

def get_gap_status(gap_mm):

    if gap_mm <= 3.0:

        return "NORMAL"

    elif gap_mm <= 5.0:

        return "WARNING"

    else:

        return "CRITICAL"


# ============================================================
# CAMERA CAPTURE THREAD
#
# IMPORTANT:
# Only newest frame is kept.
# Old frames are discarded.
# ============================================================

def camera_capture_loop():

    global latest_frame

    print(
        "[INFO] Camera capture thread started"
    )

    while camera_running:

        success, frame = camera.read()

        if not success:
            time.sleep(0.005)
            continue

        cv2.imwrite("live_test.jpg", frame)

        with frame_lock:
            latest_frame = frame


# ============================================================
# AI PROCESSING THREAD
# ============================================================

def ai_processing_loop():

    global latest_frame
    global processed_frame

    global last_gap
    global last_status

    global gap_detected_for_current_joint

    global joint_was_seen
    global joint_lost_count

    global crack_detected
    global hole_detected

    print(
        "[INFO] AI processing thread started"
    )

    while camera_running:

        # ====================================================
        # GET LATEST FRAME
        # ====================================================

        with frame_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()

        if frame is None:

            time.sleep(0.005)

            continue


        # ====================================================
        # JOINT DETECTION
        # ====================================================

        joint_result = joint_model.predict(

            source=frame,

            imgsz=640,

            conf=JOINT_CONF,

            device=0,

            verbose=False

        )[0]

        joint_detected = False

        if joint_result.boxes is not None:

            if len(joint_result.boxes) > 0:

                joint_detected = True


        # ====================================================
        # NEW PHYSICAL JOINT LOGIC
        # ====================================================

        new_physical_joint = False

        if joint_detected:

            if not joint_was_seen:

                new_physical_joint = True

            elif joint_lost_count >= JOINT_LOST_FRAMES:

                new_physical_joint = True

            joint_was_seen = True
            joint_lost_count = 0

        else:

            if joint_was_seen:

                joint_lost_count += 1

                # DO NOT RESET GAP HERE.
                #
                # Joint leaving camera should keep
                # previous measurement.

                if joint_lost_count > JOINT_LOST_FRAMES:

                    joint_was_seen = False


        # ====================================================
        # RESET ONLY WHEN NEW JOINT ARRIVES
        # ====================================================

        # ====================================================
        # RESET ONLY WHEN NEW JOINT ARRIVES
        # ====================================================

        if new_physical_joint:

            gap_history.clear()

            with vision_lock:

                last_gap = 0.0

                last_status = "WAITING"

                gap_detected_for_current_joint = False

        # ====================================================
        # GAP SEGMENTATION
        #
        # FULL FRAME
        #
        # NO ROI
        #
        # NO CROP
        # ====================================================

        gap_result = gap_model.predict(

            source=frame,

            imgsz=640,

            conf=GAP_CONF,

            device=0,

            verbose=False

        )[0]

        current_gap_values = []


        # ====================================================
        # GAP MASKS
        # ====================================================

        if gap_result.masks is not None:

            polygons = gap_result.masks.xy

            if polygons is not None:

                detections = []

                for i, polygon in enumerate(polygons):

                    if polygon is None:

                        continue

                    if len(polygon) < 3:

                        continue


                    # ----------------------------------------
                    # ORIGINAL IMAGE COORDINATES
                    # ----------------------------------------

                    x_coords = polygon[:, 0]
                    y_coords = polygon[:, 1]

                    pixel_height = float(
                        y_coords.max()
                        -
                        y_coords.min()
                    )


                    # ----------------------------------------
                    # MINIMUM GAP SIZE
                    # ----------------------------------------

                    if pixel_height < MIN_PIXEL_GAP:

                        continue


                    # ----------------------------------------
                    # CONFIDENCE
                    # ----------------------------------------

                    confidence = 1.0

                    if gap_result.boxes is not None:

                        if i < len(gap_result.boxes):

                            confidence = float(
                                gap_result.boxes.conf[i]
                            )

                    # Ignore low-confidence detections
                    if confidence < MIN_GAP_CONF:
                        continue

                    detections.append(
                        (
                            pixel_height,
                            confidence,
                            polygon
                        )
                    )


                # ====================================================
                # SORT BY CONFIDENCE
                # ====================================================

                detections.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                detections = detections[:MAX_GAPS]


                # ====================================================
                # PIXELS → MM
                # ====================================================

                for (
                    pixel_height,
                    confidence,
                    polygon
                ) in detections:

                    gap_mm = (
                        pixel_height
                        -
                        CAL_OFFSET
                    ) / CAL_SLOPE

                    if gap_mm < 0:

                        gap_mm = 0.0

                    current_gap_values.append(
                        gap_mm
                    )


                    # ====================================================
                    # GREEN CURRENT-FRAME GAP BOX
                    # ====================================================

                    x1 = int(
                        polygon[:, 0].min()
                    )

                    y1 = int(
                        polygon[:, 1].min()
                    )

                    x2 = int(
                        polygon[:, 0].max()
                    )

                    y2 = int(
                        polygon[:, 1].max()
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )


        # ====================================================
        # UPDATE STORED GAP
        # ====================================================

        # ====================================================
        # ====================================================
        # UPDATE STORED GAP
        # ====================================================

        if joint_detected and len(current_gap_values) > 0:

            measured_gap = (
                sum(current_gap_values)
                /
                len(current_gap_values)
            )

            # Store every valid measurement
            gap_history.append(measured_gap)

            # Take highest 3 measurements
            highest_values = sorted(
                gap_history,
                reverse=True
            )[:FINAL_GAP_COUNT]

            # Average of highest measurements
            final_gap = (
                sum(highest_values)
                /
                len(highest_values)
            )

            with vision_lock:

                last_gap = final_gap

                last_status = get_gap_status(
                    final_gap
                )

                gap_detected_for_current_joint = True


        # ====================================================
        # CRACK DETECTION
        #
        # Uses fixed ROI.
        # ====================================================

        roi = frame[
            ROI_Y1:ROI_Y2,
            ROI_X1:ROI_X2
        ]

        crack_result = crack_model.predict(

            source=roi,

            imgsz=640,

            conf=CRACK_CONF,

            device=0,

            verbose=False

        )[0]

        current_crack_detected = False

        if crack_result.boxes is not None:

            if len(crack_result.boxes) > 0:

                current_crack_detected = True


        # ====================================================
        # DRAW CRACK DETECTIONS
        # ====================================================

        if crack_result.boxes is not None:

            for box in crack_result.boxes:

                coords = box.xyxy[0].cpu().numpy()

                x1 = int(coords[0]) + ROI_X1
                y1 = int(coords[1]) + ROI_Y1
                x2 = int(coords[2]) + ROI_X1
                y2 = int(coords[3]) + ROI_Y1

                confidence = float(
                    box.conf[0]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"CRACK {confidence:.2f}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )


        # ====================================================
        # HOLE DETECTION
        #
        # Uses fixed ROI.
        # ====================================================

        hole_result = hole_model.predict(

            source=roi,

            imgsz=640,

            conf=HOLE_CONF,

            device=0,

            verbose=False

        )[0]

        current_hole_detected = False

        if hole_result.boxes is not None:

            if len(hole_result.boxes) > 0:

                current_hole_detected = True


        # ====================================================
        # DRAW HOLE DETECTIONS
        # ====================================================

        if hole_result.boxes is not None:

            for box in hole_result.boxes:

                coords = box.xyxy[0].cpu().numpy()

                x1 = int(coords[0]) + ROI_X1
                y1 = int(coords[1]) + ROI_Y1
                x2 = int(coords[2]) + ROI_X1
                y2 = int(coords[3]) + ROI_Y1

                confidence = float(
                    box.conf[0]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"HOLE {confidence:.2f}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2
                )


        # ====================================================
        # STORE DAMAGE STATE
        # ====================================================

        with vision_lock:

            crack_detected = current_crack_detected
            hole_detected = current_hole_detected


        # ====================================================
        # DISPLAY
        # ====================================================

        with vision_lock:

            display_gap = last_gap
            display_status = last_status
            display_gap_detected = (
                gap_detected_for_current_joint
            )

            display_crack = crack_detected
            display_hole = hole_detected


        cv2.putText(

            frame,

            f"Joint Gap: {display_gap:.2f} mm",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Status: {display_status}",

            (20, 65),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (255, 255, 255),

            2

        )


        if display_gap_detected:

            gap_text = "Gap: DETECTED"

        else:

            gap_text = "Gap: NOT DETECTED"


        cv2.putText(

            frame,

            gap_text,

            (20, 95),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (255, 255, 255),

            2

        )


        # ====================================================
        # DAMAGE STATUS DISPLAY
        # ====================================================

        if display_crack:

            crack_text = "Crack: DETECTED"

        else:

            crack_text = "Crack: NOT DETECTED"


        if display_hole:

            hole_text = "Hole: DETECTED"

        else:

            hole_text = "Hole: NOT DETECTED"


        cv2.putText(

            frame,

            crack_text,

            (20, 125),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 0, 255) if display_crack else (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            hole_text,

            (20, 155),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 0, 0) if display_hole else (255, 255, 255),

            2

        )


        # ====================================================
        # SAVE PROCESSED FRAME
        # ====================================================

        with processed_lock:

            processed_frame = frame.copy()


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    while camera_running:

        with processed_lock:

            if processed_frame is None:

                frame = None

            else:

                frame = processed_frame.copy()

        if frame is None:

            time.sleep(0.01)

            continue

        success, buffer = cv2.imencode(

            ".jpg",

            frame,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                85
            ]

        )

        if not success:

            continue

        frame_bytes = buffer.tobytes()

        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            +
            frame_bytes
            +
            b"\r\n"

        )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return """
    <h2>SIH Conveyor Vision Server</h2>

    <p>DroidCam + AI Vision</p>

    <p>
        <a href="/video_feed">
        Open Live Camera Feed
        </a>
    </p>

    <p>
        <a href="/vision_data">
        Open Live Vision Data
        </a>
    </p>
    """


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=
        "multipart/x-mixed-replace; boundary=frame"

    )


# ============================================================
# VISION DATA API
#
# Dashboard reads live AI results from here.
# ============================================================

@app.route("/vision_data")
def vision_data():

    with vision_lock:

        data = {
            "joint_detection":
              "DETECTED"
              if joint_was_seen
             else
              "NOT DETECTED",
              
            "joint_gap_mm": round(last_gap, 2),
            "gap_status": last_status,
            "gap_detected": gap_detected_for_current_joint,

            "crack_detection":
                "DETECTED"
                if crack_detected
                else
                "NOT DETECTED",

            "hole_detection":
                "DETECTED"
                if hole_detected
                else
                "NOT DETECTED",

            "vision_status": "NORMAL"
        }

    response = jsonify(data)

    response.headers[
        "Access-Control-Allow-Origin"
    ] = "*"

    return response


# ============================================================
# START THREADS
# ============================================================

camera_thread = threading.Thread(

    target=camera_capture_loop,

    daemon=True

)

ai_thread = threading.Thread(

    target=ai_processing_loop,

    daemon=True

)

camera_thread.start()
ai_thread.start()


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "========================================"
    )

    print(
        "SIH CONVEYOR VISION SERVER"
    )

    print(
        "========================================"
    )

    print(
        "Camera       : DroidCam"
    )

    print(
        "Index        :",
        CAMERA_INDEX
    )

    print(
        "Resolution   :",
        FRAME_WIDTH,
        "x",
        FRAME_HEIGHT
    )

    print(
        "Requested FPS:",
        CAMERA_FPS
    )

    print(
        "AI Device    : RTX 3050 GPU"
    )

    print(
        "Models       : Joint + Gap + Crack + Hole"
    )

    print(
        "========================================"
    )

    print(
        "Video:"
    )

    print(
        "http://127.0.0.1:5000/video_feed"
    )

    print(
        "Vision Data:"
    )

    print(
        "http://127.0.0.1:5000/vision_data"
    )

    print(
        "========================================"
    )

    print()


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True

    )