import cv2
import serial
import json
import threading

from ultralytics import YOLO


# =====================================================
# ESP32 SERIAL
# =====================================================

COM_PORT = "COM4"
BAUD_RATE = 115200

sensor_data = {
    "current": 0.0,
    "rpm": 0.0,
    "vibration": 0.0,
    "temperature": 0.0,
    "ir_status": 1,
    "belt_discontinuity": False,
    "sequence_id": 0
}

sensor_lock = threading.Lock()


# =====================================================
# SERIAL READER
# =====================================================

def read_sensor_data():

    global sensor_data

    try:

        ser = serial.Serial(
            COM_PORT,
            BAUD_RATE,
            timeout=1
        )

        print("ESP32 connected:", COM_PORT)

        while True:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            if not line.startswith("{"):
                continue

            try:

                data = json.loads(line)

                with sensor_lock:

                    sensor_data["current"] = float(
                        data.get("current", 0)
                    )

                    sensor_data["rpm"] = float(
                        data.get("rpm", 0)
                    )

                    sensor_data["vibration"] = float(
                        data.get("vibration", 0)
                    )

                    sensor_data["temperature"] = float(
                        data.get("temperature", 0)
                    )

                    sensor_data["ir_status"] = int(
                        data.get("ir_status", 1)
                    )

                    sensor_data["belt_discontinuity"] = bool(
                        data.get(
                            "belt_discontinuity",
                            False
                        )
                    )

                    sensor_data["sequence_id"] = int(
                        data.get("sequence_id", 0)
                    )

            except json.JSONDecodeError:

                continue

    except Exception as e:

        print("ESP32 SERIAL ERROR:")
        print(e)


# =====================================================
# START SENSOR THREAD
# =====================================================

sensor_thread = threading.Thread(
    target=read_sensor_data,
    daemon=True
)

sensor_thread.start()


# =====================================================
# MODEL PATHS
# =====================================================

JOINT_MODEL_PATH = r"runs\detect\train\weights\best.pt"

GAP_MODEL_PATH = r"runs\segment\train\weights\best.pt"


# =====================================================
# GAP CALIBRATION
# =====================================================

PIXELS_PER_MM = 4.46

# Empirical calibration:
#
# pixels = 4.4434 × mm + 1.4513
#
# mm = (pixels - 1.4513) / 4.4434

CAL_SLOPE = 4.4434
CAL_OFFSET = 1.4513


# =====================================================
# GAP FILTERS
# =====================================================

MIN_CONF = 0.30

MIN_PIXEL_GAP = 5

MAX_GAPS = 5


# =====================================================
# JOINT TRACKING
# =====================================================

# Joint must disappear for this many consecutive
# frames before the system considers the next
# detected joint as a new physical joint.

JOINT_LOST_FRAMES = 8


# =====================================================
# LOAD MODELS
# =====================================================

print("Loading Joint Model...")

joint_model = YOLO(JOINT_MODEL_PATH)

print("Loading Gap Model...")

gap_model = YOLO(GAP_MODEL_PATH)

print("Models loaded.")


# =====================================================
# DROIDCAM
# =====================================================

cap = cv2.VideoCapture(
    2,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("DroidCam open nahi hua!")

    exit()


print()
print("LIVE GAP MEASUREMENT")
print("Press Q to exit")
print()


# =====================================================
# STORED RESULT
# =====================================================

# Last measured gap.
#
# IMPORTANT:
# This is NOT reset when the joint leaves.
#
# It is reset ONLY when the NEXT joint is detected.

last_gap = 0.0

last_status = "NO GAP"


# =====================================================
# CURRENT GAP DETECTION
# =====================================================

# Green boxes belong only to current frame.

detected_boxes = []

# Current frame gap detection

gap_detected_now = False


# =====================================================
# CURRENT JOINT GAP STATUS
# =====================================================

# This status belongs to the CURRENT PHYSICAL JOINT.
#
# Once detected, it remains DETECTED until
# the next physical joint arrives.

current_joint_gap_status = "NOT DETECTED"


# =====================================================
# JOINT STATE
# =====================================================

# True = a physical joint is currently active
# or has recently been tracked.

joint_active = False

# Number of consecutive frames without joint.

joint_lost_count = 0


# =====================================================
# MEASUREMENT STATE
# =====================================================

# False = current joint has not been measured.
# True = current joint has already been measured.

measurement_done = False


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # =================================================
    # RESIZE
    # =================================================

    frame = cv2.resize(
        frame,
        (640, 480)
    )


    # =================================================
    # JOINT DETECTION
    # =================================================

    joint_results = joint_model.predict(

        source=frame,

        imgsz=320,

        conf=0.30,

        verbose=False
    )

    joint_result = joint_results[0]


    # =================================================
    # CURRENT JOINT
    # =================================================

    current_joint_box = None


    if len(joint_result.boxes) > 0:

        # Highest confidence joint

        confidences = (
            joint_result.boxes.conf
            .cpu()
            .numpy()
        )

        best_index = confidences.argmax()

        box = (
            joint_result.boxes.xyxy
            .cpu()
            .numpy()[best_index]
        )

        x1, y1, x2, y2 = map(
            int,
            box
        )

        current_joint_box = (
            x1,
            y1,
            x2,
            y2
        )


    # =================================================
    # JOINT PRESENT
    # =================================================

    if current_joint_box is not None:

        # Joint is currently visible

        joint_lost_count = 0


        # =================================================
        # NEW PHYSICAL JOINT
        # =================================================

        if not joint_active:

            print()
            print("==============================")
            print("NEW JOINT DETECTED")
            print("==============================")


            joint_active = True

            measurement_done = False


            # -------------------------------------------------
            # IMPORTANT:
            #
            # RESET OLD DATA ONLY HERE.
            #
            # NOT when previous joint leaves.
            # -------------------------------------------------

            last_gap = 0.0

            last_status = "NO GAP"

            current_joint_gap_status = "NOT DETECTED"


        # =================================================
        # GAP SEGMENTATION
        # =================================================

        gap_results = gap_model.predict(

            source=frame,

            imgsz=320,

            conf=MIN_CONF,

            verbose=False
        )

        gap_result = gap_results[0]


        # =================================================
        # RESET CURRENT FRAME GAP
        # =================================================

        detected_boxes = []

        gap_measurements = []


        # =================================================
        # READ GAP MASKS
        # =================================================

        if gap_result.masks is not None:

            masks = (
                gap_result.masks.data
                .cpu()
                .numpy()
            )

            confidences = (
                gap_result.boxes.conf
                .cpu()
                .numpy()
            )

            boxes = (
                gap_result.boxes.xyxy
                .cpu()
                .numpy()
            )


            candidates = []


            # =================================================
            # FILTER CANDIDATES
            # =================================================

            for i in range(len(masks)):

                conf = float(
                    confidences[i]
                )

                x1, y1, x2, y2 = boxes[i]


                pixel_height = (
                    y2 - y1
                )


                # Confidence filter

                if conf < MIN_CONF:
                    continue


                # Pixel size filter

                if pixel_height < MIN_PIXEL_GAP:
                    continue


                candidates.append(

                    (
                        pixel_height,
                        conf,
                        masks[i],
                        (
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        )
                    )
                )


            # =================================================
            # SORT BY GAP SIZE
            # =================================================

            candidates.sort(

                key=lambda x: x[0],

                reverse=True
            )


            candidates = candidates[:MAX_GAPS]


            # =================================================
            # CALCULATE GAP
            # =================================================

            for candidate in candidates:

                pixel_height = candidate[0]

                box = candidate[3]


                # ---------------------------------------------
                # PIXELS → MM
                # ---------------------------------------------

                gap_mm = (
                    pixel_height
                    - CAL_OFFSET
                ) / CAL_SLOPE


                if gap_mm < 0:

                    gap_mm = 0.0


                gap_measurements.append(
                    gap_mm
                )


                detected_boxes.append(

                    (
                        box,
                        gap_mm
                    )
                )


        # =================================================
        # CURRENT FRAME GAP DETECTION
        # =================================================

        if len(gap_measurements) > 0:

            gap_detected_now = True

        else:

            gap_detected_now = False


        # =================================================
        # UPDATE CURRENT JOINT GAP STATUS
        # =================================================

        if gap_detected_now:

            current_joint_gap_status = "DETECTED"


        # =================================================
        # SAVE FIRST VALID MEASUREMENT
        # =================================================

        if (
            not measurement_done
            and len(gap_measurements) > 0
        ):

            last_gap = (
                sum(gap_measurements)
                /
                len(gap_measurements)
            )


            # ---------------------------------------------
            # STATUS
            # ---------------------------------------------

            if last_gap <= 3:

                last_status = "NORMAL"

            elif last_gap <= 5:

                last_status = "WARNING"

            else:

                last_status = "CRITICAL"


            measurement_done = True


            print(
                f"New Gap: "
                f"{last_gap:.2f} mm"
            )

            print(
                f"Status: "
                f"{last_status}"
            )


    # =====================================================
    # JOINT NOT PRESENT
    # =====================================================

    else:

        # ---------------------------------------------
        # Current frame has no joint
        # ---------------------------------------------

        gap_detected_now = False

        detected_boxes = []


        if joint_active:

            joint_lost_count += 1


            # ---------------------------------------------
            # JOINT HAS LEFT CAMERA
            # ---------------------------------------------

            if (
                joint_lost_count
                >= JOINT_LOST_FRAMES
            ):

                joint_active = False

                joint_lost_count = 0

                measurement_done = False


                # =================================================
                # IMPORTANT:
                #
                # DO NOT RESET:
                #
                # last_gap
                # last_status
                # current_joint_gap_status
                #
                # They MUST remain on screen until NEXT JOINT.
                # =================================================

                print(
                    "Joint left camera."
                )

                print(
                    "Holding previous result."
                )

                print(
                    "Waiting for next joint..."
                )


        else:

            # ---------------------------------------------
            # NO JOINT HAS EVER BEEN DETECTED
            # ---------------------------------------------
            #
            # Only here should initial display be zero.
            #
            # Once a joint has been measured, its result
            # will be held until another joint arrives.

            pass


    # =================================================
    # DRAW CURRENT JOINT BOX
    # =================================================

    if current_joint_box is not None:

        x1, y1, x2, y2 = current_joint_box


        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (255, 0, 0),

            2
        )


        cv2.putText(

            frame,

            "JOINT",

            (
                x1,
                max(y1 - 8, 20)
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 0, 0),

            2
        )


    # =================================================
    # DRAW CURRENT GAP BOXES
    # =================================================

    # Green boxes are ONLY for the gap currently
    # detected by the model.
    #
    # If gap disappears from current frame:
    # green boxes disappear.
    #
    # The stored numerical result does NOT disappear.

    if gap_detected_now:

        for box, gap_mm in detected_boxes:

            x1, y1, x2, y2 = box


            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )


            label = (
                f"{gap_mm:.2f} mm"
            )


            cv2.putText(

                frame,

                label,

                (
                    x1,
                    max(y1 - 5, 20)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 255, 0),

                2
            )


    # =================================================
    # PANEL
    # =================================================

    overlay = frame.copy()


    cv2.rectangle(

        overlay,

        (10, 10),

        (300, 115),

        (0, 0, 0),

        -1
    )


    cv2.addWeighted(

        overlay,

        0.45,

        frame,

        0.55,

        0,

        frame
    )


    # =================================================
    # LINE 1
    # =================================================

    cv2.putText(

        frame,

        f"Joint Gap: {last_gap:.2f} mm",

        (20, 38),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2
    )


    # =================================================
    # LINE 2
    # =================================================

    cv2.putText(

        frame,

        f"Status: {last_status}",

        (20, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2
    )


    # =================================================
    # LINE 3
    # =================================================

    if current_joint_gap_status == "DETECTED":

        gap_text = "Gap: DETECTED"

    else:

        gap_text = "Gap: NOT DETECTED"


    cv2.putText(

        frame,

        gap_text,

        (20, 92),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2
    )


    # =================================================
    # DISPLAY
    # =================================================

    cv2.imshow(

        "LIVE GAP MEASUREMENT",

        frame
    )


    # =================================================
    # EXIT
    # =================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =====================================================
# CLEANUP
# =====================================================

cap.release()

cv2.destroyAllWindows()