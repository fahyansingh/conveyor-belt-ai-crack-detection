// ============================================================
// INTELLIGENT CONVEYOR BELT HEALTH MONITORING
// USB PRIMARY + FIREBASE FALLBACK
// ============================================================

const FIREBASE_ROOT_URL =
    "https://conveyor-health-monitoring-default-rtdb.asia-southeast1.firebasedatabase.app/sensorData.json";

const FIREBASE_VISION_URL =
    "https://conveyor-health-monitoring-default-rtdb.asia-southeast1.firebasedatabase.app/vision.json";

const USB_DATA_URL =
    "http://127.0.0.1:5001/usb_data";

// ============================================================
// LIVE VISION SERVER
// ============================================================

const LOCAL_VISION_URL =
    "http://127.0.0.1:5000/vision_data";


// ============================================================
// REAL SENSOR VARIABLES
// ============================================================

let motorCurrent = null;
let motorSpeed = null;
let vibration = null;
let temperature = null;
let beltDiscontinuity = null;

let lastDiscontinuityTime = 0;
let discontinuityHoldTimer = null;
const DISCONTINUITY_HOLD_DURATION = 2000;

// ============================================================
// DATA SOURCE CONTROL
// USB = PRIMARY
// FIREBASE = FALLBACK
// ============================================================

let usbActive = false;
let dataSource = "FIREBASE";

let lastUSBPacketKey = null;
let lastFirebasePacketKey = null;

let lastSensorUpdateTime = 0;

const USB_TIMEOUT = 1000;


// ============================================================
// VISION
// ============================================================

let visionStatus = "CHECKING...";


// ============================================================
// CAMERA INSPECTION / JOINT GAP
// ============================================================

let jointGapMm = null;
let gapStatus = "WAITING";
let crackDetection = "NOT DETECTED";
let jointDetection = "NOT DETECTED";
let holeDetection = "NOT DETECTED";
let cameraVisionStatus = "NORMAL";

function updateCameraInspection() {

    const jointElement =
        document.getElementById("jointDetection");

    const gapElement =
        document.getElementById("jointGapValue");

    const gapStatusElement =
        document.getElementById("gapStatus");

    const crackElement =
        document.getElementById("crackDetection");

    const holeElement =
        document.getElementById("holeDetection");
    
    const visionElement =
        document.getElementById("cameraVisionStatus");

    
    if (jointElement) {

        jointElement.textContent =
            jointDetection || "NOT DETECTED";

    }    
    
    if (holeElement) {

        holeElement.textContent =
         holeDetection || "NOT DETECTED";

    }
    
    if (gapElement) {

        gapElement.textContent =
            isValidNumber(jointGapMm)
                ? Number(jointGapMm).toFixed(1) + " mm"
                : "N/A";

    }

    if (gapStatusElement) {

        gapStatusElement.textContent =
            gapStatus || "WAITING";

    }

    if (crackElement) {

        crackElement.textContent =
            crackDetection || "NOT DETECTED";

    }

    if (visionElement) {

        visionElement.textContent =
            cameraVisionStatus || "NORMAL";

    }

}


// ============================================================
// CHART SETTINGS
// ============================================================

const MAX_POINTS = 20;

let timeLabels = [];

let currentData = [];
let speedData = [];
let vibrationData = [];
let temperatureData = [];

let currentChart = null;
let speedChart = null;
let vibrationChart = null;
let temperatureChart = null;


// ============================================================
// HELPER FUNCTIONS
// ============================================================

function isValidNumber(value) {

    return (
        value !== null &&
        value !== undefined &&
        value !== "" &&
        !isNaN(Number(value))
    );

}


function convertNumber(value) {

    if (isValidNumber(value)) {
        return Number(value);
    }

    return null;

}


function formatValue(value, decimals = 2) {

    if (!isValidNumber(value)) {
        return "N/A";
    }

    return Number(value).toFixed(decimals);

}


// ============================================================
// SENSOR PACKET KEY
// device_id + sequence_id
// ============================================================

function getPacketKey(data) {

    if (!data || typeof data !== "object") {
        return null;
    }

    if (
        data.device_id === undefined ||
        data.sequence_id === undefined
    ) {
        return null;
    }

    return (
        String(data.device_id) +
        "_" +
        String(data.sequence_id)
    );

}


// ============================================================
// START DASHBOARD
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        console.log(
            "REAL-TIME CONVEYOR DASHBOARD STARTED"
        );

        console.log(
            "Sensor priority: USB PRIMARY → Firebase FALLBACK"
        );

        console.log(
            "Vision source: LOCAL AI SERVER"
        );

        initializeCharts();

        updateCameraInspection();

        // Start USB immediately
        readUSBData();

        // Firebase fallback fetch
        readRealSensorData();

        // Live AI vision server
        readVisionData();

        // USB checked frequently
        setInterval(
            readUSBData,
            500
        );

        // Firebase checked every 2 sec
        setInterval(
            readRealSensorData,
            2000
        );

        // Live AI vision
        setInterval(
            readVisionData,
            500
        );

    }
);


// ============================================================
// FIREBASE REAL-TIME EVENT
// FIREBASE IS USED ONLY WHEN USB IS NOT ACTIVE
// ============================================================

window.addEventListener(
    "firebaseSensorData",
    function (event) {

        const rawData =
            event.detail;

        if (!rawData) {
            return;
        }

        if (usbActive) {

            console.log(
                "Firebase packet ignored because USB is active."
            );

            return;

        }

        console.log(
            "Firebase fallback packet received:",
            rawData
        );

        processFirebaseSensorData(
            rawData
        );

    }
);


// ============================================================
// READ USB DATA
// USB = PRIMARY SOURCE
// ============================================================

async function readUSBData() {

    try {

        const response =
            await fetch(
                USB_DATA_URL +
                "?t=" +
                Date.now(),
                {
                    cache: "no-store"
                }
            );

        if (!response.ok) {

            throw new Error(
                "USB bridge error: " +
                response.status
            );

        }

        const result =
            await response.json();

        if (
            result.available &&
            result.data
        ) {

            usbActive = true;
            dataSource = "USB";

            const data =
                result.data;

            const packetKey =
                getPacketKey(data);

            // ------------------------------------------------
            // Ignore duplicate USB packet
            // ------------------------------------------------

            if (
                packetKey !== null &&
                packetKey === lastUSBPacketKey
            ) {

                return;

            }

            lastUSBPacketKey =
                packetKey;

            console.log(
                "PRIMARY SOURCE: USB | Seq:",
                data.sequence_id
            );

            processRootSensorData(
                data
            );

            updateDataSourceIndicator();

            return;

        }

        // ----------------------------------------------------
        // USB unavailable
        // ----------------------------------------------------

        if (usbActive) {

            console.log(
                "USB disconnected → Firebase fallback"
            );

        }

        usbActive = false;
        dataSource = "FIREBASE";

        updateDataSourceIndicator();

    }

    catch (error) {

        if (usbActive) {

            console.log(
                "USB connection lost → Firebase fallback"
            );

        }

        usbActive = false;
        dataSource = "FIREBASE";

        updateDataSourceIndicator();

    }

}


// ============================================================
// DATA SOURCE INDICATOR
// Optional element if added to HTML
// ============================================================

function updateDataSourceIndicator() {

    const element =
        document.getElementById(
            "dataSource"
        );

    if (!element) {
        return;
    }

    if (usbActive) {

        element.textContent =
            "USB PRIMARY";

        element.style.color =
            "#22c55e";

    }

    else {

        element.textContent =
            "FIREBASE BACKUP";

        element.style.color =
            "#eab308";

    }

}


// ============================================================
// READ ROOT FIREBASE DATA
// FIREBASE = FALLBACK ONLY
// ============================================================

async function readRealSensorData() {

    try {

        const response =
            await fetch(
                FIREBASE_ROOT_URL +
                "?t=" +
                Date.now(),
                {
                    cache: "no-store"
                }
            );

        if (!response.ok) {

            throw new Error(
                "Firebase error: " +
                response.status
            );

        }

        const data =
            await response.json();

        if (!data) {

            console.warn(
                "Firebase root is empty."
            );

            if (!usbActive) {
                clearSensorValues();
            }

            return;

        }

        // ----------------------------------------------------
        // IMPORTANT:
        // USB active → Firebase ignored
        // USB inactive → Firebase used
        // ----------------------------------------------------

        if (!usbActive) {

            processFirebaseSensorData(
                data
            );

        }
        else {

            console.log(
                "Firebase packet ignored — USB is PRIMARY."
            );

        }

    }

    catch (error) {

        console.error(
            "Firebase sensor read error:",
            error
        );

    }

}


// ============================================================
// PROCESS FIREBASE SENSOR DATA
// ============================================================

function processFirebaseSensorData(
    rawData
) {

    const data =
        extractLatestPacket(
            rawData
        );

    if (
        !data ||
        typeof data !== "object"
    ) {

        return;

    }

    const packetKey =
        getPacketKey(data);

    // --------------------------------------------------------
    // Ignore duplicate Firebase packet
    // --------------------------------------------------------

    if (
        packetKey !== null &&
        packetKey === lastFirebasePacketKey
    ) {

        return;

    }

    lastFirebasePacketKey =
        packetKey;

    dataSource =
        "FIREBASE";

    processRootSensorData(
        data
    );

    updateDataSourceIndicator();

}


// ============================================================
// EXTRACT LATEST PACKET HELPER
// ============================================================

function extractLatestPacket(
    rawData
) {

    if (
        !rawData ||
        typeof rawData !== "object"
    ) {

        return rawData;

    }


    // 1. Firebase push packets

    const pushKeys =
        Object.keys(
            rawData
        ).filter(
            function (key) {

                return (
                    key.startsWith("-") &&
                    typeof rawData[key] === "object" &&
                    rawData[key] !== null
                );

            }
        );


    if (
        pushKeys.length > 0
    ) {

        pushKeys.sort();

        const latestKey =
            pushKeys[
                pushKeys.length - 1
            ];

        return rawData[
            latestKey
        ];

    }


    // 2. Nested packet objects

    const objectKeys =
        Object.keys(
            rawData
        ).filter(
            function (key) {

                return (
                    typeof rawData[key] === "object" &&
                    rawData[key] !== null
                );

            }
        );


    if (
        objectKeys.length > 0
    ) {

        objectKeys.sort();

        const latestKey =
            objectKeys[
                objectKeys.length - 1
            ];

        return rawData[
            latestKey
        ];

    }


    // 3. Direct packet

    return rawData;

}


// ============================================================
// PROCESS ROOT SENSOR DATA
// ============================================================

function processRootSensorData(
    rawData
) {

    const data =
        extractLatestPacket(
            rawData
        );

    if (
        !data ||
        typeof data !== "object"
    ) {

        return;

    }


    motorCurrent =
        convertNumber(
            data.current !== undefined
                ? data.current
                :
            data.motorCurrent !== undefined
                ? data.motorCurrent
                :
            data.Current !== undefined
                ? data.Current
                :
            data.motor_current !== undefined
                ? data.motor_current
                :
            null
        );


    motorSpeed =
        convertNumber(
            data.rpm !== undefined
                ? data.rpm
                :
            data.motorSpeed !== undefined
                ? data.motorSpeed
                :
            data.speed !== undefined
                ? data.speed
                :
            data.Speed !== undefined
                ? data.Speed
                :
            data.RPM !== undefined
                ? data.RPM
                :
            data.motor_speed !== undefined
                ? data.motor_speed
                :
            null
        );


    temperature =
        convertNumber(
            data.temperature !== undefined
                ? data.temperature
                :
            data.temp !== undefined
                ? data.temp
                :
            data.Temp !== undefined
                ? data.Temp
                :
            data.Temperature !== undefined
                ? data.Temperature
                :
            null
        );


    vibration =
        convertNumber(
            data.vibration !== undefined
                ? data.vibration
                :
            data.vib !== undefined
                ? data.vib
                :
            data.Vibration !== undefined
                ? data.Vibration
                :
            null
        );


    const rawBelt =
        data.belt_discontinuity !== undefined
            ? data.belt_discontinuity
            :
        data.beltDiscontinuity !== undefined
            ? data.beltDiscontinuity
            :
        data.belt !== undefined
            ? data.belt
            :
        data.discontinuity !== undefined
            ? data.discontinuity
            :
        data.ir_status !== undefined
            ? (
                data.ir_status === 0 ||
                data.ir_status === "0"
              )
            :
        data.ir !== undefined
            ? data.ir
            :
        null;


    let detectedNow =
        false;


    if (
        rawBelt === true ||
        rawBelt === 1 ||
        rawBelt === "1" ||
        String(
            rawBelt
        ).toLowerCase() === "true" ||
        String(
            rawBelt
        ).toLowerCase() === "damage" ||
        String(
            rawBelt
        ).toLowerCase() === "detected"
    ) {

        detectedNow =
            true;

    }

    else if (
        rawBelt === false ||
        rawBelt === 0 ||
        rawBelt === "0" ||
        String(
            rawBelt
        ).toLowerCase() === "false" ||
        String(
            rawBelt
        ).toLowerCase() === "normal"
    ) {

        detectedNow =
            false;

    }


    const now =
        Date.now();


    if (
        detectedNow
    ) {

        lastDiscontinuityTime =
            now;

        beltDiscontinuity =
            true;


        if (
            discontinuityHoldTimer
        ) {

            clearTimeout(
                discontinuityHoldTimer
            );

        }


        discontinuityHoldTimer =
            setTimeout(
                function () {

                    if (
                        Date.now() -
                        lastDiscontinuityTime >=
                        DISCONTINUITY_HOLD_DURATION
                    ) {

                        beltDiscontinuity =
                            false;

                        updateBeltStatus();
                        updateHealth();
                        updateAlerts();

                    }

                },
                DISCONTINUITY_HOLD_DURATION
            );

    }

    else {

        if (
            now -
            lastDiscontinuityTime <
            DISCONTINUITY_HOLD_DURATION
        ) {

            beltDiscontinuity =
                true;

        }

        else {

            beltDiscontinuity =
                (
                    rawBelt === null
                        ? null
                        : false
                );

        }

    }


    console.log(
        "PARSED SENSOR VALUES:",
        {
            source: dataSource,
            device_id: data.device_id,
            sequence_id: data.sequence_id,
            motorCurrent: motorCurrent,
            motorSpeed: motorSpeed,
            vibration: vibration,
            temperature: temperature,
            beltDiscontinuity:
                beltDiscontinuity
        }
    );


    updateSensorCards();

    updateBeltStatus();

    updateHealth();

    updateAlerts();

    updateCharts();

}


// ============================================================
// CLEAR SENSOR VALUES
// ============================================================

function clearSensorValues() {

    motorCurrent = null;
    motorSpeed = null;
    vibration = null;
    temperature = null;
    beltDiscontinuity = null;

    updateSensorCards();

    updateBeltStatus();

    updateHealth();

    updateAlerts();

}


// ============================================================
// UPDATE SENSOR CARDS
// ============================================================

function updateSensorCards() {

    const currentElement =
        document.getElementById(
            "currentValue"
        );

    const speedElement =
        document.getElementById(
            "speedValue"
        );

    const vibrationElement =
        document.getElementById(
            "vibrationValue"
        );

    const temperatureElement =
        document.getElementById(
            "temperatureValue"
        );


    if (currentElement) {

        currentElement.textContent =
            formatValue(
                motorCurrent,
                2
            );

    }


    if (speedElement) {

        speedElement.textContent =
            formatValue(
                motorSpeed,
                0
            );

    }


    if (vibrationElement) {

        vibrationElement.textContent =
            formatValue(
                vibration,
                2
            );

    }


    if (temperatureElement) {

        temperatureElement.textContent =
            formatValue(
                temperature,
                2
            );

    }


    updateParameterStatus(
        "currentMotorStatus",
        motorCurrent,
        0.2,
        0.7
    );


    updateSpeedStatus();


    updateParameterStatus(
        "vibrationStatus",
        vibration,
        5,
        8
    );


    updateParameterStatus(
        "temperatureStatus",
        temperature,
        70,
        85
    );

}


// ============================================================
// PARAMETER STATUS
// ============================================================

function updateParameterStatus(
    elementId,
    value,
    warningLimit,
    criticalLimit
) {

    const element =
        document.getElementById(
            elementId
        );

    if (!element) {
        return;
    }


    if (
        !isValidNumber(value)
    ) {

        element.className =
            "normal";

        element.textContent =
            "● N/A";

        return;

    }


    if (
        Number(value) >=
        criticalLimit
    ) {

        element.className =
            "critical";

        element.textContent =
            "● CRITICAL";

    }

    else if (
        Number(value) >=
        warningLimit
    ) {

        element.className =
            "warning";

        element.textContent =
            "● WARNING";

    }

    else {

        element.className =
            "normal";

        element.textContent =
            "● NORMAL";

    }

}


// ============================================================
// RPM STATUS
// ============================================================

function updateSpeedStatus() {

    const element =
        document.getElementById(
            "speedStatus"
        );

    if (!element) {
        return;
    }


    if (
        !isValidNumber(
            motorSpeed
        )
    ) {

        element.className =
            "normal";

        element.textContent =
            "● N/A";

        return;

    }


    if (
        motorSpeed < 100||
        motorSpeed > 250
    ) {

        element.className =
            "critical";

        element.textContent =
            "● CRITICAL";

    }

    else if (
        motorSpeed < 100 ||
        motorSpeed > 250
    ) {

        element.className =
            "warning";

        element.textContent =
            "● WARNING";

    }

    else {

        element.className =
            "normal";

        element.textContent =
            "● NORMAL";

    }

}


// ============================================================
// BELT / IR STATUS
// ============================================================

function updateBeltStatus() {

    const objectElement =
        document.getElementById(
            "objectDetection"
        );

    const irElement =
        document.getElementById(
            "irSensorStatus"
        );

    const beltElement =
        document.getElementById(
            "beltCondition"
        );


    if (
        beltDiscontinuity ===
        null
    ) {

        if (objectElement) {
            objectElement.textContent =
                "BELT STATUS: N/A";
        }

        if (irElement) {
            irElement.textContent =
                "N/A";
        }

        if (beltElement) {
            beltElement.textContent =
                "N/A";
        }

        return;

    }


    if (irElement) {

        irElement.textContent =
            "ACTIVE";

    }


    if (
        beltDiscontinuity ===
        true
    ) {

        if (objectElement) {

            objectElement.textContent =
                "BELT STATUS: DAMAGE DETECTED";

            objectElement.style.color =
                "#ef4444";

        }


        if (beltElement) {

            beltElement.textContent =
                "DISCONTINUITY DETECTED";

            beltElement.style.color =
                "#ef4444";

        }

    }

    else {

        if (objectElement) {

            objectElement.textContent =
                "BELT STATUS: NORMAL";

            objectElement.style.color =
                "#38bdf8";

        }


        if (beltElement) {

            beltElement.textContent =
                "NORMAL";

            beltElement.style.color =
                "#38bdf8";

        }

    }

}


// ============================================================
// CAMERA / VISION
// LIVE LOCAL AI SERVER
// ============================================================

async function readVisionData() {

    try {

        const response =
            await fetch(
                LOCAL_VISION_URL +
                "?t=" +
                Date.now(),
                {
                    cache: "no-store"
                }
            );


        if (!response.ok) {

            throw new Error(
                "Local vision server error: " +
                response.status
            );

        }


        const data =
            await response.json();


        if (!data) {

            return;

        }

        // ----------------------------------------------------
        // JOINT DETECTION
        // ----------------------------------------------------

        if (
        data.joint_detection !== undefined &&
         data.joint_detection !== null
     ) {

          jointDetection =
             String(
                 data.joint_detection
             );

}
        // ----------------------------------------------------
        // JOINT GAP
        // ----------------------------------------------------

        if (
            data.joint_gap_mm !== undefined &&
            data.joint_gap_mm !== null
        ) {

            jointGapMm =
                Number(
                    data.joint_gap_mm
                );

        }

        else {

            jointGapMm =
                null;

        }


        // ----------------------------------------------------
        // GAP STATUS
        // ----------------------------------------------------

        if (
            data.gap_status !== undefined &&
            data.gap_status !== null
        ) {

            gapStatus =
                String(
                    data.gap_status
                );

        }

        else {

            gapStatus =
                "WAITING";

        }


        // ----------------------------------------------------
        // GAP DETECTION
        // ----------------------------------------------------

        if (
            data.gap_detected !== undefined
        ) {

            // Keep this information available
            // without changing the existing
            // Camera Inspection layout.

            if (
                data.gap_detected === false &&
                gapStatus === "WAITING"
            ) {

                gapStatus =
                    "WAITING";

            }

        }


        // ----------------------------------------------------
        // CRACK DETECTION
        // ----------------------------------------------------

        if (
            data.crack_detection !== undefined &&
            data.crack_detection !== null
        ) {

            crackDetection =
                String(
                    data.crack_detection
                );

        }


        // ----------------------------------------------------
        // HOLE DETECTION
        // ----------------------------------------------------

        if (
            data.hole_detection !== undefined &&
            data.hole_detection !== null
        ) {

            holeDetection =
                String(
                   data.hole_detection
                );

        }


        // ----------------------------------------------------
        // VISION STATUS
        // ----------------------------------------------------

        if (
            data.vision_status !== undefined &&
            data.vision_status !== null
        ) {

            cameraVisionStatus =
                String(
                    data.vision_status
                );

        }

        else {

            cameraVisionStatus =
                "NORMAL";

        }


        // ----------------------------------------------------
        // EXISTING GLOBAL VISION STATUS
        // ----------------------------------------------------

        if (
            gapStatus === "CRITICAL" ||
            crackDetection === "DETECTED" ||
            crackDetection === "CRACK DETECTED" ||
            cameraVisionStatus === "DAMAGE DETECTED"
        ) {

            visionStatus =
                "DAMAGE DETECTED";

        }

        else {

            visionStatus =
                "BELT NORMAL";

        }


        // ----------------------------------------------------
        // UPDATE CAMERA INSPECTION
        // ----------------------------------------------------

        updateCameraInspection();


        // ----------------------------------------------------
        // UPDATE HEALTH + ALERTS
        // ----------------------------------------------------

        updateHealth();

        updateAlerts();


        console.log(
            "LIVE AI VISION:",
            data
        );

    }

    catch (error) {

        console.error(
            "Local vision read error:",
            error
        );

    }

}


// ============================================================
// HEALTH CALCULATION
// ============================================================

function updateHealth() {

    const scoreElement =
        document.getElementById(
            "healthScore"
        );

    const statusElement =
        document.getElementById(
            "healthStatus"
        );

    const messageElement =
        document.getElementById(
            "healthMessage"
        );


    let score = 100;

    let hasData = false;


    if (
        isValidNumber(
            motorCurrent
        )
    ) {

        hasData = true;

        if (
            motorCurrent >= 0.7
        ) {

            score -= 20;

        }

        else if (
            motorCurrent >= 0.2
        ) {

            score -= 8;

        }

    }


    if (
        isValidNumber(
            motorSpeed
        )
    ) {

        hasData = true;

        if (
            motorSpeed < 100 ||
            motorSpeed > 700
        ) {

            score -= 20;

        }

        else if (
            motorSpeed < 200 ||
            motorSpeed > 600
        ) {

            score -= 8;

        }

    }


    if (
        isValidNumber(
            vibration
        )
    ) {

        hasData = true;

        if (
            vibration >= 8
        ) {

            score -= 20;

        }

        else if (
            vibration >= 5
        ) {

            score -= 8;

        }

    }


    if (
        isValidNumber(
            temperature
        )
    ) {

        hasData = true;

        if (
            temperature >= 85
        ) {

            score -= 20;

        }

        else if (
            temperature >= 70
        ) {

            score -= 8;

        }

    }


    if (
        beltDiscontinuity !==
        null
    ) {

        hasData = true;

        if (
            beltDiscontinuity ===
            true
        ) {

            score -= 25;

        }

    }


    if (
        visionStatus ===
        "DAMAGE DETECTED"
    ) {

        hasData = true;

        score -= 20;

    }


    score =
        Math.max(
            0,
            Math.min(
                100,
                score
            )
        );


    if (!hasData) {

        if (scoreElement) {
            scoreElement.textContent =
                "N/A";
        }

        if (statusElement) {

            statusElement.textContent =
                "WAITING";

            statusElement.style.color =
                "#94a3b8";

        }

        if (messageElement) {

            messageElement.textContent =
                "Waiting for sensor data";

        }

        return;

    }


    if (scoreElement) {

        scoreElement.textContent =
            score + "%";

    }


    if (score >= 80) {

        if (statusElement) {

            statusElement.textContent =
                "HEALTHY";

            statusElement.style.color =
                "#22c55e";

        }

        if (messageElement) {

            messageElement.textContent =
                "Conveyor operating normally";

        }

    }

    else if (score >= 60) {

        if (statusElement) {

            statusElement.textContent =
                "WARNING";

            statusElement.style.color =
                "#eab308";

        }

        if (messageElement) {

            messageElement.textContent =
                "Abnormal condition detected";

        }

    }

    else {

        if (statusElement) {

            statusElement.textContent =
                "CRITICAL";

            statusElement.style.color =
                "#ef4444";

        }

        if (messageElement) {

            messageElement.textContent =
                "Critical conveyor condition detected";

        }

    }

}


// ============================================================
// SYSTEM ALERTS
// ============================================================

function updateAlerts() {

    updateGenericAlert(
        "currentAlert",
        "Motor current",
        motorCurrent,
        0.2,
        0.7
    );


    updateSpeedAlert();


    updateGenericAlert(
        "vibrationAlert",
        "Motor vibration",
        vibration,
        5,
        8
    );


    updateGenericAlert(
        "temperatureAlert",
        "Motor temperature",
        temperature,
        70,
        85
    );


    const beltAlert =
        document.getElementById(
            "beltAlert"
        );


    if (beltAlert) {

        if (
            beltDiscontinuity ===
            true
        ) {

            beltAlert.className =
                "alert critical-alert";

            beltAlert.textContent =
                "⚠ Belt discontinuity detected";

        }

        else if (
            beltDiscontinuity ===
            false
        ) {

            beltAlert.className =
                "alert normal-alert";

            beltAlert.textContent =
                "✓ Belt condition normal";

        }

        else {

            beltAlert.className =
                "alert normal-alert";

            beltAlert.textContent =
                "• Waiting for IR sensor";

        }

    }


    const liveAlert =
        document.getElementById(
            "liveAlert"
        );


    if (liveAlert) {

        if (
            visionStatus ===
            "DAMAGE DETECTED" ||
            beltDiscontinuity ===
            true
        ) {

            liveAlert.className =
                "alert critical-alert";

            liveAlert.textContent =
                "⚠ Damage detected — inspection required";

        }

        else {

            liveAlert.className =
                "alert normal-alert";

            liveAlert.textContent =
                "✓ All systems operating normally";

        }

    }

}


// ============================================================
// GENERIC ALERT
// ============================================================

function updateGenericAlert(
    elementId,
    name,
    value,
    warningLimit,
    criticalLimit
) {

    const element =
        document.getElementById(
            elementId
        );

    if (!element) {
        return;
    }


    if (
        !isValidNumber(value)
    ) {

        element.className =
            "alert normal-alert";

        element.textContent =
            "• " +
            name +
            " data unavailable";

        return;

    }


    if (
        Number(value) >=
        criticalLimit
    ) {

        element.className =
            "alert critical-alert";

        element.textContent =
            "⚠ " +
            name +
            " critical";

    }

    else if (
        Number(value) >=
        warningLimit
    ) {

        element.className =
            "alert warning-alert";

        element.textContent =
            "⚠ " +
            name +
            " above normal range";

    }

    else {

        element.className =
            "alert normal-alert";

        element.textContent =
            "✓ " +
            name +
            " within normal range";

    }

}


// ============================================================
// RPM ALERT
// ============================================================

function updateSpeedAlert() {

    const element =
        document.getElementById(
            "speedAlert"
        );

    if (!element) {
        return;
    }


    if (
        !isValidNumber(
            motorSpeed
        )
    ) {

        element.className =
            "alert normal-alert";

        element.textContent =
            "• Motor speed data unavailable";

        return;

    }


    if (
        motorSpeed < 100 ||
        motorSpeed > 250
    ) {

        element.className =
            "alert critical-alert";

        element.textContent =
            "⚠ Motor speed critical";

    }

    else if (
        motorSpeed < 100 ||
        motorSpeed > 250
    ) {

        element.className =
            "alert warning-alert";

        element.textContent =
            "⚠ Motor speed outside normal range";

    }

    else {

        element.className =
            "alert normal-alert";

        element.textContent =
            "✓ Motor speed within normal range";

    }

}


// ============================================================
// CHART INITIALIZATION
// ============================================================

function initializeCharts() {

    if (
        typeof Chart ===
        "undefined"
    ) {

        console.error(
            "Chart.js is not loaded."
        );

        return;

    }


    const commonOptions = {

        responsive: true,

        maintainAspectRatio: false,

        animation: false,

        plugins: {

            legend: {

                display: false

            }

        },

        scales: {

            x: {

                ticks: {

                    color:
                        "#94a3b8"

                },

                grid: {

                    color:
                        "#334155"

                }

            },

            y: {

                ticks: {

                    color:
                        "#94a3b8"

                },

                grid: {

                    color:
                        "#334155"

                }

            }

        }

    };


    const currentCanvas =
        document.getElementById(
            "currentChart"
        );

    const speedCanvas =
        document.getElementById(
            "speedChart"
        );

    const vibrationCanvas =
        document.getElementById(
            "vibrationChart"
        );

    const temperatureCanvas =
        document.getElementById(
            "temperatureChart"
        );


    if (currentCanvas) {

        currentChart =
            new Chart(
                currentCanvas,
                createChartConfig(
                    "Current",
                    currentData,
                    commonOptions
                )
            );

    }


    if (speedCanvas) {

        speedChart =
            new Chart(
                speedCanvas,
                createChartConfig(
                    "Speed",
                    speedData,
                    commonOptions
                )
            );

    }


    if (vibrationCanvas) {

        vibrationChart =
            new Chart(
                vibrationCanvas,
                createChartConfig(
                    "Vibration",
                    vibrationData,
                    commonOptions
                )
            );

    }


    if (temperatureCanvas) {

        temperatureChart =
            new Chart(
                temperatureCanvas,
                createChartConfig(
                    "Temperature",
                    temperatureData,
                    commonOptions
                )
            );

    }

}


// ============================================================
// CREATE CHART
// ============================================================

function createChartConfig(
    label,
    dataArray,
    options
) {

    return {

        type: "line",

        data: {

            labels:
                timeLabels,

            datasets: [

                {

                    label:
                        label,

                    data:
                        dataArray,

                    borderWidth:
                        2,

                    pointRadius:
                        2,

                    tension:
                        0.25,

                    fill:
                        false

                }

            ]

        },

        options:
            options

    };

}


// ============================================================
// UPDATE CHARTS
// ============================================================

function updateCharts() {

    const now =
        new Date();

    const time =
        now.toLocaleTimeString();

    timeLabels.push(
        time
    );

    currentData.push(
        motorCurrent
    );

    speedData.push(
        motorSpeed
    );

    vibrationData.push(
        vibration
    );

    temperatureData.push(
        temperature
    );


    if (
        timeLabels.length >
        MAX_POINTS
    ) {

        timeLabels.shift();

    }

    if (
        currentData.length >
        MAX_POINTS
    ) {

        currentData.shift();

    }

    if (
        speedData.length >
        MAX_POINTS
    ) {

        speedData.shift();

    }

    if (
        vibrationData.length >
        MAX_POINTS
    ) {

        vibrationData.shift();

    }

    if (
        temperatureData.length >
        MAX_POINTS
    ) {

        temperatureData.shift();

    }


    refreshChart(
        currentChart,
        currentData
    );

    refreshChart(
        speedChart,
        speedData
    );

    refreshChart(
        vibrationChart,
        vibrationData
    );

    refreshChart(
        temperatureChart,
        temperatureData
    );

}


// ============================================================
// REFRESH CHART
// ============================================================

function refreshChart(
    chart,
    dataArray
) {

    if (!chart) {
        return;
    }


    chart.data.labels =
        timeLabels;

    chart.data.datasets[0].data =
        dataArray;

    chart.update(
        "none"
    );

}


// ============================================================
// FINAL DEBUG INFORMATION
// ============================================================

console.log(
    "Dashboard is READ-ONLY."
);

console.log(
    "Sensor source: USB PRIMARY → Firebase FALLBACK"
);

console.log(
    "USB endpoint: http://127.0.0.1:5001/usb_data"
);

console.log(
    "Vision source: LOCAL AI SERVER"
);

console.log(
    "Vision endpoint: http://127.0.0.1:5000/vision_data"
);

console.log(
    "Device identity: device_id + sequence_id"
);

console.log(
    "No simulated sensor values are used."
);

console.log(
    "No sensor values are written to Firebase."
);