import serial
import json

COM_PORT = "COM4"
BAUD_RATE = 115200

ser = serial.Serial(
    COM_PORT,
    BAUD_RATE,
    timeout=1
)

print("ESP32 SENSOR READER")
print("Connected to", COM_PORT)
print("Waiting for sensor data...\n")

while True:
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        # Sirf JSON packets process karo
        if not line.startswith("{"):
            continue

        data = json.loads(line)

        print(
            f"Current: {data['current']:.3f} A | "
            f"RPM: {data['rpm']:.1f} | "
            f"Vibration: {data['vibration']:.3f} | "
            f"Temperature: {data['temperature']:.2f} °C | "
            f"IR: {data['ir_status']} | "
            f"Belt: {data['belt_discontinuity']} | "
            f"Seq: {data['sequence_id']}"
        )

    except json.JSONDecodeError:
        continue

    except KeyboardInterrupt:
        print("\nStopped.")
        ser.close()
        break

    except Exception as e:
        print("Error:", e)
        break