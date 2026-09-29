import serial
import json
import time
import threading
from flask import Flask, jsonify
COM_PORT = "COM4"
BAUD_RATE = 115200

app = Flask(__name__)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response
latest_data = None
last_packet_time = 0
data_lock = threading.Lock()


def serial_reader():

    global latest_data
    global last_packet_time

    while True:

        try:
            print(f"Connecting to {COM_PORT}...")

            ser = serial.Serial(
                COM_PORT,
                BAUD_RATE,
                timeout=1
            )

            print("ESP32 USB CONNECTED")
            print("Waiting for sensor data...\n")

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

                    # Basic packet validation
                    if "sequence_id" not in data:
                        continue

                    if "device_id" not in data:
                        continue

                    # Store latest USB packet
                    with data_lock:
                        latest_data = data
                        last_packet_time = time.time()

                    print(
                        f"USB | "
                        f"Current: {data.get('current', 0):.3f} A | "
                        f"RPM: {data.get('rpm', 0):.1f} | "
                        f"Vibration: {data.get('vibration', 0):.3f} | "
                        f"Temperature: {data.get('temperature', 0):.2f} °C | "
                        f"IR: {data.get('ir_status')} | "
                        f"Belt: {data.get('belt_discontinuity')} | "
                        f"Seq: {data.get('sequence_id')}"
                    )

                except json.JSONDecodeError:
                    continue

        except serial.SerialException as e:

            print("USB connection error:", e)

        except Exception as e:

            print("USB reader error:", e)

        finally:

            try:
                ser.close()
            except:
                pass

        print("ESP32 USB disconnected.")
        print("Retrying in 2 seconds...\n")

        time.sleep(2)


@app.route("/usb_data")
def usb_data():

    with data_lock:

        data = latest_data
        packet_time = last_packet_time

    if data is None:

        return jsonify({
            "available": False,
            "data": None
        })

    age = time.time() - packet_time

    # USB considered active if packet received
    # within the last 1 second
    usb_active = age <= 1.0

    return jsonify({
        "available": usb_active,
        "age": age,
        "data": data
    })


if __name__ == "__main__":

    reader_thread = threading.Thread(
        target=serial_reader,
        daemon=True
    )

    reader_thread.start()

    print("\n================================")
    print("ESP32 USB DASHBOARD BRIDGE")
    print("================================")
    print("USB data endpoint:")
    print("http://127.0.0.1:5001/usb_data")
    print("================================\n")

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        threaded=True
    )