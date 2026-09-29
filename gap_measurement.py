import cv2
import os

# ==============================
# CURRENT GAP
# ==============================
GAP = "repair"   # <-- 2mm ke liye 2mm
                #     3mm ke liye 3mm
                #     4mm ke liye 4mm, etc.

# DroidCam = Camera Index 1
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("DroidCam open nahi hua!")
    exit()

# Save folder
save_folder = "dataset/images"
os.makedirs(save_folder, exist_ok=True)

# Find next number ONLY for current gap
existing_files = [
    f for f in os.listdir(save_folder)
    if f.startswith(GAP + "_") and f.lower().endswith(".jpg")
]

image_number = len(existing_files) + 1

print("DroidCam connected!")
print(f"Current Gap: {GAP}")
print("S = Capture image")
print("Q = Exit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame nahi mila!")
        break

    cv2.imshow("SIH - DroidCam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        if image_number > 30:
            print(f"{GAP} ke 30 images already complete!")
            continue

        filename = f"{GAP}_{image_number:02d}.jpg"
        filepath = os.path.join(save_folder, filename)

        cv2.imwrite(filepath, frame)

        print(f"Saved: {filepath}")
        image_number += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()