import cv2
import os
import subprocess

def check_camera_module():
    print("[INFO] Checking if PiCam is detected by system...")
    try:
        output = subprocess.check_output("vcgencmd get_camera", shell=True).decode("utf-8")
        print("[INFO] vcgencmd result:", output.strip())
    except Exception as e:
        print("[WARN] Could not run vcgencmd (maybe using Bullseye with libcamera):", e)

def test_opencv_camera():
    print("[INFO] Trying to access PiCam with OpenCV...")
    cap = cv2.VideoCapture(0)  # Camera index 0
    if not cap.isOpened():
        print("[ERROR] Could not open PiCam via OpenCV. Try installing libcamera support.")
        return

    print("[INFO] Camera opened successfully. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
        cv2.imshow("PiCam Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_camera_module()
    test_opencv_camera()
