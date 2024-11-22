import cv2

def check_cameras():
    for i in range(5):  # Thử các index từ 0 đến 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is working")
            cap.release()
        else:
            print(f"Camera {i} not found")

check_cameras()