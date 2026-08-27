import cv2
import time

cap = cv2.VideoCapture(0)
start = time.time()
frame_count = 0

while time.time() - start < 10:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break
    frame_count += 1
    cv2.imshow('Camera Test', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print(f"Done. Frames shown: {frame_count}")