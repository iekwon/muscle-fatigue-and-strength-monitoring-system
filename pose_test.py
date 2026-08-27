import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MOVEMENTS = {
    "1": ("Elbow (Biceps Curl)", {"L": (11, 13, 15), "R": (12, 14, 16)}),
    "2": ("Shoulder", {"L": (13, 11, 23), "R": (14, 12, 24)}),
    "3": ("Knee (Squat/Lunge)", {"L": (23, 25, 27), "R": (24, 26, 28)}),
    "4": ("Hip Flexion", {"L": (11, 23, 25), "R": (12, 24, 26)}),
    "5": ("Ankle", {"L": (25, 27, 31), "R": (26, 28, 32)}),
}

print("Select movement to track:")
for key, (name, _) in MOVEMENTS.items():
    print(f"  {key}. {name}")
movement_choice = input("Enter number: ").strip()

side_choice = input("Which side? (L/R): ").strip().upper()

movement_name, sides = MOVEMENTS[movement_choice]
a_idx, b_idx, c_idx = sides[side_choice]
print(f"Tracking: {movement_name} ({side_choice})")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def angle_to_color(angle, min_angle=30, max_angle=170):
    t = np.clip((angle - min_angle) / (max_angle - min_angle), 0, 1)
    return (0, int(255*t), int(255*(1-t)))  # BGR: red->green

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.VIDEO)

cap = cv2.VideoCapture(0)
frame_idx = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000/30))
        frame_idx += 1
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        h, w, _ = frame.shape

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            a = [lm[a_idx].x, lm[a_idx].y]
            b = [lm[b_idx].x, lm[b_idx].y]
            c = [lm[c_idx].x, lm[c_idx].y]

            angle = calculate_angle(a, b, c)
            color = angle_to_color(angle)

            pa, pb, pc = (int(a[0]*w),int(a[1]*h)), (int(b[0]*w),int(b[1]*h)), (int(c[0]*w),int(c[1]*h))
            cv2.line(frame, pa, pb, color, 6)
            cv2.line(frame, pb, pc, color, 6)
            for pt in [pa, pb, pc]:
                cv2.circle(frame, pt, 8, (255,255,255), -1)
                cv2.circle(frame, pt, 8, (0,0,0), 2)

            cv2.rectangle(frame, (10,10), (320,70), (30,30,30), -1)
            cv2.putText(frame, f"{movement_name}: {angle:.0f} deg", (20,45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('EMG-CV Dashboard', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()