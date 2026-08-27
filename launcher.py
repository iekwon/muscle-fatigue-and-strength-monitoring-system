import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import threading
from collections import deque
import nidaqmx
from scipy import signal as sg

MOVEMENTS = {
    "Elbow (Biceps Curl)": {"L": (11, 13, 15), "R": (12, 14, 16)},
    "Shoulder": {"L": (13, 11, 23), "R": (14, 12, 24)},
    "Knee (Squat/Lunge)": {"L": (23, 25, 27), "R": (24, 26, 28)},
    "Hip Flexion": {"L": (11, 23, 25), "R": (12, 24, 26)},
    "Ankle": {"L": (25, 27, 31), "R": (26, 28, 32)},
}

# ---- EMG (NI DAQ) config ----
CHANNEL = "Dev1/ai0"
FS = 2500
CHUNK = 100
WIN = int(FS * 0.2)
PLOT_SECONDS = 3
PLOT_DOWNSAMPLE = 5

latest_rms = 0.0
plot_buffer = deque(maxlen=int(FS * PLOT_SECONDS / PLOT_DOWNSAMPLE))
emg_running = True


def resource_path(filename):
    """Finds bundled files whether running as .py or as a PyInstaller .exe."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return filename


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def angle_to_color(angle, min_angle=30, max_angle=170):
    t = np.clip((angle - min_angle) / (max_angle - min_angle), 0, 1)
    return (0, int(255*t), int(255*(1-t)))


def emg_thread():
    """Background thread: reads live EMG from the NI DAQ, filters it,
    updates latest_rms and plot_buffer continuously."""
    global latest_rms
    nyq = FS / 2
    bp_sos = sg.butter(4, [20/nyq, 450/nyq], btype='bandpass', output='sos')
    notch_b, notch_a = sg.iirnotch(50/nyq, 30)
    bp_zi = sg.sosfilt_zi(bp_sos)
    notch_zi = sg.lfilter_zi(notch_b, notch_a)
    buffer = np.empty(0)

    try:
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(CHANNEL)
            task.timing.cfg_samp_clk_timing(rate=FS, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            task.start()

            while emg_running:
                chunk = np.array(task.read(number_of_samples_per_channel=CHUNK))
                filtered, bp_zi = sg.sosfilt(bp_sos, chunk, zi=bp_zi)
                filtered, notch_zi = sg.lfilter(notch_b, notch_a, filtered, zi=notch_zi)
                buffer = np.concatenate([buffer, filtered])

                if len(buffer) >= WIN:
                    w = buffer[:WIN]
                    buffer = buffer[WIN:]
                    latest_rms = float(np.sqrt(np.mean(w**2)))
                    plot_buffer.extend(w[::PLOT_DOWNSAMPLE])
    except Exception as e:
        print(f"EMG thread error: {e}")


def run_dashboard(movement_name, side):
    global emg_running
    a_idx, b_idx, c_idx = MOVEMENTS[movement_name][side]

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=resource_path("pose_landmarker_lite.task")),
        running_mode=VisionRunningMode.VIDEO)

    emg_running = True
    threading.Thread(target=emg_thread, daemon=True).start()

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

            # ---- EMG waveform strip ----
            plot_x, plot_y, plot_w, plot_h = 10, 90, 400, 100
            cv2.rectangle(frame, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (20, 20, 20), -1)
            cv2.rectangle(frame, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (255, 255, 255), 1)

            if len(plot_buffer) > 1:
                vals = np.array(plot_buffer)
                v_range = max(np.max(np.abs(vals)), 0.1)
                n = len(vals)
                xs = np.linspace(plot_x, plot_x + plot_w, n).astype(int)
                ys = (plot_y + plot_h/2 - (vals / v_range) * (plot_h/2 - 5)).astype(int)
                pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=1)

            cv2.putText(frame, f"EMG RMS: {latest_rms:.3f}", (plot_x + plot_w + 10, plot_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('EMG-CV Dashboard', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    emg_running = False
    cap.release()
    cv2.destroyAllWindows()


def start_dashboard():
    movement = movement_var.get()
    side = "L" if side_var.get() == "Left" else "R"
    root.withdraw()
    run_dashboard(movement, side)
    root.deiconify()


root = tk.Tk()
root.title("EMG-CV Fatigue & Strength Dashboard")
root.geometry("400x250")
root.configure(bg="#1e1e1e")

tk.Label(root, text="Muscle Movement Tracker", font=("Segoe UI", 16, "bold"),
         fg="white", bg="#1e1e1e").pack(pady=20)

tk.Label(root, text="Select Movement:", fg="white", bg="#1e1e1e").pack()
movement_var = tk.StringVar(value=list(MOVEMENTS.keys())[0])
ttk.Combobox(root, textvariable=movement_var, values=list(MOVEMENTS.keys()), state="readonly", width=30).pack(pady=5)

tk.Label(root, text="Select Side:", fg="white", bg="#1e1e1e").pack()
side_var = tk.StringVar(value="Left")
ttk.Combobox(root, textvariable=side_var, values=["Left", "Right"], state="readonly", width=30).pack(pady=5)

tk.Button(root, text="Start Dashboard", font=("Segoe UI", 12, "bold"),
          bg="#2ecc71", fg="white", command=start_dashboard, padx=20, pady=10).pack(pady=25)

root.mainloop()