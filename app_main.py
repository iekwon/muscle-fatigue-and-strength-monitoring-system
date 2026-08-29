import sys
import os
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

MOVEMENTS = {
    "Elbow (Biceps Curl)": {"L": (11, 13, 15), "R": (12, 14, 16)},
    "Shoulder": {"L": (13, 11, 23), "R": (14, 12, 24)},
    "Knee (Squat/Lunge)": {"L": (23, 25, 27), "R": (24, 26, 28)},
    "Hip Flexion": {"L": (11, 23, 25), "R": (12, 24, 26)},
    "Ankle": {"L": (25, 27, 31), "R": (26, 28, 32)},
}

ACCENT = "#00e5ff"
BG = "#0f0f14"
PANEL = "#1a1a22"
TEXT_DIM = "#7d8590"

# ---- EMG (NI DAQ) config -- set SIMULATE=True if no DAQ connected ----
SIMULATE_EMG = True  # flip to False on the real machine with DAQ wired up
CHANNEL = "Dev1/ai0"
FS = 2500
CHUNK = 100
WIN = int (FS * 0.2)
PLOT_SECONDS = 3
PLOT_DOWNSAMPLE = 5


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def angle_to_color(angle, min_angle=30, max_angle=170):
    t = np.clip((angle - min_angle) / (max_angle - min_angle), 0, 1)
    return (int(255*(1-t)), int(255*t), 0)  # BGR: red(low)->green(high)


class SplashScreen(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background-color: {BG};")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        logo = QtWidgets.QLabel("⚡")
        logo.setStyleSheet(f"font-size: 72px; color: {ACCENT};")
        logo.setAlignment(QtCore.Qt.AlignCenter)

        title = QtWidgets.QLabel("MYOTRACK")
        title.setStyleSheet("""
            font-size: 32px; font-weight: 700; letter-spacing: 6px;
            color: #ffffff; font-family: 'Segoe UI';
        """)
        title.setAlignment(QtCore.Qt.AlignCenter)

        subtitle = QtWidgets.QLabel("Welcome")
        subtitle.setStyleSheet(f"font-size: 16px; color: {TEXT_DIM}; font-family: 'Segoe UI';")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)

        layout.addStretch()
        layout.addWidget(logo)
        layout.addSpacing(10)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()

        QtCore.QTimer.singleShot(5000, self.finished.emit)


class OptionsScreen(QtWidgets.QWidget):
    start_requested = QtCore.pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background-color: {BG};")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setSpacing(18)

        title = QtWidgets.QLabel("Set Up Your Session")
        title.setStyleSheet("font-size: 24px; font-weight: 600; color: white; font-family: 'Segoe UI';")
        title.setAlignment(QtCore.Qt.AlignCenter)

        combo_style = f"""
            QComboBox {{
                background-color: {PANEL}; color: white; border: 1px solid #2a2a35;
                border-radius: 8px; padding: 10px 14px; font-size: 14px; min-width: 280px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: {PANEL}; color: white;
                selection-background-color: {ACCENT}; selection-color: black;
            }}
        """

        movement_label = QtWidgets.QLabel("MOVEMENT")
        movement_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; letter-spacing: 2px;")
        movement_label.setAlignment(QtCore.Qt.AlignCenter)

        self.movement_combo = QtWidgets.QComboBox()
        self.movement_combo.addItems(list(MOVEMENTS.keys()))
        self.movement_combo.setStyleSheet(combo_style)

        side_label = QtWidgets.QLabel("SIDE")
        side_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; letter-spacing: 2px;")
        side_label.setAlignment(QtCore.Qt.AlignCenter)

        self.side_combo = QtWidgets.QComboBox()
        self.side_combo.addItems(["Left", "Right"])
        self.side_combo.setStyleSheet(combo_style)

        start_btn = QtWidgets.QPushButton("Start Session")
        start_btn.setCursor(QtCore.Qt.PointingHandCursor)
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT}; color: #0f0f14; font-size: 15px; font-weight: 700;
                border-radius: 10px; padding: 12px 30px; margin-top: 10px;
            }}
            QPushButton:hover {{ background-color: #33eaff; }}
        """)
        start_btn.clicked.connect(self._on_start)

        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(movement_label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.movement_combo, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(side_label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.side_combo, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(start_btn, alignment=QtCore.Qt.AlignCenter)
        layout.addStretch()

    def _on_start(self):
        movement = self.movement_combo.currentText()
        side = "L" if self.side_combo.currentText() == "Left" else "R"
        self.start_requested.emit(movement, side)


class EmgWorker(QtCore.QThread):
    """Runs EMG acquisition + filtering in a background Qt thread.
    Emits (rms, waveform_chunk) so the GUI thread never blocks on hardware I/O."""
    new_sample = QtCore.pyqtSignal(float, float, np.ndarray)

    def __init__(self):
        super().__init__()
        self._running = True
        self._buffer = np.empty(0)
        self.paused = False

    def run(self):
        from scipy import signal as sg
        nyq = FS / 2
        bp_sos = sg.butter(4, [20/nyq, 450/nyq], btype='bandpass', output='sos')
        notch_b, notch_a = sg.iirnotch(50/nyq, 30)
        bp_zi = sg.sosfilt_zi(bp_sos)
        notch_zi = sg.lfilter_zi(notch_b, notch_a)

        if SIMULATE_EMG:
            rng = np.random.default_rng()
            t = 0.0
            while self._running:
                if self.paused:
                    QtCore.QThread.msleep(50)
                    continue
                # fabricate a plausible bursty EMG-like chunk for development without hardware
                envelope = 0.3 + 0.7 * abs(np.sin(t / 3.0))
                chunk = rng.normal(0, envelope, CHUNK)
                t += CHUNK / FS
                filtered, bp_zi = sg.sosfilt(bp_sos, chunk, zi=bp_zi)
                filtered, notch_zi = sg.lfilter(notch_b, notch_a, filtered, zi=notch_zi)
                self._buffer = np.concatenate([self._buffer, filtered])
                if len(self._buffer) >= WIN:
                    w = self._buffer[:WIN]
                    self._buffer = self._buffer[WIN:]
                    rms = float(np.sqrt(np.mean(w**2)))
                    freqs, psd = sg.welch(w, fs=FS, nperseg=min(256, len(w)))
                    cum = np.cumsum(psd)
                    mdf = float(freqs[min(np.searchsorted(cum, cum[-1]/2), len(freqs)-1)]) if cum[-1] > 1e-12 else 0.0
                    self.new_sample.emit(rms, mdf, w[::PLOT_DOWNSAMPLE])
                QtCore.QThread.msleep(int(1000 * CHUNK / FS))
        else:
            import nidaqmx
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(CHANNEL)
                task.timing.cfg_samp_clk_timing(rate=FS, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
                task.start()
                while self._running:
                    if self.paused:
                        QtCore.QThread.msleep(50)
                        continue
                    chunk = np.array(task.read(number_of_samples_per_channel=CHUNK))
                    filtered, bp_zi = sg.sosfilt(bp_sos, chunk, zi=bp_zi)
                    filtered, notch_zi = sg.lfilter(notch_b, notch_a, filtered, zi=notch_zi)
                    self._buffer = np.concatenate([self._buffer, filtered])
                    if len(self._buffer) >= WIN:
                        w = self._buffer[:WIN]
                        self._buffer = self._buffer[WIN:]
                        rms = float(np.sqrt(np.mean(w**2)))
                        freqs, psd = sg.welch(w, fs=FS, nperseg=min(256, len(w)))
                        cum = np.cumsum(psd)
                        mdf = float(freqs[min(np.searchsorted(cum, cum[-1]/2), len(freqs)-1)]) if cum[-1] > 1e-12 else 0.0
                        self.new_sample.emit(rms, mdf, w[::PLOT_DOWNSAMPLE])
    def stop(self):
        self._running = False
        self.wait(1000)


class DashboardScreen(QtWidgets.QWidget):
    back_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background-color: {BG};")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.history = {"t": [], "rms": [], "mdf": [], "angle": []}
        self.session_start = None

        self.movement_name = None
        self.side = None
        self.landmarker = None
        self.cap = None
        self.frame_idx = 0
        self.latest_rms = 0.0
        self.plot_buffer = deque(maxlen=int(FS * PLOT_SECONDS / PLOT_DOWNSAMPLE))

        # ---- Layout: camera on the left (fixed size, NOT fullscreen), info panel on the right ----
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(20)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(900, 675)
        self.video_label.setStyleSheet("background-color: black; border-radius: 12px; border: 1px solid #2a2a35;")
        root.addWidget(self.video_label, alignment=QtCore.Qt.AlignTop)

        side_panel = QtWidgets.QVBoxLayout()
        side_panel.setSpacing(14)

                # ---- Metric buttons: each opens its own popup window, pauses session while open ----
        metric_row = QtWidgets.QHBoxLayout()
        for metric_key, metric_label in [("rms", "RMS"), ("mdf", "MDF"), ("angle", "Angle")]:
            btn = QtWidgets.QPushButton(metric_label)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{ background-color: {PANEL}; color: white; border-radius: 8px;
                               padding: 8px 14px; font-size: 12px; border: 1px solid #2a2a35; }}
                QPushButton:hover {{ background-color: #26262f; }}
            """)
            btn.clicked.connect(lambda checked, k=metric_key, l=metric_label: self._open_metric_dialog(k, l))
            metric_row.addWidget(btn)
        side_panel.addLayout(metric_row)

    def start_session(self, movement_name, side):
        self.movement_name = movement_name
        self.side = side

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
            running_mode=VisionRunningMode.VIDEO)
        self.landmarker = PoseLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        self.frame_idx = 0

        self.emg_worker = EmgWorker()
        self.emg_worker.new_sample.connect(self._on_emg_sample)
        self.emg_worker.start()

        self.timer.start(30)  # ~33fps

    def stop_session(self):
        self.timer.stop()
        if self.emg_worker:
            self.emg_worker.stop()
            self.emg_worker = None
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

    def _on_back(self):
        self.stop_session()
        self.back_requested.emit()

    def _on_emg_sample(self, rms, mdf, waveform_chunk):
        import time
        self.latest_rms = rms
        self.plot_buffer.extend(waveform_chunk)
        self.rms_value.setText(f"{rms:.3f}")
        if self.session_start is None:
            self.session_start = time.time()
        self.history["t"].append(time.time() - self.session_start)
        self.history["rms"].append(rms)
        self.history["mdf"].append(mdf)
        self.history["angle"].append(self._last_angle if hasattr(self, "_last_angle") else 0)

    def _open_metric_dialog(self, metric_key, metric_label):
        # Pause camera + EMG processing while the graph is open
        self.timer.stop()
        if self.emg_worker:
            self.emg_worker.paused = True

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(metric_label)
        dlg.resize(600, 420)
        dlg.setStyleSheet(f"background-color: {BG};")
        layout = QtWidgets.QVBoxLayout(dlg)

        figure = Figure(figsize=(5, 3.2), facecolor=PANEL)
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)
        ax.plot(self.history["t"], self.history[metric_key], color=ACCENT)
        ax.set_title(metric_label, color="white", fontsize=12)
        ax.set_xlabel("time (s)", color=TEXT_DIM, fontsize=9)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color("#2a2a35")
        figure.tight_layout()
        layout.addWidget(canvas)

        dlg.exec_()  # blocks here until closed -- camera/EMG stay paused the whole time

        # Resume after the dialog closes
        if self.emg_worker:
            self.emg_worker.paused = False
        self.timer.start(30)
        
    def _update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        a_idx, b_idx, c_idx = MOVEMENTS[self.movement_name][self.side]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(self.frame_idx * (1000/30))
        self.frame_idx += 1
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        h, w, _ = frame.shape

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            a = [lm[a_idx].x, lm[a_idx].y]
            b = [lm[b_idx].x, lm[b_idx].y]
            c = [lm[c_idx].x, lm[c_idx].y]
            angle = calculate_angle(a, b, c)
            color = angle_to_color(angle)

            pa = (int(a[0]*w), int(a[1]*h))
            pb = (int(b[0]*w), int(b[1]*h))
            pc = (int(c[0]*w), int(c[1]*h))
            cv2.line(frame, pa, pb, color, 6, cv2.LINE_AA)
            cv2.line(frame, pb, pc, color, 6, cv2.LINE_AA)
            for pt in [pa, pb, pc]:
                cv2.circle(frame, pt, 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 8, (0, 0, 0), 2, cv2.LINE_AA)

            self.angle_value.setText(f"{angle:.0f}°")

        # ---- Transparent EMG waveform: line only, no background box ----
        if len(self.plot_buffer) > 1:
            vals = np.array(self.plot_buffer)
            v_range = max(np.max(np.abs(vals)), 0.1)
            n = len(vals)
            plot_x, plot_y, plot_w, plot_h = 10, h - 110, w - 20, 90
            xs = np.linspace(plot_x, plot_x + plot_w, n).astype(int)
            ys = (plot_y + plot_h/2 - (vals / v_range) * (plot_h/2 - 5)).astype(int)
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 229, 255), thickness=2, lineType=cv2.LINE_AA)

        rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb_out.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(), QtCore.Qt.KeepAspectRatio))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MyoTrack")
        self.resize(1000, 700)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.splash = SplashScreen()
        self.options = OptionsScreen()
        self.dashboard = DashboardScreen()

        self.stack.addWidget(self.splash)     # index 0
        self.stack.addWidget(self.options)    # index 1
        self.stack.addWidget(self.dashboard)  # index 2

        self.splash.finished.connect(lambda: self.stack.setCurrentIndex(1))
        self.options.start_requested.connect(self.on_start_session)
        self.dashboard.back_requested.connect(lambda: self.stack.setCurrentIndex(1))

    def on_start_session(self, movement, side):
        self.dashboard.start_session(movement, side)
        self.stack.setCurrentIndex(2)

    def closeEvent(self, event):
        self.dashboard.stop_session()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())