import sys
import time
import threading
import queue
import csv
import socket
import json
from datetime import datetime
from collections import deque

import numpy as np
import cv2
import mediapipe as mp

from PyQt5 import QtWidgets, QtCore, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pyqtgraph as pg


# ============================================================
# MYOTRACK - Camera + IMU ROM + EMG Dashboard
# ============================================================

PI_IP = "192.168.137.50"
PI_PORT = 6000


# ============================================================
# Shared Raspberry Pi data
# ============================================================

pi_data_lock = threading.Lock()

pi_data = {
    "connected": False,

    # IMU
    "angle_deg": 0.0,
    "rom_max_deg": 0.0,
    "rom_min_deg": 0.0,
    "imu_calibrated": False,
    "imu_calibration_status": "Not calibrated",

    # EMG
    "emg_stale": True,
    "emg_rate_sps": 0.0,
    "emg_latest_adc": 0,
    "emg_sample_count": 0,
    "emg_packet_count": 0,
    "emg_sample_gaps": 0,
    "emg_packet_gaps": 0,
    "emg_malformed": 0,

    # Raw EMG samples
    "emg_samples": [],
}


# ============================================================
# Movement definitions
# ============================================================

MOVEMENTS = {
    "Elbow (Biceps Curl)": {
        "L": (11, 13, 15),
        "R": (12, 14, 16),
    },

    "Shoulder": {
        "L": (13, 11, 23),
        "R": (14, 12, 24),
    },

    "Knee (Squat/Lunge)": {
        "L": (23, 25, 27),
        "R": (24, 26, 28),
    },

    "Hip Flexion": {
        "L": (11, 23, 25),
        "R": (12, 24, 26),
    },

    "Ankle": {
        "L": (25, 27, 31),
        "R": (26, 28, 32),
    },
}


# ============================================================
# UI colors
# ============================================================

ACCENT = "#00e5ff"
BG = "#0f0f14"
PANEL = "#1a1a22"
TEXT_DIM = "#7d8590"
GREEN = "#38d996"
RED = "#ff5555"
YELLOW = "#ffd166"


# ============================================================
# EMG configuration
# ============================================================

# Actual EMG sampling:
#
# Arduino Nano -> 2000 Hz
# Nano -> ESP32 #1 -> Raspberry Pi -> Windows
#
# Windows processes the incoming samples.
# No NI-DAQ is used.

EMG_FS = 2000

# Pi sends approximately 100 samples every 50 ms.
EMG_CHUNK = 100

# 200 ms analysis window
EMG_WINDOW_SECONDS = 0.2
EMG_WINDOW_SAMPLES = int(
    EMG_FS * EMG_WINDOW_SECONDS
)

# Display approximately 3 seconds of waveform.
EMG_PLOT_SECONDS = 3

# Display downsampling only.
# Signal processing still uses all 2000 Hz samples.
EMG_PLOT_DOWNSAMPLE = 5

# Bandpass
EMG_LOW_CUTOFF = 20
EMG_HIGH_CUTOFF = 450

# Notch
EMG_NOTCH_FREQ = 50
EMG_NOTCH_Q = 30


# ============================================================
# Rep detection
# ============================================================

REP_MIN_ANGLE = 40.0
REP_MAX_ANGLE = 150.0
REP_MIN_TRAVEL = 25.0
REP_COOLDOWN = 0.45
REP_SMOOTHING = 5


# ============================================================
# Utility functions
# ============================================================

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(
            c[1] - b[1],
            c[0] - b[0],
        )
        -
        np.arctan2(
            a[1] - b[1],
            a[0] - b[0],
        )
    )

    angle = np.abs(
        radians * 180.0 / np.pi
    )

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def angle_to_color(
    angle,
    min_angle=30,
    max_angle=170,
):

    t = np.clip(
        (angle - min_angle)
        / (max_angle - min_angle),
        0,
        1,
    )

    return (
        int(255 * (1 - t)),
        int(255 * t),
        0,
    )


# ============================================================
# Splash screen
# ============================================================

class SplashScreen(QtWidgets.QWidget):

    finished = QtCore.pyqtSignal()

    def __init__(self):

        super().__init__()

        self.setAutoFillBackground(True)

        self.setStyleSheet(
            f"background-color: {BG};"
        )

        self.setAttribute(
            QtCore.Qt.WA_StyledBackground,
            True,
        )

        layout = QtWidgets.QVBoxLayout(self)

        layout.setAlignment(
            QtCore.Qt.AlignCenter
        )

        logo = QtWidgets.QLabel("⚡")

        logo.setStyleSheet(
            f"""
            font-size: 72px;
            color: {ACCENT};
            """
        )

        logo.setAlignment(
            QtCore.Qt.AlignCenter
        )

        title = QtWidgets.QLabel(
            "MYOTRACK"
        )

        title.setStyleSheet(
            """
            font-size: 32px;
            font-weight: 700;
            letter-spacing: 6px;
            color: #ffffff;
            font-family: 'Segoe UI';
            """
        )

        title.setAlignment(
            QtCore.Qt.AlignCenter
        )

        subtitle = QtWidgets.QLabel(
            "Welcome"
        )

        subtitle.setStyleSheet(
            f"""
            font-size: 16px;
            color: {TEXT_DIM};
            font-family: 'Segoe UI';
            """
        )

        subtitle.setAlignment(
            QtCore.Qt.AlignCenter
        )

        layout.addStretch()
        layout.addWidget(logo)
        layout.addSpacing(10)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()

        self.splash_timer = QtCore.QTimer(
            self
        )

        self.splash_timer.setSingleShot(
            True
        )

        self.splash_timer.timeout.connect(
            self.finished.emit
        )

        self.splash_timer.start(2000)


# ============================================================
# Options screen
# ============================================================

class OptionsScreen(QtWidgets.QWidget):

    start_requested = QtCore.pyqtSignal(
        str,
        str,
    )

    def __init__(self):

        super().__init__()

        self.setAutoFillBackground(True)

        self.setStyleSheet(
            f"background-color: {BG};"
        )

        self.setAttribute(
            QtCore.Qt.WA_StyledBackground,
            True,
        )

        layout = QtWidgets.QVBoxLayout(
            self
        )

        layout.setAlignment(
            QtCore.Qt.AlignCenter
        )

        layout.setSpacing(18)

        title = QtWidgets.QLabel(
            "Set Up Your Session"
        )

        title.setStyleSheet(
            """
            font-size: 24px;
            font-weight: 600;
            color: white;
            font-family: 'Segoe UI';
            """
        )

        title.setAlignment(
            QtCore.Qt.AlignCenter
        )

        combo_style = f"""
            QComboBox {{
                background-color: {PANEL};
                color: white;
                border: 1px solid #2a2a35;
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 14px;
                min-width: 280px;
            }}

            QComboBox::drop-down {{
                border: none;
            }}

            QComboBox QAbstractItemView {{
                background-color: {PANEL};
                color: white;
                selection-background-color: {ACCENT};
                selection-color: black;
            }}
        """

        movement_label = QtWidgets.QLabel(
            "MOVEMENT"
        )

        movement_label.setStyleSheet(
            f"""
            color: {TEXT_DIM};
            font-size: 11px;
            letter-spacing: 2px;
            """
        )

        movement_label.setAlignment(
            QtCore.Qt.AlignCenter
        )

        self.movement_combo = (
            QtWidgets.QComboBox()
        )

        self.movement_combo.addItems(
            list(MOVEMENTS.keys())
        )

        self.movement_combo.setStyleSheet(
            combo_style
        )

        side_label = QtWidgets.QLabel(
            "SIDE"
        )

        side_label.setStyleSheet(
            f"""
            color: {TEXT_DIM};
            font-size: 11px;
            letter-spacing: 2px;
            """
        )

        side_label.setAlignment(
            QtCore.Qt.AlignCenter
        )

        self.side_combo = (
            QtWidgets.QComboBox()
        )

        self.side_combo.addItems(
            ["Left", "Right"]
        )

        self.side_combo.setStyleSheet(
            combo_style
        )

        start_btn = QtWidgets.QPushButton(
            "Start Session"
        )

        start_btn.setCursor(
            QtCore.Qt.PointingHandCursor
        )

        start_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {ACCENT};
                color: #0f0f14;
                font-size: 15px;
                font-weight: 700;
                border-radius: 10px;
                padding: 12px 30px;
                margin-top: 10px;
            }}

            QPushButton:hover {{
                background-color: #33eaff;
            }}
            """
        )

        start_btn.clicked.connect(
            self._on_start
        )

        layout.addStretch()

        layout.addWidget(
            title
        )

        layout.addSpacing(20)

        layout.addWidget(
            movement_label,
            alignment=QtCore.Qt.AlignCenter,
        )

        layout.addWidget(
            self.movement_combo,
            alignment=QtCore.Qt.AlignCenter,
        )

        layout.addWidget(
            side_label,
            alignment=QtCore.Qt.AlignCenter,
        )

        layout.addWidget(
            self.side_combo,
            alignment=QtCore.Qt.AlignCenter,
        )

        layout.addWidget(
            start_btn,
            alignment=QtCore.Qt.AlignCenter,
        )

        layout.addStretch()

    def _on_start(self):

        movement = (
            self.movement_combo.currentText()
        )

        side = (
            "L"
            if self.side_combo.currentText()
            == "Left"
            else "R"
        )

        self.start_requested.emit(
            movement,
            side,
        )


# ============================================================
# Raspberry Pi TCP worker
# ============================================================

class PiDataWorker(threading.Thread):

    def __init__(self):

        super().__init__(
            daemon=True
        )

        self.running = True

        self.sock = None

        # Protect socket writes.
        #
        # The receive loop runs in this worker thread,
        # while the Qt button can call send_command()
        # from the GUI thread.
        self.send_lock = threading.Lock()

    # --------------------------------------------------------
    # Stop worker
    # --------------------------------------------------------

    def stop(self):

        self.running = False

        with self.send_lock:

            sock = self.sock

            self.sock = None

        if sock:

            try:

                sock.shutdown(
                    socket.SHUT_RDWR
                )

            except OSError:

                pass

            try:

                sock.close()

            except OSError:

                pass

    # --------------------------------------------------------
    # Send command to Raspberry Pi
    # --------------------------------------------------------

    def send_command(
        self,
        command,
    ):

        payload = (
            json.dumps(
                {
                    "command": command
                },
                separators=(",", ":"),
            )
            + "\n"
        )

        with self.send_lock:

            sock = self.sock

            if (
                sock is None
                or not self.running
            ):

                return False

            try:

                sock.sendall(
                    payload.encode(
                        "utf-8"
                    )
                )

                print(
                    "PI COMMAND SENT:",
                    command,
                )

                return True

            except OSError as exc:

                print(
                    "PI COMMAND ERROR:",
                    exc,
                )

                return False

    # --------------------------------------------------------
    # Receive data from Raspberry Pi
    # --------------------------------------------------------

    def run(self):

        global pi_data

        while self.running:

            sock = None

            try:

                sock = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                )

                sock.settimeout(5)

                sock.connect(
                    (
                        PI_IP,
                        PI_PORT,
                    )
                )

                sock.settimeout(None)

                with self.send_lock:

                    self.sock = sock

                print(
                    "PI CONNECTED"
                )

                with pi_data_lock:

                    pi_data[
                        "connected"
                    ] = True

                buffer = ""

                while self.running:

                    chunk = sock.recv(
                        16384
                    )

                    if not chunk:

                        break

                    buffer += chunk.decode(
                        "utf-8",
                        errors="replace",
                    )

                    while "\n" in buffer:

                        line, buffer = (
                            buffer.split(
                                "\n",
                                1,
                            )
                        )

                        if not line.strip():

                            continue

                        try:

                            data = json.loads(
                                line
                            )

                        except json.JSONDecodeError as exc:

                            print(
                                "PI JSON ERROR:",
                                exc,
                            )

                            continue

                        if not isinstance(
                            data,
                            dict,
                        ):

                            continue

                        with pi_data_lock:

                            pi_data.update(
                                data
                            )

                        # ------------------------------------------------
                        # Print calibration acknowledgements.
                        # Useful during testing.
                        # ------------------------------------------------

                        if (
                            data.get("type")
                            == "command_ack"
                            and
                            data.get("command")
                            == "calibrate_imu"
                        ):

                            print(
                                "PI CALIBRATION ACK:",
                                data,
                            )

                with pi_data_lock:

                    pi_data[
                        "connected"
                    ] = False

                with self.send_lock:

                    if self.sock is sock:

                        self.sock = None

                try:

                    sock.close()

                except OSError:

                    pass

            except Exception as exc:

                if self.running:

                    print(
                        "PI WORKER ERROR:",
                        exc,
                    )

                with pi_data_lock:

                    pi_data[
                        "connected"
                    ] = False

                with self.send_lock:

                    if self.sock is sock:

                        self.sock = None

                if sock:

                    try:

                        sock.close()

                    except OSError:

                        pass

                if self.running:

                    time.sleep(1)


# ============================================================
# EMG processing worker
# ============================================================

class EmgWorker:

    """
    Processes actual EMG samples received from
    Raspberry Pi.

    Input:

        [sample_number, timestamp_us, adc_value]

    Sampling rate:

        2000 Hz

    No NI-DAQ is used.
    """

    def __init__(self):

        self.input_queue = queue.Queue(
            maxsize=100
        )

        self.result_queue = queue.Queue(
            maxsize=50
        )

        self.error_queue = queue.Queue(
            maxsize=5
        )

        self._running = True
        self.paused = False

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.last_sample_number = None

        self.received_samples = 0
        self.duplicate_samples = 0
        self.sequence_gaps = 0

    def start(self):

        self._thread.start()

    def push_samples(
        self,
        samples,
    ):

        if not samples:

            return

        try:

            self.input_queue.put_nowait(
                samples
            )

        except queue.Full:

            # Drop only a Windows processing chunk.
            # Pi continues receiving actual samples.
            pass

    def _run(self):

        try:

            self._run_impl()

        except Exception:

            import traceback

            try:

                self.error_queue.put_nowait(
                    traceback.format_exc()
                )

            except queue.Full:

                pass

    def _run_impl(self):

        from scipy import signal as sg

        # ----------------------------------------------------
        # Filter design
        # ----------------------------------------------------

        nyq = EMG_FS / 2.0

        bp_sos = sg.butter(
            4,
            [
                EMG_LOW_CUTOFF / nyq,
                EMG_HIGH_CUTOFF / nyq,
            ],
            btype="bandpass",
            output="sos",
        )

        notch_b, notch_a = sg.iirnotch(
            EMG_NOTCH_FREQ / nyq,
            EMG_NOTCH_Q,
        )

        bp_zi = sg.sosfilt_zi(
            bp_sos
        )

        notch_zi = sg.lfilter_zi(
            notch_b,
            notch_a,
        )

        analysis_buffer = np.empty(
            0,
            dtype=np.float64,
        )

        display_buffer = deque(
            maxlen=int(
                EMG_FS
                * EMG_PLOT_SECONDS
                / EMG_PLOT_DOWNSAMPLE
            )
        )

        def process_chunk(
            raw_adc,
        ):

            nonlocal \
                bp_zi, \
                notch_zi, \
                analysis_buffer

            raw_adc = np.asarray(
                raw_adc,
                dtype=np.float64,
            )

            if raw_adc.size == 0:

                return

            # ------------------------------------------------
            # Remove ADC DC offset
            # ------------------------------------------------

            centered = (
                raw_adc
                - np.mean(raw_adc)
            )

            # ------------------------------------------------
            # Bandpass
            # ------------------------------------------------

            filtered, bp_zi = (
                sg.sosfilt(
                    bp_sos,
                    centered,
                    zi=bp_zi,
                )
            )

            # ------------------------------------------------
            # 50 Hz notch
            # ------------------------------------------------

            filtered, notch_zi = (
                sg.lfilter(
                    notch_b,
                    notch_a,
                    filtered,
                    zi=notch_zi,
                )
            )

            # ------------------------------------------------
            # Analysis buffer
            # ------------------------------------------------

            analysis_buffer = np.concatenate(
                [
                    analysis_buffer,
                    filtered,
                ]
            )

            # ------------------------------------------------
            # Display waveform
            # ------------------------------------------------

            for value in filtered:

                display_buffer.append(
                    float(value)
                )

            # ------------------------------------------------
            # RMS / MDF
            # ------------------------------------------------

            while (
                len(analysis_buffer)
                >= EMG_WINDOW_SAMPLES
            ):

                window = (
                    analysis_buffer[
                        :EMG_WINDOW_SAMPLES
                    ]
                )

                analysis_buffer = (
                    analysis_buffer[
                        EMG_WINDOW_SAMPLES:
                    ]
                )

                rms = float(
                    np.sqrt(
                        np.mean(
                            window ** 2
                        )
                    )
                )

                freqs, psd = sg.welch(
                    window,
                    fs=EMG_FS,
                    nperseg=min(
                        512,
                        len(window),
                    ),
                )

                total_power = float(
                    np.sum(psd)
                )

                if total_power > 1e-12:

                    cumulative = (
                        np.cumsum(psd)
                    )

                    target = (
                        cumulative[-1]
                        / 2.0
                    )

                    index = min(
                        int(
                            np.searchsorted(
                                cumulative,
                                target,
                            )
                        ),
                        len(freqs) - 1,
                    )

                    mdf = float(
                        freqs[index]
                    )

                else:

                    mdf = 0.0

                waveform = np.asarray(
                    display_buffer,
                    dtype=np.float64,
                ).copy()

                try:

                    self.result_queue.put_nowait(
                        (
                            rms,
                            mdf,
                            waveform,
                        )
                    )

                except queue.Full:

                    pass

        # ----------------------------------------------------
        # Main processing loop
        # ----------------------------------------------------

        while self._running:

            try:

                samples = (
                    self.input_queue.get(
                        timeout=0.1
                    )
                )

            except queue.Empty:

                continue

            if self.paused:

                continue

            adc_values = []

            for sample in samples:

                if (
                    not isinstance(
                        sample,
                        (list, tuple),
                    )
                    or len(sample) < 3
                ):

                    continue

                try:

                    sample_number = int(
                        sample[0]
                    )

                    adc_value = float(
                        sample[2]
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    continue

                # --------------------------------------------
                # Detect duplicates / gaps
                # --------------------------------------------

                if (
                    self.last_sample_number
                    is not None
                ):

                    delta = (
                        sample_number
                        -
                        self.last_sample_number
                    )

                    if delta <= 0:

                        self.duplicate_samples += 1

                        continue

                    if delta > 1:

                        self.sequence_gaps += (
                            delta - 1
                        )

                self.last_sample_number = (
                    sample_number
                )

                self.received_samples += 1

                adc_values.append(
                    adc_value
                )

            if adc_values:

                process_chunk(
                    adc_values
                )

    def stop(self):

        self._running = False

        if self._thread.is_alive():

            self._thread.join(
                timeout=1.5
            )


# ============================================================
# Dashboard
# ============================================================

class DashboardScreen(
    QtWidgets.QWidget
):

    back_requested = QtCore.pyqtSignal()

    def __init__(self):

        super().__init__()

        self.setAutoFillBackground(
            True
        )

        self.setStyleSheet(
            f"background-color: {BG};"
        )

        self.setAttribute(
            QtCore.Qt.WA_StyledBackground,
            True,
        )

        # ----------------------------------------------------
        # Session history
        # ----------------------------------------------------

        self.history = {
            "t": [],

            # Camera
            "camera_angle": [],
            "camera_rom": [],

            # IMU
            "imu_angle": [],
            "imu_rom": [],

            # EMG
            "rms": [],
            "mdf": [],
        }

        self.session_start = None

        self.session_elapsed_before_pause = (
            0.0
        )

        self.session_paused_at = None

        self.movement_name = None
        self.side = None

        # ----------------------------------------------------
        # Camera / MediaPipe
        # ----------------------------------------------------

        self.landmarker = None
        self.cap = None

        self.frame_idx = 0
        self.last_frame_time = (
            time.monotonic()
        )

        # ----------------------------------------------------
        # EMG
        # ----------------------------------------------------

        self.latest_rms = 0.0
        self.latest_mdf = 0.0

        self.emg_worker = None
        self.pi_worker = None

        self.plot_buffer = deque(
            maxlen=int(
                EMG_FS
                * EMG_PLOT_SECONDS
                / EMG_PLOT_DOWNSAMPLE
            )
        )

        # ----------------------------------------------------
        # Camera ROM
        # ----------------------------------------------------

        self.camera_angle_min = None
        self.camera_angle_max = None
        self.camera_rom = 0.0

        # ----------------------------------------------------
        # IMU ROM
        # ----------------------------------------------------

        self.pi_angle = None
        self.pi_rom_max = 0.0
        self.pi_rom_min = 0.0
        self.pi_rom = 0.0

        # ----------------------------------------------------
        # IMU calibration state
        # ----------------------------------------------------

        self.imu_calibrated = False

        self.imu_calibration_status = (
            "Not calibrated"
        )

        self.imu_calibration_pending = False

        # ----------------------------------------------------
        # Repetition detection
        # ----------------------------------------------------

        self._last_angle = 0.0

        self.angle_history = deque(
            maxlen=REP_SMOOTHING
        )

        self.rep_count = 0
        self.rep_phase = "waiting"

        self.rep_high = None
        self.rep_low = None

        self.last_rep_time = 0.0

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.camera_ok = False
        self.emg_ok = False

        # ----------------------------------------------------
        # Timers
        # ----------------------------------------------------

        self.timer = QtCore.QTimer()

        self.timer.timeout.connect(
            self._update_frame
        )

        self.emg_poll_timer = (
            QtCore.QTimer()
        )

        self.emg_poll_timer.timeout.connect(
            self._poll_emg
        )

        self.session_timer = (
            QtCore.QTimer()
        )

        self.session_timer.timeout.connect(
            self._update_session_time
        )

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self._build_ui()

    # ========================================================
    # UI
    # ========================================================

    def _build_ui(self):

        root = QtWidgets.QVBoxLayout(
            self
        )

        root.setContentsMargins(
            18,
            18,
            18,
            18,
        )

        root.setSpacing(12)

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = QtWidgets.QHBoxLayout()

        title = QtWidgets.QLabel(
            "MYOTRACK"
        )

        title.setStyleSheet(
            f"""
            font-size: 22px;
            font-weight: 700;
            color: {ACCENT};
            """
        )

        self.session_status = (
            QtWidgets.QLabel(
                "READY"
            )
        )

        self.session_status.setStyleSheet(
            f"""
            color: {GREEN};
            background: {PANEL};
            border: 1px solid #2a2a35;
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 11px;
            font-weight: 700;
            """
        )

        header.addWidget(title)
        header.addStretch()
        header.addWidget(
            self.session_status
        )

        root.addLayout(header)

        # ----------------------------------------------------
        # Main top section
        # ----------------------------------------------------

        top = QtWidgets.QHBoxLayout()

        top.setSpacing(14)

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.video_label = (
            QtWidgets.QLabel()
        )

        self.video_label.setMinimumSize(
            600,
            450,
        )

        self.video_label.setStyleSheet(
            """
            background-color: black;
            border-radius: 12px;
            border: 1px solid #2a2a35;
            """
        )

        self.video_label.setAlignment(
            QtCore.Qt.AlignCenter
        )

        self.video_label.setText(
            "Camera not started"
        )

        top.addWidget(
            self.video_label,
            3,
        )

        # ----------------------------------------------------
        # Right panel
        # ----------------------------------------------------

        side_panel = (
            QtWidgets.QVBoxLayout()
        )

        side_panel.setSpacing(8)

        back_btn = (
            QtWidgets.QPushButton(
                "← Back"
            )
        )

        back_btn.setCursor(
            QtCore.Qt.PointingHandCursor
        )

        back_btn.setStyleSheet(
            self._secondary_button_style()
        )

        back_btn.clicked.connect(
            self._on_back
        )

        side_panel.addWidget(
            back_btn,
            alignment=QtCore.Qt.AlignLeft,
        )

        session_info = (
            QtWidgets.QLabel(
                "SESSION"
            )
        )

        session_info.setStyleSheet(
            self._section_style()
        )

        side_panel.addWidget(
            session_info
        )

        self.movement_value = (
            self._value_label("--")
        )

        self.side_value = (
            self._value_label("--")
        )

        self.duration_value = (
            self._value_label("00:00")
        )

        side_panel.addWidget(
            self._label("MOVEMENT")
        )

        side_panel.addWidget(
            self.movement_value
        )

        side_panel.addWidget(
            self._label("SIDE")
        )

        side_panel.addWidget(
            self.side_value
        )

        side_panel.addWidget(
            self._label("DURATION")
        )

        side_panel.addWidget(
            self.duration_value
        )

        # ----------------------------------------------------
        # Camera ROM
        # ----------------------------------------------------

        side_panel.addSpacing(5)

        camera_title = (
            QtWidgets.QLabel(
                "CAMERA ROM"
            )
        )

        camera_title.setStyleSheet(
            self._section_style()
        )

        side_panel.addWidget(
            camera_title
        )

        self.camera_angle_value = (
            self._value_label("--°")
        )

        self.camera_rom_value = (
            self._value_label("--°")
        )

        side_panel.addWidget(
            self._label(
                "CURRENT CAMERA ANGLE"
            )
        )

        side_panel.addWidget(
            self.camera_angle_value
        )

        side_panel.addWidget(
            self._label(
                "CAMERA ROM"
            )
        )

        side_panel.addWidget(
            self.camera_rom_value
        )

        # ----------------------------------------------------
        # IMU ROM
        # ----------------------------------------------------

        side_panel.addSpacing(5)

        imu_title = (
            QtWidgets.QLabel(
                "IMU ROM"
            )
        )

        imu_title.setStyleSheet(
            self._section_style()
        )

        side_panel.addWidget(
            imu_title
        )

        self.imu_angle_value = (
            self._value_label("--°")
        )

        self.imu_rom_value = (
            self._value_label("--°")
        )

        side_panel.addWidget(
            self._label(
                "CURRENT IMU ANGLE"
            )
        )

        side_panel.addWidget(
            self.imu_angle_value
        )

        side_panel.addWidget(
            self._label(
                "IMU ROM"
            )
        )

        side_panel.addWidget(
            self.imu_rom_value
        )

        # ----------------------------------------------------
        # EMG
        # ----------------------------------------------------

        side_panel.addSpacing(5)

        emg_title = (
            QtWidgets.QLabel(
                "EMG"
            )
        )

        emg_title.setStyleSheet(
            self._section_style()
        )

        side_panel.addWidget(
            emg_title
        )

        self.rms_value = (
            self._value_label("--")
        )

        self.mdf_value = (
            self._value_label("-- Hz")
        )

        side_panel.addWidget(
            self._label(
                "EMG RMS"
            )
        )

        side_panel.addWidget(
            self.rms_value
        )

        side_panel.addWidget(
            self._label(
                "MEDIAN FREQUENCY"
            )
        )

        side_panel.addWidget(
            self.mdf_value
        )

        # ----------------------------------------------------
        # Repetitions
        # ----------------------------------------------------

        side_panel.addWidget(
            self._label(
                "REPETITIONS"
            )
        )

        self.rep_value = (
            self._value_label("0")
        )

        side_panel.addWidget(
            self.rep_value
        )

        # ----------------------------------------------------
        # Device status
        # ----------------------------------------------------

        side_panel.addSpacing(5)

        self.camera_status = (
            QtWidgets.QLabel(
                "● Camera: --"
            )
        )

        self.imu_status = (
            QtWidgets.QLabel(
                "● IMU: --"
            )
        )

        self.emg_status = (
            QtWidgets.QLabel(
                "● EMG: --"
            )
        )

        self.camera_status.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px;"
        )

        self.imu_status.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px;"
        )

        self.emg_status.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px;"
        )

        self.emg_status.setWordWrap(
            True
        )

        side_panel.addWidget(
            self.camera_status
        )

        side_panel.addWidget(
            self.imu_status
        )

        side_panel.addWidget(
            self.emg_status
        )

        side_panel.addStretch()

        top.addLayout(
            side_panel,
            1,
        )

        root.addLayout(
            top,
            1,
        )

        # ----------------------------------------------------
        # Controls
        # ----------------------------------------------------

        controls = (
            QtWidgets.QHBoxLayout()
        )

        self.pause_btn = (
            QtWidgets.QPushButton(
                "Pause"
            )
        )

        self.resume_btn = (
            QtWidgets.QPushButton(
                "Resume"
            )
        )

        # NEW:
        # IMU calibration button
        self.calibrate_imu_btn = (
            QtWidgets.QPushButton(
                "Calibrate IMU"
            )
        )

        self.stop_btn = (
            QtWidgets.QPushButton(
                "Stop Session"
            )
        )

        self.pause_btn.setStyleSheet(
            self._secondary_button_style()
        )

        self.resume_btn.setStyleSheet(
            self._secondary_button_style()
        )

        self.calibrate_imu_btn.setStyleSheet(
            self._secondary_button_style()
        )

        self.stop_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {RED};
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 700;
            }}

            QPushButton:hover {{
                background-color: #ff7777;
            }}
            """
        )

        self.pause_btn.clicked.connect(
            self.pause_session
        )

        self.resume_btn.clicked.connect(
            self.resume_session
        )

        self.calibrate_imu_btn.clicked.connect(
            self.calibrate_imu
        )

        self.stop_btn.clicked.connect(
            self.stop_session
        )

        controls.addWidget(
            self.pause_btn
        )

        controls.addWidget(
            self.resume_btn
        )

        # NEW:
        # Place calibration beside Pause / Resume.
        controls.addWidget(
            self.calibrate_imu_btn
        )

        controls.addStretch()

        for key, label in [
            (
                "rms",
                "RMS Graph",
            ),
            (
                "mdf",
                "MDF Graph",
            ),
            (
                "camera_angle",
                "Camera Angle Graph",
            ),
            (
                "imu_angle",
                "IMU Angle Graph",
            ),
            (
                "camera_rom",
                "Camera ROM Graph",
            ),
            (
                "imu_rom",
                "IMU ROM Graph",
            ),
        ]:

            btn = QtWidgets.QPushButton(
                label
            )

            btn.setStyleSheet(
                self._secondary_button_style()
            )

            btn.clicked.connect(
                lambda checked=False,
                k=key,
                l=label:
                self._open_metric_dialog(
                    k,
                    l,
                )
            )

            controls.addWidget(
                btn
            )

        controls.addWidget(
            self.stop_btn
        )

        root.addLayout(
            controls
        )

        # ----------------------------------------------------
        # EMG waveform
        # ----------------------------------------------------

        self.emg_plot = (
            pg.PlotWidget(
                background=BG
            )
        )

        self.emg_plot.setMinimumHeight(
            150
        )

        self.emg_plot.setMouseEnabled(
            x=False,
            y=False,
        )

        self.emg_plot.hideAxis(
            "bottom"
        )

        self.emg_plot.hideAxis(
            "left"
        )

        self.emg_plot.setYRange(
            -100,
            100,
        )

        self.emg_curve = (
            self.emg_plot.plot(
                pen=pg.mkPen(
                    color=ACCENT,
                    width=1.5,
                )
            )
        )

        root.addWidget(
            self.emg_plot
        )

    # ========================================================
    # UI helpers
    # ========================================================

    def _label(self, text):

        label = QtWidgets.QLabel(
            text
        )

        label.setStyleSheet(
            f"""
            color: {TEXT_DIM};
            font-size: 10px;
            letter-spacing: 2px;
            """
        )

        return label

    def _value_label(self, text):

        label = QtWidgets.QLabel(
            text
        )

        label.setStyleSheet(
            """
            color: white;
            font-size: 19px;
            font-weight: 700;
            """
        )

        return label

    def _section_style(self):

        return (
            f"""
            color: {ACCENT};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 2px;
            """
        )

    def _secondary_button_style(self):

        return f"""
            QPushButton {{
                background-color: {PANEL};
                color: white;
                border-radius: 8px;
                padding: 8px 14px;
                font-size: 12px;
                border: 1px solid #2a2a35;
            }}

            QPushButton:hover {{
                background-color: #26262f;
            }}

            QPushButton:disabled {{
                color: #666a72;
                background-color: #15151b;
            }}
        """

    # ========================================================
    # IMU CALIBRATION
    # ========================================================

    def calibrate_imu(self):

        # ----------------------------------------------------
        # Make sure a Pi worker exists.
        # ----------------------------------------------------

        if self.pi_worker is None:

            QtWidgets.QMessageBox.warning(
                self,
                "IMU Calibration",
                "Raspberry Pi connection is not active.",
            )

            return

        # ----------------------------------------------------
        # Check current TCP connection.
        # ----------------------------------------------------

        with pi_data_lock:

            connected = bool(
                pi_data.get(
                    "connected",
                    False,
                )
            )

        if not connected:

            QtWidgets.QMessageBox.warning(
                self,
                "IMU Calibration",
                "Raspberry Pi is not connected.",
            )

            return

        # ----------------------------------------------------
        # Prevent double-clicks.
        # ----------------------------------------------------

        if self.imu_calibration_pending:

            return

        # ----------------------------------------------------
        # Tell user what is happening.
        # ----------------------------------------------------

        self.imu_calibration_pending = True

        self.calibrate_imu_btn.setEnabled(
            False
        )

        self.calibrate_imu_btn.setText(
            "Calibrating..."
        )

        self._set_imu_status(
            "● IMU: Keep both IMUs still — calibrating...",
            YELLOW,
        )

        # ----------------------------------------------------
        # Send command to Pi.
        #
        # The Pi dashboard server receives:
        #
        # {"command":"calibrate_imu"}
        #
        # and calls:
        #
        # pi_rom_server.calibrate_zero()
        # ----------------------------------------------------

        success = (
            self.pi_worker.send_command(
                "calibrate_imu"
            )
        )

        if not success:

            self.imu_calibration_pending = (
                False
            )

            self.calibrate_imu_btn.setEnabled(
                True
            )

            self.calibrate_imu_btn.setText(
                "Calibrate IMU"
            )

            self._set_imu_status(
                "● IMU: Calibration command failed",
                RED,
            )

    # ========================================================
    # Session lifecycle
    # ========================================================

    def start_session(
        self,
        movement_name,
        side,
    ):

        self.stop_session(
            silent=True
        )

        self.movement_name = (
            movement_name
        )

        self.side = side

        self.movement_value.setText(
            movement_name
        )

        self.side_value.setText(
            "Left"
            if side == "L"
            else "Right"
        )

        # ----------------------------------------------------
        # Reset history
        # ----------------------------------------------------

        self.history = {
            "t": [],

            "camera_angle": [],
            "camera_rom": [],

            "imu_angle": [],
            "imu_rom": [],

            "rms": [],
            "mdf": [],
        }

        self.plot_buffer.clear()

        # ----------------------------------------------------
        # Session time
        # ----------------------------------------------------

        self.session_start = (
            time.monotonic()
        )

        self.session_elapsed_before_pause = (
            0.0
        )

        self.session_paused_at = None

        # ----------------------------------------------------
        # Camera ROM
        # ----------------------------------------------------

        self.camera_angle_min = None
        self.camera_angle_max = None
        self.camera_rom = 0.0

        # ----------------------------------------------------
        # IMU ROM
        # ----------------------------------------------------

        self.pi_angle = None
        self.pi_rom_max = 0.0
        self.pi_rom_min = 0.0
        self.pi_rom = 0.0

        # ----------------------------------------------------
        # IMU calibration state
        #
        # We do NOT automatically calibrate.
        #
        # User presses Calibrate IMU when the two sensors
        # are physically in the desired reference position.
        # ----------------------------------------------------

        self.imu_calibrated = False

        self.imu_calibration_status = (
            "Not calibrated"
        )

        self.imu_calibration_pending = False

        self.calibrate_imu_btn.setEnabled(
            False
        )

        self.calibrate_imu_btn.setText(
            "Calibrate IMU"
        )

        # ----------------------------------------------------
        # Repetitions
        # ----------------------------------------------------

        self._last_angle = 0.0

        self.angle_history.clear()

        self.rep_count = 0
        self.rep_phase = "waiting"

        self.rep_high = None
        self.rep_low = None

        self.last_rep_time = 0.0

        # ----------------------------------------------------
        # UI reset
        # ----------------------------------------------------

        self.rep_value.setText(
            "0"
        )

        self.camera_angle_value.setText(
            "--°"
        )

        self.camera_rom_value.setText(
            "--°"
        )

        self.imu_angle_value.setText(
            "--°"
        )

        self.imu_rom_value.setText(
            "--°"
        )

        self.rms_value.setText(
            "--"
        )

        self.mdf_value.setText(
            "-- Hz"
        )

        self.duration_value.setText(
            "00:00"
        )

        self._set_session_status(
            "LIVE",
            GREEN,
        )

        self._set_imu_status(
            "● IMU: Waiting for Raspberry Pi",
            YELLOW,
        )

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        try:

            BaseOptions = (
                mp.tasks.BaseOptions
            )

            PoseLandmarker = (
                mp.tasks.vision.PoseLandmarker
            )

            PoseLandmarkerOptions = (
                mp.tasks.vision.PoseLandmarkerOptions
            )

            VisionRunningMode = (
                mp.tasks.vision.RunningMode
            )

            options = (
                PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=
                        "pose_landmarker_lite.task"
                    ),
                    running_mode=
                    VisionRunningMode.VIDEO,
                )
            )

            self.landmarker = (
                PoseLandmarker.create_from_options(
                    options
                )
            )

        except Exception as exc:

            self.landmarker = None

            self._set_camera_status(
                f"● Camera / MediaPipe: ERROR - {exc}",
                RED,
            )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(
            0
        )

        if self.cap.isOpened():

            self.camera_ok = True

            self._set_camera_status(
                "● Camera: Connected",
                GREEN,
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                1280,
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                720,
            )

        else:

            self.camera_ok = False

            self._set_camera_status(
                "● Camera: Not available",
                RED,
            )

        self.frame_idx = 0

        self.last_frame_time = (
            time.monotonic()
        )

        # ----------------------------------------------------
        # Raspberry Pi TCP
        # ----------------------------------------------------

        self.pi_worker = (
            PiDataWorker()
        )

        self.pi_worker.start()

        # ----------------------------------------------------
        # EMG processor
        # ----------------------------------------------------

        self.emg_worker = (
            EmgWorker()
        )

        self.emg_worker.start()

        # ----------------------------------------------------
        # Timers
        # ----------------------------------------------------

        self.emg_poll_timer.start(
            30
        )

        self.timer.start(
            30
        )

        self.session_timer.start(
            250
        )

        self.pause_btn.setEnabled(
            True
        )

        self.resume_btn.setEnabled(
            False
        )

    def pause_session(self):

        if (
            not self.session_start
            or self.session_paused_at
            is not None
        ):

            return

        self.session_paused_at = (
            time.monotonic()
        )

        self._set_session_status(
            "PAUSED",
            YELLOW,
        )

        if self.emg_worker:

            self.emg_worker.paused = (
                True
            )

        self.timer.stop()

        self.emg_poll_timer.stop()

        self.pause_btn.setEnabled(
            False
        )

        self.resume_btn.setEnabled(
            True
        )

    def resume_session(self):

        if (
            self.session_paused_at
            is None
        ):

            return

        paused_duration = (
            time.monotonic()
            -
            self.session_paused_at
        )

        self.session_start += (
            paused_duration
        )

        self.session_paused_at = None

        if self.emg_worker:

            self.emg_worker.paused = (
                False
            )

        self.timer.start(
            30
        )

        self.emg_poll_timer.start(
            30
        )

        self._set_session_status(
            "LIVE",
            GREEN,
        )

        self.pause_btn.setEnabled(
            True
        )

        self.resume_btn.setEnabled(
            False
        )

    def stop_session(
        self,
        silent=False,
    ):

        self.timer.stop()

        self.emg_poll_timer.stop()

        self.session_timer.stop()

        # ----------------------------------------------------
        # Reset calibration UI state.
        # ----------------------------------------------------

        self.imu_calibration_pending = (
            False
        )

        self.calibrate_imu_btn.setEnabled(
            False
        )

        self.calibrate_imu_btn.setText(
            "Calibrate IMU"
        )

        # ----------------------------------------------------
        # Stop EMG worker.
        # ----------------------------------------------------

        if self.emg_worker:

            self.emg_worker.stop()

            self.emg_worker = None

        # ----------------------------------------------------
        # Stop Pi worker.
        # ----------------------------------------------------

        if self.pi_worker:

            self.pi_worker.stop()

            self.pi_worker = None

        # ----------------------------------------------------
        # Camera.
        # ----------------------------------------------------

        if self.cap:

            self.cap.release()

            self.cap = None

        # ----------------------------------------------------
        # MediaPipe.
        # ----------------------------------------------------

        if self.landmarker:

            try:

                self.landmarker.close()

            except Exception:

                pass

            self.landmarker = None

        if not silent:

            self._set_session_status(
                "STOPPED",
                TEXT_DIM,
            )

            self._show_session_summary()

        self.pause_btn.setEnabled(
            False
        )

        self.resume_btn.setEnabled(
            False
        )

    # ========================================================
    # Session timer
    # ========================================================

    def _update_session_time(self):

        if self.session_start is None:

            return

        now = (
            self.session_paused_at
            if self.session_paused_at
            is not None
            else time.monotonic()
        )

        elapsed = max(
            0,
            now - self.session_start,
        )

        minutes = int(
            elapsed // 60
        )

        seconds = int(
            elapsed % 60
        )

        self.duration_value.setText(
            f"{minutes:02d}:{seconds:02d}"
        )

    # ========================================================
    # Status
    # ========================================================

    def _set_session_status(
        self,
        text,
        color,
    ):

        self.session_status.setText(
            text
        )

        self.session_status.setStyleSheet(
            f"""
            color: {color};
            background: {PANEL};
            border: 1px solid #2a2a35;
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 11px;
            font-weight: 700;
            """
        )

    def _set_camera_status(
        self,
        text,
        color,
    ):

        self.camera_status.setText(
            text
        )

        self.camera_status.setStyleSheet(
            f"""
            color: {color};
            font-size: 11px;
            """
        )

    def _set_imu_status(
        self,
        text,
        color,
    ):

        self.imu_status.setText(
            text
        )

        self.imu_status.setStyleSheet(
            f"""
            color: {color};
            font-size: 11px;
            """
        )

    def _set_emg_status(
        self,
        text,
        color,
    ):

        self.emg_status.setText(
            text
        )

        self.emg_status.setStyleSheet(
            f"""
            color: {color};
            font-size: 11px;
            """
        )

    # ========================================================
    # EMG
    # ========================================================

    def _poll_emg(self):

        # ----------------------------------------------------
        # Read latest Pi data.
        # ----------------------------------------------------

        with pi_data_lock:

            connected = (
                pi_data[
                    "connected"
                ]
            )

            samples = list(
                pi_data.get(
                    "emg_samples",
                    [],
                )
            )

            emg_rate = float(
                pi_data.get(
                    "emg_rate_sps",
                    0.0,
                )
            )

            emg_stale = bool(
                pi_data.get(
                    "emg_stale",
                    True,
                )
            )

            latest_adc = int(
                pi_data.get(
                    "emg_latest_adc",
                    0,
                )
            )

        # ----------------------------------------------------
        # Update IMU.
        # ----------------------------------------------------

        self._update_imu_from_pi()

        # ----------------------------------------------------
        # Forward raw EMG samples.
        # ----------------------------------------------------

        if (
            self.emg_worker
            and samples
            and not emg_stale
        ):

            self.emg_worker.push_samples(
                samples
            )

        # ----------------------------------------------------
        # EMG status.
        # ----------------------------------------------------

        if not connected:

            self._set_emg_status(
                "● EMG: Waiting for Raspberry Pi",
                YELLOW,
            )

            self.emg_ok = False

        elif emg_stale:

            self._set_emg_status(
                "● EMG: Pi reports stale data",
                RED,
            )

            self.emg_ok = False

        else:

            self._set_emg_status(
                "● EMG: Receiving  |  "
                f"{emg_rate:.0f} samples/s  |  "
                f"ADC {latest_adc}",
                GREEN,
            )

        # ----------------------------------------------------
        # Process EMG results.
        # ----------------------------------------------------

        if not self.emg_worker:

            return

        while (
            not self.emg_worker.error_queue.empty()
        ):

            message = (
                self.emg_worker
                .error_queue
                .get()
            )

            self._on_emg_error(
                message
            )

        while (
            not self.emg_worker.result_queue.empty()
        ):

            (
                rms,
                mdf,
                waveform,
            ) = (
                self.emg_worker
                .result_queue
                .get()
            )

            self._on_emg_result(
                rms,
                mdf,
                waveform,
            )

    # ========================================================
    # Update IMU data
    # ========================================================

    def _update_imu_from_pi(self):

        with pi_data_lock:

            connected = bool(
                pi_data.get(
                    "connected",
                    False,
                )
            )

            angle = float(
                pi_data.get(
                    "angle_deg",
                    0.0,
                )
            )

            rom_max = float(
                pi_data.get(
                    "rom_max_deg",
                    0.0,
                )
            )

            rom_min = float(
                pi_data.get(
                    "rom_min_deg",
                    0.0,
                )
            )

            imu_calibrated = bool(
                pi_data.get(
                    "imu_calibrated",
                    False,
                )
            )

            calibration_status = str(
                pi_data.get(
                    "imu_calibration_status",
                    "Not calibrated",
                )
            )

        # ----------------------------------------------------
        # Pi disconnected
        # ----------------------------------------------------

        if not connected:

            self.pi_angle = None

            self.imu_angle_value.setText(
                "--°"
            )

            self.imu_rom_value.setText(
                "--°"
            )

            self.calibrate_imu_btn.setEnabled(
                False
            )

            self.imu_calibration_pending = (
                False
            )

            self._set_imu_status(
                "● IMU: Disconnected",
                RED,
            )

            return

        # ----------------------------------------------------
        # Connected
        # ----------------------------------------------------

        self.pi_angle = angle

        self.pi_rom_max = rom_max

        self.pi_rom_min = rom_min

        self.pi_rom = max(
            0.0,
            rom_max - rom_min,
        )

        self.imu_calibrated = (
            imu_calibrated
        )

        self.imu_calibration_status = (
            calibration_status
        )

        self.imu_angle_value.setText(
            f"{angle:.1f}°"
        )

        self.imu_rom_value.setText(
            f"{self.pi_rom:.1f}°"
        )

        # ----------------------------------------------------
        # Calibration acknowledgement.
        # ----------------------------------------------------

        if (
            self.imu_calibration_pending
            and imu_calibrated
        ):

            self.imu_calibration_pending = (
                False
            )

            self.calibrate_imu_btn.setEnabled(
                True
            )

            self.calibrate_imu_btn.setText(
                "Calibrate IMU"
            )

        # ----------------------------------------------------
        # Status display.
        # ----------------------------------------------------

        if self.imu_calibration_pending:

            self._set_imu_status(
                "● IMU: Calibrating... | "
                "Keep sensors still",
                YELLOW,
            )

        elif imu_calibrated:

            self._set_imu_status(
                "● IMU: Calibrated  |  "
                f"Angle {angle:.1f}°",
                GREEN,
            )

            self.calibrate_imu_btn.setEnabled(
                True
            )

        else:

            self._set_imu_status(
                "● IMU: Connected  |  "
                "Not calibrated",
                YELLOW,
            )

            self.calibrate_imu_btn.setEnabled(
                True
            )

    # ========================================================
    # EMG error
    # ========================================================

    def _on_emg_error(
        self,
        message,
    ):

        print(
            "EMG PROCESSING ERROR:\n",
            message,
        )

        lines = (
            message
            .strip()
            .splitlines()
        )

        first_line = (
            lines[-1]
            if lines
            else "Unknown EMG error"
        )

        self._set_emg_status(
            f"● EMG: ERROR - {first_line}",
            RED,
        )

    # ========================================================
    # EMG result
    # ========================================================

    def _on_emg_result(
        self,
        rms,
        mdf,
        waveform,
    ):

        self.latest_rms = rms
        self.latest_mdf = mdf

        self.emg_ok = True

        self.rms_value.setText(
            f"{rms:.3f}"
        )

        self.mdf_value.setText(
            f"{mdf:.1f} Hz"
        )

        # ----------------------------------------------------
        # Waveform
        # ----------------------------------------------------

        if waveform is not None:

            self.plot_buffer.clear()

            self.plot_buffer.extend(
                waveform
            )

        if len(
            self.plot_buffer
        ) > 1:

            vals = np.asarray(
                self.plot_buffer,
                dtype=np.float64,
            )

            self.emg_curve.setData(
                vals
            )

            peak = float(
                np.max(
                    np.abs(vals)
                )
            )

            if peak < 1.0:

                peak = 1.0

            self.emg_plot.setYRange(
                -peak * 1.2,
                peak * 1.2,
            )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        if self.session_start is None:

            self.session_start = (
                time.monotonic()
            )

        elapsed = (
            time.monotonic()
            -
            self.session_start
        )

        self.history[
            "t"
        ].append(
            elapsed
        )

        self.history[
            "rms"
        ].append(
            rms
        )

        self.history[
            "mdf"
        ].append(
            mdf
        )

        self.history[
            "camera_angle"
        ].append(
            self._last_angle
        )

        self.history[
            "camera_rom"
        ].append(
            self.camera_rom
        )

        self.history[
            "imu_angle"
        ].append(
            self.pi_angle
            if self.pi_angle is not None
            else 0.0
        )

        self.history[
            "imu_rom"
        ].append(
            self.pi_rom
        )

    # ========================================================
    # Camera / MediaPipe
    # ========================================================

    def _update_frame(self):

        if (
            not self.cap
            or not self.cap.isOpened()
        ):

            return

        ret, frame = (
            self.cap.read()
        )

        if not ret:

            self._set_camera_status(
                "● Camera: Frame read failed",
                RED,
            )

            return

        a_idx, b_idx, c_idx = (
            MOVEMENTS[
                self.movement_name
            ][
                self.side
            ]
        )

        if self.landmarker is not None:

            try:

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=
                    mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = int(
                    (
                        time.monotonic()
                        -
                        self.last_frame_time
                    )
                    * 1000
                    +
                    self.frame_idx
                    * 33.333
                )

                self.frame_idx += 1

                result = (
                    self.landmarker
                    .detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )
                )

                h, w, _ = (
                    frame.shape
                )

                if result.pose_landmarks:

                    lm = (
                        result.pose_landmarks[
                            0
                        ]
                    )

                    points = [
                        lm[a_idx],
                        lm[b_idx],
                        lm[c_idx],
                    ]

                    visible = all(
                        getattr(
                            p,
                            "visibility",
                            1.0,
                        ) > 0.35
                        for p in points
                    )

                    if visible:

                        a = [
                            lm[a_idx].x,
                            lm[a_idx].y,
                        ]

                        b = [
                            lm[b_idx].x,
                            lm[b_idx].y,
                        ]

                        c = [
                            lm[c_idx].x,
                            lm[c_idx].y,
                        ]

                        angle = (
                            calculate_angle(
                                a,
                                b,
                                c,
                            )
                        )

                        # ----------------------------------------
                        # CAMERA ANGLE
                        # ----------------------------------------

                        camera_angle = (
                            float(angle)
                        )

                        self._last_angle = (
                            camera_angle
                        )

                        # ----------------------------------------
                        # CAMERA ROM
                        # ----------------------------------------

                        if (
                            self.camera_angle_min
                            is None
                        ):

                            self.camera_angle_min = (
                                camera_angle
                            )

                            self.camera_angle_max = (
                                camera_angle
                            )

                        else:

                            self.camera_angle_min = min(
                                self.camera_angle_min,
                                camera_angle,
                            )

                            self.camera_angle_max = max(
                                self.camera_angle_max,
                                camera_angle,
                            )

                        self.camera_rom = (
                            self.camera_angle_max
                            -
                            self.camera_angle_min
                        )

                        self.camera_angle_value.setText(
                            f"{camera_angle:.1f}°"
                        )

                        self.camera_rom_value.setText(
                            f"{self.camera_rom:.1f}°"
                        )

                        # ----------------------------------------
                        # REP DETECTION
                        # ----------------------------------------

                        self._process_rep_angle(
                            camera_angle
                        )

                        color = (
                            angle_to_color(
                                camera_angle
                            )
                        )

                        pa = (
                            int(a[0] * w),
                            int(a[1] * h),
                        )

                        pb = (
                            int(b[0] * w),
                            int(b[1] * h),
                        )

                        pc = (
                            int(c[0] * w),
                            int(c[1] * h),
                        )

                        cv2.line(
                            frame,
                            pa,
                            pb,
                            color,
                            6,
                            cv2.LINE_AA,
                        )

                        cv2.line(
                            frame,
                            pb,
                            pc,
                            color,
                            6,
                            cv2.LINE_AA,
                        )

                        for pt in [
                            pa,
                            pb,
                            pc,
                        ]:

                            cv2.circle(
                                frame,
                                pt,
                                8,
                                (
                                    255,
                                    255,
                                    255,
                                ),
                                -1,
                                cv2.LINE_AA,
                            )

                            cv2.circle(
                                frame,
                                pt,
                                8,
                                (
                                    0,
                                    0,
                                    0,
                                ),
                                2,
                                cv2.LINE_AA,
                            )

                        # ----------------------------------------
                        # Camera overlay
                        # ----------------------------------------

                        cv2.putText(
                            frame,
                            f"Camera Angle: "
                            f"{camera_angle:.0f} deg",
                            (
                                25,
                                45,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (
                                255,
                                255,
                                255,
                            ),
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.putText(
                            frame,
                            f"Camera ROM: "
                            f"{self.camera_rom:.0f} deg",
                            (
                                25,
                                82,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (
                                255,
                                255,
                                255,
                            ),
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.putText(
                            frame,
                            f"Reps: "
                            f"{self.rep_count}",
                            (
                                25,
                                119,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (
                                255,
                                255,
                                255,
                            ),
                            2,
                            cv2.LINE_AA,
                        )

                self._set_camera_status(
                    "● Camera: Connected  |  "
                    "Pose detected",
                    GREEN,
                )

            except Exception as exc:

                self._set_camera_status(
                    f"● Pose error: {exc}",
                    RED,
                )

        # ----------------------------------------------------
        # Display frame
        # ----------------------------------------------------

        rgb_out = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        h, w, _ = (
            rgb_out.shape
        )

        qimg = QtGui.QImage(
            rgb_out.data,
            w,
            h,
            3 * w,
            QtGui.QImage.Format_RGB888,
        )

        pixmap = (
            QtGui.QPixmap.fromImage(
                qimg
            )
        )

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    # ========================================================
    # Repetition detection
    # ========================================================

    def _process_rep_angle(
        self,
        angle,
    ):

        self.angle_history.append(
            float(angle)
        )

        if (
            len(self.angle_history)
            < REP_SMOOTHING
        ):

            return

        smooth_angle = float(
            np.mean(
                self.angle_history
            )
        )

        now = time.monotonic()

        if (
            self.last_rep_time > 0
            and
            now - self.last_rep_time
            < REP_COOLDOWN
        ):

            return

        if (
            self.rep_high is None
            or smooth_angle
            > self.rep_high
        ):

            self.rep_high = (
                smooth_angle
            )

        if (
            self.rep_low is None
            or smooth_angle
            < self.rep_low
        ):

            self.rep_low = (
                smooth_angle
            )

        if (
            smooth_angle
            >= REP_MAX_ANGLE
        ):

            self.rep_phase = (
                "extended"
            )

        elif (
            self.rep_phase
            == "extended"
            and
            smooth_angle
            <= REP_MIN_ANGLE
            and
            self.rep_high is not None
            and
            (
                self.rep_high
                -
                smooth_angle
            )
            >= REP_MIN_TRAVEL
        ):

            self.rep_count += 1

            self.rep_value.setText(
                str(
                    self.rep_count
                )
            )

            self.last_rep_time = (
                now
            )

            self.rep_phase = (
                "flexed"
            )

            self.rep_high = (
                smooth_angle
            )

            self.rep_low = (
                smooth_angle
            )

    # ========================================================
    # Graphs
    # ========================================================

    def _open_metric_dialog(
        self,
        metric_key,
        metric_label,
    ):

        if not self.history["t"]:

            QtWidgets.QMessageBox.information(
                self,
                metric_label,
                "No session data is available yet.",
            )

            return

        was_running = (
            self.session_paused_at
            is None
        )

        if was_running:

            self.pause_session()

        dlg = QtWidgets.QDialog(
            self
        )

        dlg.setWindowTitle(
            metric_label
        )

        dlg.resize(
            700,
            480,
        )

        dlg.setStyleSheet(
            f"background-color: {BG};"
        )

        layout = (
            QtWidgets.QVBoxLayout(
                dlg
            )
        )

        figure = Figure(
            figsize=(6, 4),
            facecolor=PANEL,
        )

        canvas = FigureCanvas(
            figure
        )

        ax = figure.add_subplot(
            111
        )

        x = self.history["t"]

        y = self.history[
            metric_key
        ]

        ax.plot(
            x,
            y,
            color=ACCENT,
            linewidth=1.5,
        )

        ax.set_title(
            metric_label,
            color="white",
            fontsize=13,
        )

        ax.set_xlabel(
            "Time (s)",
            color=TEXT_DIM,
            fontsize=9,
        )

        ax.tick_params(
            colors=TEXT_DIM,
            labelsize=8,
        )

        ax.set_facecolor(
            PANEL
        )

        for spine in ax.spines.values():

            spine.set_color(
                "#2a2a35"
            )

        figure.tight_layout()

        layout.addWidget(
            canvas
        )

        close_btn = (
            QtWidgets.QPushButton(
                "Close"
            )
        )

        close_btn.setStyleSheet(
            self._secondary_button_style()
        )

        close_btn.clicked.connect(
            dlg.accept
        )

        layout.addWidget(
            close_btn,
            alignment=QtCore.Qt.AlignRight,
        )

        dlg.exec_()

        if was_running:

            self.resume_session()

    # ========================================================
    # Session summary
    # ========================================================

    def _show_session_summary(
        self
    ):

        duration = 0.0

        if self.session_start is not None:

            duration = max(
                0,
                time.monotonic()
                -
                self.session_start,
            )

        rms = np.asarray(
            self.history["rms"],
            dtype=float,
        )

        mdf = np.asarray(
            self.history["mdf"],
            dtype=float,
        )

        camera_angles = np.asarray(
            self.history[
                "camera_angle"
            ],
            dtype=float,
        )

        avg_rms = (
            float(
                np.mean(rms)
            )
            if rms.size
            else 0.0
        )

        max_rms = (
            float(
                np.max(rms)
            )
            if rms.size
            else 0.0
        )

        avg_mdf = (
            float(
                np.mean(mdf)
            )
            if mdf.size
            else 0.0
        )

        # ----------------------------------------------------
        # Camera ROM
        #
        # Camera ROM remains completely independent.
        # ----------------------------------------------------

        valid_camera = (
            camera_angles[
                camera_angles > 0
            ]
        )

        camera_min = (
            float(
                np.min(valid_camera)
            )
            if valid_camera.size
            else 0.0
        )

        camera_max = (
            float(
                np.max(valid_camera)
            )
            if valid_camera.size
            else 0.0
        )

        camera_rom = (
            camera_max
            -
            camera_min
        )

        # ----------------------------------------------------
        # IMU ROM
        #
        # IMPORTANT:
        #
        # Do NOT reconstruct IMU ROM from the Windows history.
        #
        # The Raspberry Pi owns the IMU ROM calculation.
        # ----------------------------------------------------

        imu_min = float(
            self.pi_rom_min
        )

        imu_max = float(
            self.pi_rom_max
        )

        imu_rom = float(
            self.pi_rom
        )

        minutes = int(
            duration // 60
        )

        seconds = int(
            duration % 60
        )

        dlg = QtWidgets.QDialog(
            self
        )

        dlg.setWindowTitle(
            "Session Summary"
        )

        dlg.resize(
            560,
            600,
        )

        dlg.setStyleSheet(
            f"background-color: {BG};"
        )

        layout = (
            QtWidgets.QVBoxLayout(
                dlg
            )
        )

        title = QtWidgets.QLabel(
            "SESSION SUMMARY"
        )

        title.setStyleSheet(
            f"""
            color: {ACCENT};
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 2px;
            """
        )

        layout.addWidget(
            title
        )

        info = QtWidgets.QLabel(
            f"""
            <b>Movement:</b>
            {self.movement_name}<br>

            <b>Side:</b>
            {'Left' if self.side == 'L' else 'Right'}<br>

            <b>Duration:</b>
            {minutes:02d}:{seconds:02d}<br>

            <b>Repetitions:</b>
            {self.rep_count}<br><br>

            <b>Average EMG RMS:</b>
            {avg_rms:.4f}<br>

            <b>Maximum EMG RMS:</b>
            {max_rms:.4f}<br>

            <b>Average MDF:</b>
            {avg_mdf:.2f} Hz<br><br>

            <b>Camera Minimum Angle:</b>
            {camera_min:.1f}°<br>

            <b>Camera Maximum Angle:</b>
            {camera_max:.1f}°<br>

            <b>Camera ROM:</b>
            {camera_rom:.1f}°<br><br>

            <b>IMU Minimum Angle:</b>
            {imu_min:.1f}°<br>

            <b>IMU Maximum Angle:</b>
            {imu_max:.1f}°<br>

            <b>IMU ROM:</b>
            {imu_rom:.1f}°<br><br>

            <b>IMU Calibration:</b>
            {'Calibrated' if self.imu_calibrated else 'Not calibrated'}
            """
        )

        info.setStyleSheet(
            """
            color: white;
            font-size: 14px;
            line-height: 1.6;
            """
        )

        layout.addWidget(
            info
        )

        buttons = (
            QtWidgets.QHBoxLayout()
        )

        save_btn = (
            QtWidgets.QPushButton(
                "Export CSV"
            )
        )

        close_btn = (
            QtWidgets.QPushButton(
                "Close"
            )
        )

        save_btn.setStyleSheet(
            self._secondary_button_style()
        )

        close_btn.setStyleSheet(
            self._secondary_button_style()
        )

        save_btn.clicked.connect(
            lambda:
            self._export_csv(
                dlg
            )
        )

        close_btn.clicked.connect(
            dlg.accept
        )

        buttons.addWidget(
            save_btn
        )

        buttons.addStretch()

        buttons.addWidget(
            close_btn
        )

        layout.addLayout(
            buttons
        )

        dlg.exec_()

    # ========================================================
    # CSV
    # ========================================================

    def _export_csv(
        self,
        parent=None,
    ):

        if not self.history["t"]:

            QtWidgets.QMessageBox.information(
                self,
                "Export CSV",
                "There is no session data to export.",
            )

            return

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        filename, _ = (
            QtWidgets.QFileDialog
            .getSaveFileName(
                self,
                "Save Session Data",
                f"myotrack_session_{timestamp}.csv",
                "CSV Files (*.csv)",
            )
        )

        if not filename:

            return

        rows = zip(
            self.history["t"],
            self.history["rms"],
            self.history["mdf"],
            self.history[
                "camera_angle"
            ],
            self.history[
                "camera_rom"
            ],
            self.history[
                "imu_angle"
            ],
            self.history[
                "imu_rom"
            ],
        )

        try:

            with open(
                filename,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.writer(
                    f
                )

                writer.writerow(
                    [
                        "timestamp",
                        "movement",
                        "side",
                        "time_s",

                        "emg_rms",
                        "mdf_hz",

                        "camera_angle_deg",
                        "camera_rom_deg",

                        "imu_angle_deg",
                        "imu_rom_deg",

                        "rep_count",
                    ]
                )

                timestamp_iso = (
                    datetime.now()
                    .isoformat()
                )

                for (
                    t,
                    rms,
                    mdf,
                    camera_angle,
                    camera_rom,
                    imu_angle,
                    imu_rom,
                ) in rows:

                    writer.writerow(
                        [
                            timestamp_iso,

                            self.movement_name,

                            (
                                "Left"
                                if self.side == "L"
                                else "Right"
                            ),

                            f"{t:.4f}",

                            f"{rms:.8f}",

                            f"{mdf:.4f}",

                            f"{camera_angle:.4f}",

                            f"{camera_rom:.4f}",

                            f"{imu_angle:.4f}",

                            f"{imu_rom:.4f}",

                            self.rep_count,
                        ]
                    )

            QtWidgets.QMessageBox.information(
                parent or self,
                "Export Complete",
                f"Session data saved to:\n{filename}",
            )

        except Exception as exc:

            QtWidgets.QMessageBox.critical(
                parent or self,
                "Export Failed",
                f"Could not save the CSV:\n{exc}",
            )

    # ========================================================
    # Navigation
    # ========================================================

    def _on_back(self):

        self.stop_session(
            silent=True
        )

        self.back_requested.emit()


# ============================================================
# Main window
# ============================================================

class MainWindow(
    QtWidgets.QMainWindow
):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "MyoTrack"
        )

        self.resize(
            1400,
            900,
        )

        self.setMinimumSize(
            1100,
            750,
        )

        # ----------------------------------------------------
        # Stacked container
        # ----------------------------------------------------

        self.stack = (
            QtWidgets.QStackedWidget()
        )

        self.setCentralWidget(
            self.stack
        )

        # ----------------------------------------------------
        # Screens
        # ----------------------------------------------------

        self.splash = (
            SplashScreen()
        )

        self.options = (
            OptionsScreen()
        )

        self.dashboard = None

        self.stack.addWidget(
            self.splash
        )

        self.stack.addWidget(
            self.options
        )

        self.stack.setCurrentWidget(
            self.splash
        )

        # ----------------------------------------------------
        # Splash -> Options
        # ----------------------------------------------------

        self.splash.finished.connect(
            self.show_options
        )

        # ----------------------------------------------------
        # Options -> Dashboard
        # ----------------------------------------------------

        self.options.start_requested.connect(
            self.on_start_session
        )

    def show_options(self):

        self.stack.setCurrentWidget(
            self.options
        )

        self.options.show()
        self.options.raise_()
        self.options.activateWindow()

    def on_start_session(
        self,
        movement,
        side,
    ):

        if self.dashboard is None:

            self.dashboard = (
                DashboardScreen()
            )

            self.stack.addWidget(
                self.dashboard
            )

            self.dashboard.back_requested.connect(
                self.show_options
            )

        self.dashboard.start_session(
            movement,
            side,
        )

        self.stack.setCurrentWidget(
            self.dashboard
        )

    def closeEvent(
        self,
        event,
    ):

        if self.dashboard is not None:

            self.dashboard.stop_session(
                silent=True
            )

        event.accept()


# ============================================================
# Main
# ============================================================

def main():

    app = (
        QtWidgets.QApplication(
            sys.argv
        )
    )

    app.setStyle(
        "Fusion"
    )

    palette = (
        QtGui.QPalette()
    )

    palette.setColor(
        QtGui.QPalette.Window,
        QtGui.QColor(BG),
    )

    palette.setColor(
        QtGui.QPalette.WindowText,
        QtGui.QColor("white"),
    )

    palette.setColor(
        QtGui.QPalette.Base,
        QtGui.QColor(PANEL),
    )

    palette.setColor(
        QtGui.QPalette.AlternateBase,
        QtGui.QColor(BG),
    )

    palette.setColor(
        QtGui.QPalette.Text,
        QtGui.QColor("white"),
    )

    palette.setColor(
        QtGui.QPalette.Button,
        QtGui.QColor(PANEL),
    )

    palette.setColor(
        QtGui.QPalette.ButtonText,
        QtGui.QColor("white"),
    )

    app.setPalette(
        palette
    )

    win = MainWindow()

    win.show()

    sys.exit(
        app.exec_()
    )


if __name__ == "__main__":

    main()