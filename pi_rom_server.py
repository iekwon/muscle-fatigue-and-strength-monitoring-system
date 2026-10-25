"""
============================================================================
Raspberry Pi ROM + EMG Server
============================================================================

Receives two kinds of UDP packets on the same port from the ESP32 nodes:

  IMU packets (ASCII text, from BOTH ESP32 #1 and #2):
      "<id>,<seq>,<w>,<x>,<y>,<z>\n"

      id = 1 (proximal segment)
      id = 2 (distal segment)

  EMG packets (binary, from ESP32 #1 only -- relays Nano ADC samples):

      Header (12 bytes, little-endian):
          struct "<BBIIH"

          marker       uint8   = ord('E')
          client_id    uint8
          batch_seq    uint32
          esp32_micros uint32
          sample_count uint16

      Then N samples, 10 bytes each:

          struct "<IIH"

          sample_number uint32
          timestamp_us  uint32
          adc_value     uint16

Packet-type dispatch is done on the raw first byte BEFORE any UTF-8
decode is attempted.

    data[0] == ord('E')  -> EMG binary packet
    otherwise             -> IMU ASCII packet

============================================================================
ROM / joint-angle math
============================================================================

    q_rel = q1^-1 (x) q2

    angle = 2 * atan2(|vec(q_rel)|, |w(q_rel)|)

Calibration ('c') captures q_rel at a reference pose.

ROM is running max - min since last reset ('r').

============================================================================
EMG handling
============================================================================

Raw EMG samples are stored unmodified in a fixed-size ring buffer.

No filtering, rectification, RMS, FFT, or MDF is performed here.

The raw EMG samples can now also be retrieved by:

    get_latest_emg_samples(count)

This is used by pi_dashboard_server.py to transmit the newest EMG
samples to the Windows MyoTrack dashboard.

============================================================================
"""

import curses
import math
import socket
import struct
import threading
import time

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ============================================================================
# USER CONFIG
# ============================================================================

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

STALE_AFTER_S = 0.5

# 20,000 samples at approximately 2,000 Hz = approximately 10 seconds.
EMG_RING_BUFFER_CAPACITY = 20000


# ============================================================================
# TYPES
# ============================================================================

Quat = Tuple[float, float, float, float]


# ============================================================================
# QUATERNION MATH
# ============================================================================

def quat_mul(q1: Quat, q2: Quat) -> Quat:
    """Hamilton product q1 x q2."""

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_conj(q: Quat) -> Quat:
    """Quaternion conjugate."""

    w, x, y, z = q

    return (
        w,
        -x,
        -y,
        -z,
    )


def quat_normalize(q: Quat) -> Quat:
    """Normalize quaternion."""

    w, x, y, z = q

    n = math.sqrt(
        w * w +
        x * x +
        y * y +
        z * z
    )

    if n < 1e-12:
        return (
            1.0,
            0.0,
            0.0,
            0.0,
        )

    return (
        w / n,
        x / n,
        y / n,
        z / n,
    )


def quat_angle_deg(q: Quat) -> float:
    """
    Rotation angle represented by unit quaternion q.

    Result is folded into [0, 180] degrees.
    """

    w, x, y, z = quat_normalize(q)

    vec_norm = math.sqrt(
        x * x +
        y * y +
        z * z
    )

    return 2.0 * math.degrees(
        math.atan2(
            vec_norm,
            abs(w)
        )
    )


def relative_quat(
    q_proximal: Quat,
    q_distal: Quat
) -> Quat:
    """
    Relative orientation:

        q_rel = q_proximal^-1 x q_distal

    For a unit quaternion, inverse = conjugate.
    """

    return quat_mul(
        quat_conj(q_proximal),
        q_distal
    )


# ============================================================================
# IMU / ROM SHARED STATE
# ============================================================================

@dataclass
class SensorState:

    q: Quat = (
        1.0,
        0.0,
        0.0,
        0.0
    )

    last_seen: float = 0.0

    seq: int = -1

    rate_hz: float = 0.0

    _count_since: int = field(
        default=0,
        repr=False
    )

    _rate_window_start: float = field(
        default=0.0,
        repr=False
    )


class SharedState:

    def __init__(self):

        self.lock = threading.Lock()

        self.sensors = {
            1: SensorState(),
            2: SensorState()
        }

        # Relative quaternion captured during calibration.
        self.calib_offset: Optional[Quat] = None

        self.rom_max: Optional[float] = None

        self.rom_min: Optional[float] = None

        self.current_angle: Optional[float] = None

        self.running = True


state = SharedState()


# ============================================================================
# EMG CONSTANTS
# ============================================================================

EMG_MARKER_BYTE = ord("E")

# marker, id, batch_seq, esp_micros, sample_count
EMG_HEADER_STRUCT = struct.Struct(
    "<BBIIH"
)

EMG_HEADER_SIZE = EMG_HEADER_STRUCT.size


# sample_number, timestamp_us, adc_value
EMG_SAMPLE_STRUCT = struct.Struct(
    "<IIH"
)

EMG_SAMPLE_SIZE = EMG_SAMPLE_STRUCT.size


# Prevent enormous counter jumps from being interpreted as a real
# sample-loss count.
EMG_SAMPLE_GAP_SANITY_LIMIT = 1_000_000


# ============================================================================
# EMG SAMPLE
# ============================================================================

@dataclass
class EmgSample:

    sample_number: int

    timestamp_us: int

    adc_value: int


# ============================================================================
# EMG SHARED STATE
# ============================================================================

class EmgState:

    def __init__(self):

        self.lock = threading.Lock()

        # Fixed-size raw EMG ring buffer.
        #
        # New samples are appended.
        # Oldest samples are automatically evicted when full.
        self.buffer = deque(
            maxlen=EMG_RING_BUFFER_CAPACITY
        )

        # Statistics
        self.packet_count: int = 0

        self.sample_count: int = 0

        self.malformed_packet_count: int = 0

        self.dropped_sample_count: int = 0

        self.last_client_id: Optional[int] = None

        self.last_seen: float = 0.0

        self.rate_sps: float = 0.0

        self.latest_adc_value: Optional[int] = None

        # ESP32 packet sequence tracking
        self.last_batch_seq: Optional[int] = None

        self.packet_gap_count: int = 0

        # Nano sample sequence tracking
        self.last_sample_number: Optional[int] = None

        self.sample_gap_count: int = 0

        # Rate calculation
        self._count_since: int = 0

        self._rate_window_start: float = 0.0


emg_state = EmgState()


# ============================================================================
# IMU PACKET HANDLER
# ============================================================================

def handle_imu_packet(data: bytes):
    """
    Decode an existing IMU ASCII packet.

    Format:

        <id>,<seq>,<w>,<x>,<y>,<z>
    """

    try:

        text = data.decode(
            "utf-8",
            errors="ignore"
        ).strip()

        parts = text.split(",")

        if len(parts) != 6:
            return

        cid = int(parts[0])

        seq = int(parts[1])

        w, x, y, z = (
            float(p)
            for p in parts[2:6]
        )

    except (
        ValueError,
        IndexError
    ):
        return

    if cid not in (1, 2):
        return

    now = time.time()

    with state.lock:

        s = state.sensors[cid]

        s.q = (
            w,
            x,
            y,
            z
        )

        s.last_seen = now

        s.seq = seq

        s._count_since += 1

        if s._rate_window_start == 0.0:
            s._rate_window_start = now

        elapsed = (
            now -
            s._rate_window_start
        )

        if elapsed >= 1.0:

            s.rate_hz = (
                s._count_since /
                elapsed
            )

            s._count_since = 0

            s._rate_window_start = now


# ============================================================================
# EMG PACKET HANDLER
# ============================================================================

def handle_emg_packet(data: bytes):
    """
    Decode one binary EMG packet from ESP32 #1.

    Raw samples are stored unchanged.

    No EMG processing is performed here.
    """

    now = time.time()

    # ------------------------------------------------------------------------
    # Check minimum header length
    # ------------------------------------------------------------------------

    if len(data) < EMG_HEADER_SIZE:

        with emg_state.lock:
            emg_state.malformed_packet_count += 1

        return

    # ------------------------------------------------------------------------
    # Decode header
    # ------------------------------------------------------------------------

    try:

        (
            marker,
            client_id,
            batch_seq,
            esp_micros,
            sample_count
        ) = EMG_HEADER_STRUCT.unpack_from(
            data,
            0
        )

    except struct.error:

        with emg_state.lock:
            emg_state.malformed_packet_count += 1

        return

    # ------------------------------------------------------------------------
    # Validate packet
    # ------------------------------------------------------------------------

    expected_len = (
        EMG_HEADER_SIZE +
        sample_count * EMG_SAMPLE_SIZE
    )

    if (
        marker != EMG_MARKER_BYTE
        or client_id not in (1, 2)
        or len(data) != expected_len
    ):

        with emg_state.lock:
            emg_state.malformed_packet_count += 1

        return

    # ------------------------------------------------------------------------
    # Decode all samples before acquiring lock
    # ------------------------------------------------------------------------

    try:

        samples = [
            EMG_SAMPLE_STRUCT.unpack_from(
                data,
                EMG_HEADER_SIZE +
                i * EMG_SAMPLE_SIZE
            )

            for i in range(sample_count)
        ]

    except struct.error:

        with emg_state.lock:
            emg_state.malformed_packet_count += 1

        return

    # ------------------------------------------------------------------------
    # Update shared EMG state
    # ------------------------------------------------------------------------

    with emg_state.lock:

        emg_state.packet_count += 1

        emg_state.last_seen = now

        emg_state.last_client_id = client_id

        # ================================================================
        # Packet-level gap tracking
        # ================================================================

        if emg_state.last_batch_seq is not None:

            expected_batch = (
                emg_state.last_batch_seq +
                1
            ) & 0xFFFFFFFF

            if batch_seq != expected_batch:

                emg_state.packet_gap_count += 1

        emg_state.last_batch_seq = batch_seq

        # ================================================================
        # Store samples
        # ================================================================

        buf = emg_state.buffer

        cap = buf.maxlen

        for (
            sample_number,
            timestamp_us,
            adc_value
        ) in samples:

            # ------------------------------------------------------------
            # Nano sample-number gap tracking
            # ------------------------------------------------------------

            if emg_state.last_sample_number is not None:

                expected_sample = (
                    emg_state.last_sample_number +
                    1
                ) & 0xFFFFFFFF

                if sample_number != expected_sample:

                    missing = (
                        sample_number -
                        expected_sample
                    ) & 0xFFFFFFFF

                    if missing < EMG_SAMPLE_GAP_SANITY_LIMIT:

                        emg_state.sample_gap_count += missing

                    else:

                        emg_state.sample_gap_count += 1

            emg_state.last_sample_number = sample_number

            # ------------------------------------------------------------
            # Latest ADC value
            # ------------------------------------------------------------

            emg_state.latest_adc_value = adc_value

            # ------------------------------------------------------------
            # Ring-buffer overflow tracking
            # ------------------------------------------------------------

            if (
                cap is not None
                and len(buf) >= cap
            ):

                emg_state.dropped_sample_count += 1

            # ------------------------------------------------------------
            # Store raw sample
            # ------------------------------------------------------------

            buf.append(
                EmgSample(
                    sample_number,
                    timestamp_us,
                    adc_value
                )
            )

        # ================================================================
        # Sample statistics
        # ================================================================

        n = len(samples)

        emg_state.sample_count += n

        emg_state._count_since += n

        if emg_state._rate_window_start == 0.0:

            emg_state._rate_window_start = now

        elapsed = (
            now -
            emg_state._rate_window_start
        )

        if elapsed >= 1.0:

            emg_state.rate_sps = (
                emg_state._count_since /
                elapsed
            )

            emg_state._count_since = 0

            emg_state._rate_window_start = now


# ============================================================================
# UDP RECEIVER
# ============================================================================

def udp_receiver(sock: socket.socket):

    while state.running:

        try:

            data, _addr = sock.recvfrom(
                2048
            )

        except OSError:

            break

        if not data:
            continue

        # ====================================================================
        # Packet-type dispatch
        #
        # EMG begins with ASCII 'E'.
        #
        # IMU begins with ASCII '1' or '2'.
        #
        # We inspect the raw byte BEFORE attempting UTF-8 decoding.
        # ====================================================================

        if data[0] == EMG_MARKER_BYTE:

            handle_emg_packet(data)

        else:

            handle_imu_packet(data)


# ============================================================================
# IMU ANGLE / ROM
# ============================================================================

def compute_angle_and_update_rom():
    """
    Returns:

        angle_deg_or_None
        sensor1_stale
        sensor2_stale

    Also updates:

        state.current_angle
        state.rom_max
        state.rom_min
    """

    now = time.time()

    with state.lock:

        s1 = state.sensors[1]

        s2 = state.sensors[2]

        stale1 = (
            now - s1.last_seen
        ) > STALE_AFTER_S or s1.seq < 0

        stale2 = (
            now - s2.last_seen
        ) > STALE_AFTER_S or s2.seq < 0

        if stale1 or stale2:

            state.current_angle = None

            return (
                None,
                stale1,
                stale2
            )

        # ------------------------------------------------------------
        # Relative quaternion
        # ------------------------------------------------------------

        q_rel = relative_quat(
            s1.q,
            s2.q
        )

        # ------------------------------------------------------------
        # Apply calibration
        # ------------------------------------------------------------

        if state.calib_offset is not None:

            q_delta = quat_mul(
                quat_conj(
                    state.calib_offset
                ),
                q_rel
            )

            angle = quat_angle_deg(
                q_delta
            )

        else:

            angle = quat_angle_deg(
                q_rel
            )

        # ------------------------------------------------------------
        # Update current angle
        # ------------------------------------------------------------

        state.current_angle = angle

        # ------------------------------------------------------------
        # Update ROM
        # ------------------------------------------------------------

        if (
            state.rom_max is None
            or angle > state.rom_max
        ):

            state.rom_max = angle

        if (
            state.rom_min is None
            or angle < state.rom_min
        ):

            state.rom_min = angle

        return (
            angle,
            False,
            False
        )


# ============================================================================
# IMU CALIBRATION
# ============================================================================

def calibrate_zero():
    """
    Capture the current relative quaternion as the zero-reference pose.
    """

    now = time.time()

    with state.lock:

        s1 = state.sensors[1]

        s2 = state.sensors[2]

        if (
            (now - s1.last_seen) > STALE_AFTER_S
            or
            (now - s2.last_seen) > STALE_AFTER_S
        ):

            return False

        state.calib_offset = relative_quat(
            s1.q,
            s2.q
        )

        state.rom_max = None

        state.rom_min = None

        state.current_angle = 0.0

        return True


# ============================================================================
# RESET ROM
# ============================================================================

def reset_rom():

    with state.lock:

        state.rom_max = None

        state.rom_min = None


# ============================================================================
# EMG SNAPSHOT
# ============================================================================

@dataclass
class EmgSnapshot:

    stale: bool

    packet_count: int

    sample_count: int

    malformed_packet_count: int

    dropped_sample_count: int

    packet_gap_count: int

    sample_gap_count: int

    latest_adc_value: Optional[int]

    buffer_len: int

    buffer_capacity: int

    rate_sps: float


def compute_emg_snapshot() -> EmgSnapshot:
    """
    Return current EMG statistics.

    This function does not process the waveform.
    """

    now = time.time()

    with emg_state.lock:

        stale = (
            (
                now -
                emg_state.last_seen
            ) > STALE_AFTER_S
            or
            emg_state.last_seen == 0.0
        )

        return EmgSnapshot(

            stale=stale,

            packet_count=(
                emg_state.packet_count
            ),

            sample_count=(
                emg_state.sample_count
            ),

            malformed_packet_count=(
                emg_state.malformed_packet_count
            ),

            dropped_sample_count=(
                emg_state.dropped_sample_count
            ),

            packet_gap_count=(
                emg_state.packet_gap_count
            ),

            sample_gap_count=(
                emg_state.sample_gap_count
            ),

            latest_adc_value=(
                emg_state.latest_adc_value
            ),

            buffer_len=(
                len(emg_state.buffer)
            ),

            buffer_capacity=(
                emg_state.buffer.maxlen or 0
            ),

            rate_sps=(
                emg_state.rate_sps
            ),
        )


# ============================================================================
# NEW:
# GET RAW EMG SAMPLES
# ============================================================================

def get_latest_emg_samples(count=100):
    """
    Return the newest `count` raw EMG samples.

    Each returned sample is:

        [
            sample_number,
            timestamp_us,
            adc_value
        ]

    Example:

        [
            [10001, 123456789, 460],
            [10002, 123457289, 463],
            [10003, 123457789, 468]
        ]

    A copy is returned.

    This is important because the TCP dashboard server should NOT hold
    the EMG lock while sending data over the network.
    """

    with emg_state.lock:

        samples = list(
            emg_state.buffer
        )[-count:]

    return [
        [
            sample.sample_number,
            sample.timestamp_us,
            sample.adc_value
        ]

        for sample in samples
    ]


# ============================================================================
# CURSES DASHBOARD
# ============================================================================

def draw_dashboard(stdscr):

    curses.curs_set(0)

    stdscr.nodelay(True)

    stdscr.timeout(50)

    curses.start_color()

    curses.use_default_colors()

    curses.init_pair(
        1,
        curses.COLOR_GREEN,
        -1
    )

    curses.init_pair(
        2,
        curses.COLOR_RED,
        -1
    )

    curses.init_pair(
        3,
        curses.COLOR_YELLOW,
        -1
    )

    curses.init_pair(
        4,
        curses.COLOR_CYAN,
        -1
    )

    last_status_msg = ""

    last_status_time = 0.0

    while state.running:

        # ================================================================
        # IMU
        # ================================================================

        angle, stale1, stale2 = (
            compute_angle_and_update_rom()
        )

        # ================================================================
        # EMG
        # ================================================================

        emg = compute_emg_snapshot()

        # ================================================================
        # Screen
        # ================================================================

        stdscr.erase()

        h, w = stdscr.getmaxyx()

        def put(
            row,
            col,
            text,
            attr=0
        ):

            if (
                0 <= row < h
                and 0 <= col < w
            ):

                stdscr.addnstr(
                    row,
                    col,
                    text,
                    max(
                        0,
                        w - col - 1
                    ),
                    attr
                )

        put(
            0,
            2,
            "LIVE JOINT ROM + EMG MONITOR",
            curses.A_BOLD |
            curses.color_pair(4)
        )

        put(
            1,
            2,
            "-" * min(60, w - 4)
        )

        # ================================================================
        # IMU SENSOR STATUS
        # ================================================================

        with state.lock:

            s1 = state.sensors[1]

            s2 = state.sensors[2]

            r1 = s1.rate_hz

            r2 = s2.rate_hz

        def link_line(
            cid,
            label,
            stale,
            rate
        ):

            color = (
                curses.color_pair(2)
                if stale
                else curses.color_pair(1)
            )

            status = (
                "NO SIGNAL"
                if stale
                else f"OK  ({rate:4.1f} Hz)"
            )

            return (
                f"  ID {cid} "
                f"[{label:>9}]: "
                f"{status}",
                color
            )

        line, color = link_line(
            1,
            "PROXIMAL",
            stale1,
            r1
        )

        put(
            3,
            2,
            line,
            color
        )

        line, color = link_line(
            2,
            "DISTAL",
            stale2,
            r2
        )

        put(
            4,
            2,
            line,
            color
        )

        put(
            6,
            2,
            "-" * min(60, w - 4)
        )

        # ================================================================
        # CALIBRATION
        # ================================================================

        with state.lock:

            calibrated = (
                state.calib_offset is not None
            )

            rom_max = state.rom_max

            rom_min = state.rom_min

        calib_txt = (
            "CALIBRATED"
            if calibrated
            else
            "NOT CALIBRATED (press 'c')"
        )

        calib_color = (
            curses.color_pair(1)
            if calibrated
            else curses.color_pair(3)
        )

        put(
            8,
            2,
            f"  Calibration: {calib_txt}",
            calib_color
        )

        # ================================================================
        # CURRENT ANGLE
        # ================================================================

        if angle is not None:

            put(
                11,
                2,
                f"  Current Joint Angle:   "
                f"{angle:6.1f} deg",
                curses.A_BOLD |
                curses.color_pair(4)
            )

        else:

            put(
                11,
                2,
                "  Current Joint Angle:      "
                "--.- deg  (waiting for sensors)",
                curses.color_pair(2)
            )

        # ================================================================
        # ROM
        # ================================================================

        if (
            rom_max is not None
            and rom_min is not None
        ):

            rom = (
                rom_max -
                rom_min
            )

            put(
                13,
                2,
                f"  Active Set ROM:         "
                f"{rom:6.1f} deg",
                curses.A_BOLD
            )

            put(
                14,
                2,
                f"    Peak (max) angle:     "
                f"{rom_max:6.1f} deg"
            )

            put(
                15,
                2,
                f"    Trough (min) angle:   "
                f"{rom_min:6.1f} deg"
            )

        else:

            put(
                13,
                2,
                "  Active Set ROM:            "
                "--.- deg"
            )

        # ================================================================
        # EMG
        # ================================================================

        put(
            17,
            2,
            "-" * min(60, w - 4)
        )

        put(
            18,
            2,
            "EMG (ESP32 #1 <- Nano ADC)",
            curses.A_BOLD |
            curses.color_pair(4)
        )

        emg_color = (
            curses.color_pair(2)
            if emg.stale
            else curses.color_pair(1)
        )

        emg_status_word = (
            "NO DATA"
            if emg.stale
            else
            "OK"
        )

        adc_txt = (
            "----"
            if emg.latest_adc_value is None
            else
            str(emg.latest_adc_value)
        )

        put(
            19,
            2,
            f"  Status: {emg_status_word}",
            emg_color
        )

        put(
            20,
            2,
            f"  Rate: {emg.rate_sps:8.1f} samples/s"
        )

        put(
            21,
            2,
            f"  Latest ADC: {adc_txt}"
        )

        put(
            22,
            2,
            f"  Packets: {emg.packet_count}"
        )

        put(
            23,
            2,
            f"  Sample gaps: {emg.sample_gap_count}"
        )

        put(
            24,
            2,
            f"  Packet gaps: {emg.packet_gap_count}"
        )

        put(
            25,
            2,
            f"  (buffer "
            f"{emg.buffer_len}/"
            f"{emg.buffer_capacity}  "
            f"malformed "
            f"{emg.malformed_packet_count}  "
            f"overflow-dropped "
            f"{emg.dropped_sample_count})",
            curses.A_DIM
        )

        # ================================================================
        # FOOTER
        # ================================================================

        put(
            h - 3,
            2,
            "-" * min(60, w - 4)
        )

        put(
            h - 2,
            2,
            "  [c] calibrate/zero   "
            "[r] reset ROM   "
            "[q] quit",
            curses.A_DIM
        )

        if (
            last_status_msg
            and
            (
                time.time() -
                last_status_time
            ) < 2.0
        ):

            put(
                9,
                2,
                f"  >> {last_status_msg}",
                curses.color_pair(3)
            )

        stdscr.refresh()

        # ================================================================
        # KEYBOARD
        # ================================================================

        try:

            key = stdscr.getch()

        except curses.error:

            key = -1

        if key in (
            ord("q"),
            ord("Q")
        ):

            state.running = False

        elif key in (
            ord("c"),
            ord("C")
        ):

            ok = calibrate_zero()

            last_status_msg = (
                "Zero point set."
                if ok
                else
                "Can't calibrate: sensor(s) "
                "not live yet."
            )

            last_status_time = time.time()

        elif key in (
            ord("r"),
            ord("R")
        ):

            reset_rom()

            last_status_msg = "ROM reset."

            last_status_time = time.time()


# ============================================================================
# MAIN
# ============================================================================

def main():

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    sock.bind(
        (
            UDP_IP,
            UDP_PORT
        )
    )

    sock.settimeout(1.0)

    recv_thread = threading.Thread(
        target=udp_receiver,
        args=(sock,),
        daemon=True
    )

    recv_thread.start()

    print(
        f"Listening for IMU + EMG packets "
        f"on UDP {UDP_IP}:{UDP_PORT} ..."
    )

    print(
        "Starting dashboard (curses)..."
    )

    time.sleep(0.5)

    try:

        curses.wrapper(
            draw_dashboard
        )

    finally:

        state.running = False

        try:
            sock.close()
        except OSError:
            pass

        recv_thread.join(
            timeout=2.0
        )

        print(
            "Shut down cleanly."
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()
