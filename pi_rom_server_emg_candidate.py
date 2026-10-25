#!/usr/bin/env python3
"""
============================================================================
 Raspberry Pi ROM + EMG Server
============================================================================
Receives two kinds of UDP packets on the same port from the ESP32 nodes:

  IMU packets (ASCII text, from BOTH ESP32 #1 and #2):
      "<id>,<seq>,<w>,<x>,<y>,<z>\n"
      id = 1 (proximal segment) or 2 (distal segment)

  EMG packets (binary, from ESP32 #1 only -- relays Nano ADC samples):
      Header (12 bytes, little-endian):  struct "<BBIIH"
          marker       uint8   = ord('E')
          client_id    uint8
          batch_seq    uint32  (ESP32-assigned packet counter)
          esp32_micros uint32  (ESP32 micros() at packet build time)
          sample_count uint16  (N samples follow)
      Then N samples, 10 bytes each, little-endian:  struct "<IIH"
          sample_number uint32  (from Nano)
          timestamp_us  uint32  (from Nano)
          adc_value     uint16  (from Nano)

Packet-type dispatch is done on the raw first byte, BEFORE any UTF-8 decode
is attempted (EMG payload is binary and is not valid/meaningful text):
    data[0] == ord('E')  -> EMG binary packet
    otherwise             -> existing IMU ASCII packet

--------------------------------------------------------------------------
ROM / joint-angle math (UNCHANGED from the original IMU-only version):

    q_rel = q1^-1 (x) q2          (quaternion "difference" between segments)
    angle = 2 * atan2(|vec(q_rel)|, |w(q_rel)|)   in degrees

Calibration ('c') captures q_rel at a reference pose; all angles are then
reported relative to that pose. ROM is running max - min since last reset
('r'). See the original module docstring history for details -- none of
this behavior changed.

--------------------------------------------------------------------------
EMG handling:

  - Decoding happens directly in the UDP receiver thread (it's cheap --
    just struct.unpack calls), so nothing blocks the socket loop.
  - Raw samples are stored, unmodified, in a fixed-size ring buffer
    (collections.deque(maxlen=...)) for later processing -- no filtering,
    rectification, RMS, or FFT is done here.
  - EMG state (buffer + counters) lives in its own dataclass with its own
    lock, kept deliberately separate from the IMU/ROM `state` object so the
    two data paths don't contend with each other.
  - Malformed / truncated / unexpected-length EMG packets are counted and
    discarded; they can never crash the server or corrupt IMU processing.
  - Two independent gap counters are tracked, and are NEVER conflated:
      * packet_gap_count -- gaps in the ESP32's per-UDP-packet batch_seq
        counter (i.e. UDP packets lost between ESP32 and Pi).
      * sample_gap_count -- gaps in the Nano's per-sample sample_number
        counter (i.e. actual ADC samples the Nano believes it produced
        but that never arrived here). A UDP packet gap is NOT assumed to
        mean any particular number of Nano samples were dropped, and a
        Nano sample gap is NOT assumed to mean a UDP packet was lost --
        each is measured independently from its own counter field.
  - No samples are ever synthesized/invented to "fill in" a gap; gaps are
    only counted, never interpolated.

No external dependencies beyond the Python standard library (curses is
part of the standard library on Linux, which includes Raspberry Pi OS).

Run:
    python3 pi_rom_server.py
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

# ============================== USER CONFIG =================================
UDP_IP = "0.0.0.0"      # listen on all interfaces
UDP_PORT = 5005
STALE_AFTER_S = 0.5     # a sensor/feed is considered "lost" if silent this long

EMG_RING_BUFFER_CAPACITY = 20000   # ~10s of raw samples at 2000 Hz; storage only
# =============================================================================

Quat = Tuple[float, float, float, float]  # (w, x, y, z)


# ----------------------------------------------------------------------------
# Quaternion math (pure Python, no numpy/scipy needed) -- UNCHANGED
# ----------------------------------------------------------------------------
def quat_mul(q1: Quat, q2: Quat) -> Quat:
    """Hamilton product q1 (x) q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_conj(q: Quat) -> Quat:
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_normalize(q: Quat) -> Quat:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def quat_angle_deg(q: Quat) -> float:
    """
    Rotation angle represented by unit quaternion q, in degrees, folded
    into [0, 180]. Uses atan2 on the vector-part magnitude rather than
    acos(w) -- numerically stable near 0 deg and near 180 deg, and
    immune to the q / -q double-cover ambiguity (same rotation, sign
    flip) because we take abs(w).
    """
    w, x, y, z = quat_normalize(q)
    vec_norm = math.sqrt(x * x + y * y + z * z)
    return 2.0 * math.degrees(math.atan2(vec_norm, abs(w)))


def relative_quat(q_proximal: Quat, q_distal: Quat) -> Quat:
    """q1^-1 (x) q2 -- for a unit quaternion, inverse == conjugate."""
    return quat_mul(quat_conj(q_proximal), q_distal)


# ----------------------------------------------------------------------------
# IMU / ROM shared state -- UNCHANGED
# ----------------------------------------------------------------------------
@dataclass
class SensorState:
    q: Quat = (1.0, 0.0, 0.0, 0.0)
    last_seen: float = 0.0
    seq: int = -1
    rate_hz: float = 0.0
    _count_since: int = field(default=0, repr=False)
    _rate_window_start: float = field(default=0.0, repr=False)


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.sensors = {1: SensorState(), 2: SensorState()}
        self.calib_offset: Optional[Quat] = None   # q_rel captured at zero pose
        self.rom_max: Optional[float] = None
        self.rom_min: Optional[float] = None
        self.current_angle: Optional[float] = None
        self.running = True


state = SharedState()


# ----------------------------------------------------------------------------
# EMG shared state -- deliberately separate lock from `state` above so
# EMG traffic never blocks/contends with IMU/ROM processing.
# ----------------------------------------------------------------------------
EMG_MARKER_BYTE = ord("E")
EMG_HEADER_STRUCT = struct.Struct("<BBIIH")   # marker, id, batch_seq, esp_micros, count
EMG_HEADER_SIZE = EMG_HEADER_STRUCT.size      # 12 bytes
EMG_SAMPLE_STRUCT = struct.Struct("<IIH")     # sample_number, timestamp_us, adc_value
EMG_SAMPLE_SIZE = EMG_SAMPLE_STRUCT.size      # 10 bytes

# Guard used when computing sample-number gap sizes: a raw (actual -
# expected) delta this large almost certainly means a counter reset /
# 32-bit wraparound / first-ever packet edge case rather than a genuine
# multi-hundred-thousand-sample gap, so it's counted as a single gap
# event instead of being added to the running total verbatim.
EMG_SAMPLE_GAP_SANITY_LIMIT = 1_000_000


@dataclass
class EmgSample:
    sample_number: int
    timestamp_us: int
    adc_value: int


class EmgState:
    def __init__(self):
        self.lock = threading.Lock()
        # Fixed-size ring buffer of raw samples. No processing is done on
        # them here -- deque.append() auto-evicts the oldest entry once
        # `maxlen` is reached, which is exactly the ring-buffer behavior
        # we want (newest data is always kept, oldest silently ages out).
        self.buffer: deque = deque(maxlen=EMG_RING_BUFFER_CAPACITY)

        self.packet_count: int = 0
        self.sample_count: int = 0          # total valid samples ever received
        self.malformed_packet_count: int = 0
        self.dropped_sample_count: int = 0  # samples evicted because buffer was full
        self.last_client_id: Optional[int] = None
        self.last_seen: float = 0.0
        self.rate_sps: float = 0.0
        self.latest_adc_value: Optional[int] = None

        # Packet-level loss indicator, from the ESP32-assigned batch_seq
        # counter in the packet header. This reflects UDP packets lost
        # in transit -- it says nothing about how many (if any) individual
        # ADC samples were lost, since a lost packet's sample_count is
        # unknown to us.
        self.last_batch_seq: Optional[int] = None
        self.packet_gap_count: int = 0

        # Sample-level loss indicator, from the Nano-assigned sample_number
        # field carried in every sample. This is the authoritative measure
        # of actual missing ADC samples and is tracked completely
        # independently of packet_gap_count (see module docstring).
        self.last_sample_number: Optional[int] = None
        self.sample_gap_count: int = 0

        self._count_since: int = 0
        self._rate_window_start: float = 0.0


emg_state = EmgState()


# ----------------------------------------------------------------------------
# UDP receiver thread
# ----------------------------------------------------------------------------
def handle_imu_packet(data: bytes):
    """Existing IMU text-packet decode path -- logic unchanged."""
    try:
        text = data.decode("utf-8", errors="ignore").strip()
        parts = text.split(",")
        if len(parts) != 6:
            return
        cid = int(parts[0])
        seq = int(parts[1])
        w, x, y, z = (float(p) for p in parts[2:6])
    except (ValueError, IndexError):
        return

    if cid not in (1, 2):
        return

    now = time.time()
    with state.lock:
        s = state.sensors[cid]
        s.q = (w, x, y, z)
        s.last_seen = now
        s.seq = seq
        s._count_since += 1
        if s._rate_window_start == 0.0:
            s._rate_window_start = now
        elapsed = now - s._rate_window_start
        if elapsed >= 1.0:
            s.rate_hz = s._count_since / elapsed
            s._count_since = 0
            s._rate_window_start = now


def handle_emg_packet(data: bytes):
    """
    Decode one binary EMG batch packet from ESP32 #1 and store its samples,
    unmodified, in the ring buffer. Robust against truncated/malformed
    packets -- any problem just increments a counter and returns, never
    raises out of the receiver loop. No samples are ever invented: gaps
    are only detected and counted, never filled in.
    """
    now = time.time()

    if len(data) < EMG_HEADER_SIZE:
        with emg_state.lock:
            emg_state.malformed_packet_count += 1
        return

    try:
        marker, client_id, batch_seq, esp_micros, sample_count = (
            EMG_HEADER_STRUCT.unpack_from(data, 0)
        )
    except struct.error:
        with emg_state.lock:
            emg_state.malformed_packet_count += 1
        return

    expected_len = EMG_HEADER_SIZE + sample_count * EMG_SAMPLE_SIZE
    if marker != EMG_MARKER_BYTE or client_id not in (1, 2) or len(data) != expected_len:
        with emg_state.lock:
            emg_state.malformed_packet_count += 1
        return

    # Unpack all samples up front (fast -- fixed-size struct, no branching)
    # before taking the lock, so we hold it for as short a time as possible.
    try:
        samples = [
            EMG_SAMPLE_STRUCT.unpack_from(data, EMG_HEADER_SIZE + i * EMG_SAMPLE_SIZE)
            for i in range(sample_count)
        ]
    except struct.error:
        with emg_state.lock:
            emg_state.malformed_packet_count += 1
        return

    with emg_state.lock:
        emg_state.packet_count += 1
        emg_state.last_seen = now
        emg_state.last_client_id = client_id

        # ---- Packet-level (UDP) gap tracking, from ESP32's batch_seq ----
        if emg_state.last_batch_seq is not None:
            expected_batch = (emg_state.last_batch_seq + 1) & 0xFFFFFFFF
            if batch_seq != expected_batch:
                emg_state.packet_gap_count += 1
        emg_state.last_batch_seq = batch_seq

        buf = emg_state.buffer
        cap = buf.maxlen
        for sample_number, timestamp_us, adc_value in samples:
            # ---- Sample-level (Nano) gap tracking, from sample_number ----
            # This is intentionally independent of the packet-level check
            # above: it measures actual missing ADC samples, not lost UDP
            # packets, per the "do not treat packet gaps as sample drops"
            # requirement.
            if emg_state.last_sample_number is not None:
                expected_sample = (emg_state.last_sample_number + 1) & 0xFFFFFFFF
                if sample_number != expected_sample:
                    missing = (sample_number - expected_sample) & 0xFFFFFFFF
                    if missing < EMG_SAMPLE_GAP_SANITY_LIMIT:
                        emg_state.sample_gap_count += missing
                    else:
                        emg_state.sample_gap_count += 1
            emg_state.last_sample_number = sample_number
            emg_state.latest_adc_value = adc_value

            if cap is not None and len(buf) >= cap:
                emg_state.dropped_sample_count += 1  # oldest sample about to be evicted
            buf.append(EmgSample(sample_number, timestamp_us, adc_value))

        n = len(samples)
        emg_state.sample_count += n
        emg_state._count_since += n
        if emg_state._rate_window_start == 0.0:
            emg_state._rate_window_start = now
        elapsed = now - emg_state._rate_window_start
        if elapsed >= 1.0:
            emg_state.rate_sps = emg_state._count_since / elapsed
            emg_state._count_since = 0
            emg_state._rate_window_start = now


def udp_receiver(sock: socket.socket):
    while state.running:
        try:
            data, _addr = sock.recvfrom(2048)  # EMG batches can be up to ~1012 bytes
        except OSError:
            break  # socket closed on shutdown

        if not data:
            continue

        # ---- Packet-type dispatch on the raw first byte ----
        # EMG packets start with ASCII 'E' (0x45) and are binary; IMU
        # packets start with ASCII '1' or '2' and are UTF-8 text. Checking
        # the byte first means we never attempt to UTF-8-decode binary EMG
        # data (which would be meaningless/could contain decode errors).
        if data[0] == EMG_MARKER_BYTE:
            handle_emg_packet(data)
        else:
            handle_imu_packet(data)


# ----------------------------------------------------------------------------
# Angle / ROM computation (called from the UI loop) -- UNCHANGED
# ----------------------------------------------------------------------------
def compute_angle_and_update_rom() -> Tuple[Optional[float], bool, bool]:
    """
    Returns (angle_deg_or_None, sensor1_stale, sensor2_stale).
    Updates state.current_angle / rom_max / rom_min as a side effect.
    """
    now = time.time()
    with state.lock:
        s1 = state.sensors[1]
        s2 = state.sensors[2]
        stale1 = (now - s1.last_seen) > STALE_AFTER_S or s1.seq < 0
        stale2 = (now - s2.last_seen) > STALE_AFTER_S or s2.seq < 0

        if stale1 or stale2:
            state.current_angle = None
            return None, stale1, stale2

        q_rel = relative_quat(s1.q, s2.q)

        if state.calib_offset is not None:
            q_delta = quat_mul(quat_conj(state.calib_offset), q_rel)
            angle = quat_angle_deg(q_delta)
        else:
            angle = quat_angle_deg(q_rel)

        state.current_angle = angle
        if state.rom_max is None or angle > state.rom_max:
            state.rom_max = angle
        if state.rom_min is None or angle < state.rom_min:
            state.rom_min = angle

        return angle, False, False


def calibrate_zero():
    """Capture current q_rel as the new zero-reference pose."""
    now = time.time()
    with state.lock:
        s1 = state.sensors[1]
        s2 = state.sensors[2]
        if (now - s1.last_seen) > STALE_AFTER_S or (now - s2.last_seen) > STALE_AFTER_S:
            return False  # can't calibrate without live data from both nodes
        state.calib_offset = relative_quat(s1.q, s2.q)
        state.rom_max = None
        state.rom_min = None
        state.current_angle = 0.0
        return True


def reset_rom():
    with state.lock:
        state.rom_max = None
        state.rom_min = None


# ----------------------------------------------------------------------------
# EMG snapshot -- called from the UI loop, mirrors compute_angle_and_update_rom
# in style but does no processing beyond reading counters, matching the "no
# advanced EMG features yet" requirement.
# ----------------------------------------------------------------------------
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
    now = time.time()
    with emg_state.lock:
        stale = (now - emg_state.last_seen) > STALE_AFTER_S or emg_state.last_seen == 0.0
        return EmgSnapshot(
            stale=stale,
            packet_count=emg_state.packet_count,
            sample_count=emg_state.sample_count,
            malformed_packet_count=emg_state.malformed_packet_count,
            dropped_sample_count=emg_state.dropped_sample_count,
            packet_gap_count=emg_state.packet_gap_count,
            sample_gap_count=emg_state.sample_gap_count,
            latest_adc_value=emg_state.latest_adc_value,
            buffer_len=len(emg_state.buffer),
            buffer_capacity=emg_state.buffer.maxlen or 0,
            rate_sps=emg_state.rate_sps,
        )


# ----------------------------------------------------------------------------
# curses dashboard
# ----------------------------------------------------------------------------
def draw_dashboard(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)  # ~20 fps UI refresh

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)

    last_status_msg = ""
    last_status_time = 0.0

    while state.running:
        angle, stale1, stale2 = compute_angle_and_update_rom()
        emg = compute_emg_snapshot()

        stdscr.erase()
        h, w = stdscr.getmaxyx()

        def put(row, col, text, attr=0):
            if 0 <= row < h and 0 <= col < w:
                stdscr.addnstr(row, col, text, max(0, w - col - 1), attr)

        put(0, 2, "LIVE JOINT ROM + EMG MONITOR", curses.A_BOLD | curses.color_pair(4))
        put(1, 2, "-" * min(60, w - 4))

        # ---- IMU sensor link status (unchanged) ----
        with state.lock:
            s1, s2 = state.sensors[1], state.sensors[2]
            r1, r2 = s1.rate_hz, s2.rate_hz

        def link_line(cid, label, stale, rate):
            color = curses.color_pair(2) if stale else curses.color_pair(1)
            status = "NO SIGNAL" if stale else f"OK  ({rate:4.1f} Hz)"
            return f"  ID {cid} [{label:>9}]: {status}", color

        line, color = link_line(1, "PROXIMAL", stale1, r1)
        put(3, 2, line, color)
        line, color = link_line(2, "DISTAL", stale2, r2)
        put(4, 2, line, color)

        put(6, 2, "-" * min(60, w - 4))

        # ---- Calibration / angle / ROM (unchanged) ----
        with state.lock:
            calibrated = state.calib_offset is not None
            rom_max = state.rom_max
            rom_min = state.rom_min

        calib_txt = "CALIBRATED" if calibrated else "NOT CALIBRATED (press 'c')"
        calib_color = curses.color_pair(1) if calibrated else curses.color_pair(3)
        put(8, 2, f"  Calibration: {calib_txt}", calib_color)

        if angle is not None:
            put(11, 2, f"  Current Joint Angle:   {angle:6.1f} deg",
                curses.A_BOLD | curses.color_pair(4))
        else:
            put(11, 2, "  Current Joint Angle:      --.- deg  (waiting for sensors)",
                curses.color_pair(2))

        if rom_max is not None and rom_min is not None:
            rom = rom_max - rom_min
            put(13, 2, f"  Active Set ROM:         {rom:6.1f} deg", curses.A_BOLD)
            put(14, 2, f"    Peak (max) angle:     {rom_max:6.1f} deg")
            put(15, 2, f"    Trough (min) angle:   {rom_min:6.1f} deg")
        else:
            put(13, 2, "  Active Set ROM:            --.- deg")

        # ---- EMG section ----
        put(17, 2, "-" * min(60, w - 4))
        put(18, 2, "EMG (ESP32 #1 <- Nano ADC)", curses.A_BOLD | curses.color_pair(4))

        emg_color = curses.color_pair(2) if emg.stale else curses.color_pair(1)
        emg_status_word = "NO DATA" if emg.stale else "OK"
        adc_txt = "----" if emg.latest_adc_value is None else str(emg.latest_adc_value)

        put(19, 2, f"  Status: {emg_status_word}", emg_color)
        put(20, 2, f"  Rate: {emg.rate_sps:8.1f} samples/s")
        put(21, 2, f"  Latest ADC: {adc_txt}")
        put(22, 2, f"  Packets: {emg.packet_count}")
        put(23, 2, f"  Sample gaps: {emg.sample_gap_count}")
        put(24, 2, f"  Packet gaps: {emg.packet_gap_count}")
        put(25, 2,
            f"  (buffer {emg.buffer_len}/{emg.buffer_capacity}  "
            f"malformed {emg.malformed_packet_count}  "
            f"overflow-dropped {emg.dropped_sample_count})",
            curses.A_DIM)

        put(h - 3, 2, "-" * min(60, w - 4))
        put(h - 2, 2,
            "  [c] calibrate/zero   [r] reset ROM   [q] quit",
            curses.A_DIM)

        if last_status_msg and (time.time() - last_status_time) < 2.0:
            put(9, 2, f"  >> {last_status_msg}", curses.color_pair(3))

        stdscr.refresh()

        try:
            key = stdscr.getch()
        except curses.error:
            key = -1

        if key in (ord("q"), ord("Q")):
            state.running = False
        elif key in (ord("c"), ord("C")):
            ok = calibrate_zero()
            last_status_msg = "Zero point set." if ok else "Can't calibrate: sensor(s) not live yet."
            last_status_time = time.time()
        elif key in (ord("r"), ord("R")):
            reset_rom()
            last_status_msg = "ROM reset."
            last_status_time = time.time()


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)  # so the receiver thread can check state.running

    recv_thread = threading.Thread(target=udp_receiver, args=(sock,), daemon=True)
    recv_thread.start()

    print(f"Listening for IMU + EMG packets on UDP {UDP_IP}:{UDP_PORT} ...")
    print("Starting dashboard (curses)...")
    time.sleep(0.5)

    try:
        curses.wrapper(draw_dashboard)
    finally:
        state.running = False
        sock.close()
        recv_thread.join(timeout=2.0)
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()
