import curses
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple
 
# ============================== USER CONFIG =================================
UDP_IP = "0.0.0.0"      # listen on all interfaces
UDP_PORT = 5005
STALE_AFTER_S = 0.5     # a sensor is considered "lost" if silent this long
# =============================================================================
 
Quat = Tuple[float, float, float, float]  # (w, x, y, z)
 
 
# ----------------------------------------------------------------------------
# Quaternion math (pure Python, no numpy/scipy needed)
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
# Shared state
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
# UDP receiver thread
# ----------------------------------------------------------------------------
def udp_receiver(sock: socket.socket):
    while state.running:
        try:
            data, _addr = sock.recvfrom(1024)
        except OSError:
            break  # socket closed on shutdown
 
        try:
            text = data.decode("utf-8", errors="ignore").strip()
            parts = text.split(",")
            if len(parts) != 6:
                continue
            cid = int(parts[0])
            seq = int(parts[1])
            w, x, y, z = (float(p) for p in parts[2:6])
        except (ValueError, IndexError):
            continue
 
        if cid not in (1, 2):
            continue
 
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
 
 
# ----------------------------------------------------------------------------
# Angle / ROM computation (called from the UI loop)
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
 
        stdscr.erase()
        h, w = stdscr.getmaxyx()
 
        def put(row, col, text, attr=0):
            if 0 <= row < h and 0 <= col < w:
                stdscr.addnstr(row, col, text, max(0, w - col - 1), attr)
 
        put(0, 2, "LIVE JOINT ROM MONITOR", curses.A_BOLD | curses.color_pair(4))
        put(1, 2, "-" * min(60, w - 4))
 
        # Sensor link status
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
 
    print(f"Listening for IMU packets on UDP {UDP_IP}:{UDP_PORT} ...")
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
 
