import socket
import threading
import json
import time
import sys

# Import the existing sensor/EMG processing state.
# This file does NOT duplicate or modify the IMU/EMG calculations.
import pi_rom_server


HOST = "0.0.0.0"
PORT = 6000


clients = []
clients_lock = threading.Lock()


# ============================================================================
# DASHBOARD DATA
# ============================================================================

def get_dashboard_data():
    """Read the latest values from the existing ROM/EMG server."""

    angle, stale1, stale2 = (
        pi_rom_server.compute_angle_and_update_rom()
    )

    emg = pi_rom_server.compute_emg_snapshot()

    # Get the newest 100 raw EMG samples.
    #
    # Each sample is:
    # [sample_number, timestamp_us, adc_value]
    emg_samples = pi_rom_server.get_latest_emg_samples(100)

    with pi_rom_server.state.lock:

        rom_max = pi_rom_server.state.rom_max
        rom_min = pi_rom_server.state.rom_min

        calibrated = (
            pi_rom_server.state.calib_offset is not None
        )

    return {
        "type": "sensor_update",

        # ================================================================
        # IMU / ROM
        # ================================================================

        "angle_deg": angle,

        "rom_max_deg": rom_max,

        "rom_min_deg": rom_min,

        "sensor1_stale": stale1,

        "sensor2_stale": stale2,

        "imu_calibrated": calibrated,

	"imu_calibration_status": (
    		"Calibrated"
    		if calibrated
    		else "Not calibrated"
	),

        # ================================================================
        # EMG statistics
        # ================================================================

        "emg_stale": emg.stale,

        "emg_rate_sps": emg.rate_sps,

        "emg_latest_adc": emg.latest_adc_value,

        "emg_sample_count": emg.sample_count,

        "emg_packet_count": emg.packet_count,

        "emg_sample_gaps": emg.sample_gap_count,

        "emg_packet_gaps": emg.packet_gap_count,

        "emg_malformed": emg.malformed_packet_count,

        "emg_buffer_len": emg.buffer_len,

        "emg_buffer_capacity": emg.buffer_capacity,

        # ================================================================
        # RAW EMG SAMPLES
        # ================================================================

        "emg_samples": emg_samples,

        # ================================================================
        # Dashboard timestamp
        # ================================================================

        "timestamp": time.time(),
    }


# ============================================================================
# SEND TO ALL DASHBOARDS
# ============================================================================

def send_to_all(message):
    """Send one JSON message to every connected dashboard."""

    data = (
        json.dumps(
            message,
            separators=(",", ":")
        ) + "\n"
    ).encode("utf-8")

    dead = []

    with clients_lock:
        current_clients = list(clients)

    for conn in current_clients:

        try:

            conn.sendall(data)

        except OSError:

            dead.append(conn)

    if dead:

        with clients_lock:

            for conn in dead:

                if conn in clients:
                    clients.remove(conn)

                try:
                    conn.close()

                except OSError:
                    pass


# ============================================================================
# HANDLE WINDOWS COMMAND
# ============================================================================

def handle_dashboard_command(conn, command):
    """
    Handle a command received from the Windows MyoTrack dashboard.

    Currently supported:

        calibrate_imu

    """

    command = str(command).strip()

    # ------------------------------------------------------------------------
    # CALIBRATE IMU
    # ------------------------------------------------------------------------

    if command == "calibrate_imu":

        print(
            "Dashboard requested IMU calibration."
        )

        ok = pi_rom_server.calibrate_zero()

        if ok:

            print(
                "IMU calibration successful."
            )

            response = {
                "type": "command_ack",

                "command": "calibrate_imu",

                "success": True,

                "imu_calibrated": True,

                "imu_calibration_status": (
                    "Calibrated"
                ),

                "angle_deg": 0.0,

                "rom_max_deg": None,

                "rom_min_deg": None,

                "timestamp": time.time(),
            }

        else:

            print(
                "IMU calibration failed: "
                "sensor(s) not live."
            )

            response = {
                "type": "command_ack",

                "command": "calibrate_imu",

                "success": False,

                "imu_calibrated": False,

                "imu_calibration_status": (
                    "Calibration failed: "
                    "sensor(s) not live"
                ),

                "timestamp": time.time(),
            }

        try:

            data = (
                json.dumps(
                    response,
                    separators=(",", ":")
                ) + "\n"
            ).encode("utf-8")

            conn.sendall(data)

        except OSError:

            pass

        return

    # ------------------------------------------------------------------------
    # UNKNOWN COMMAND
    # ------------------------------------------------------------------------

    print(
        f"Unknown dashboard command: {command}"
    )

    response = {
        "type": "command_ack",

        "command": command,

        "success": False,

        "error": "Unknown command",

        "timestamp": time.time(),
    }

    try:

        data = (
            json.dumps(
                response,
                separators=(",", ":")
            ) + "\n"
        ).encode("utf-8")

        conn.sendall(data)

    except OSError:

        pass


# ============================================================================
# DASHBOARD CLIENT HANDLER
# ============================================================================

def client_handler(conn, addr):

    print(
        f"Dashboard connected: {addr}"
    )

    with clients_lock:

        clients.append(conn)

    # ------------------------------------------------------------------------
    # IMPORTANT
    #
    # Previously this function did:
    #
    #     conn.recv(1)
    #
    # merely to keep the connection alive.
    #
    # Now Windows sends newline-delimited JSON commands, so we must actually
    # read the incoming stream.
    # ------------------------------------------------------------------------

    buffer = b""

    try:

        while True:

            chunk = conn.recv(4096)

            if not chunk:
                break

            buffer += chunk

            while b"\n" in buffer:

                line, buffer = buffer.split(
                    b"\n",
                    1
                )

                line = line.strip()

                if not line:
                    continue

                try:

                    message = json.loads(
                        line.decode("utf-8")
                    )

                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError
                ):

                    print(
                        "Invalid dashboard JSON command."
                    )

                    continue

                if not isinstance(message, dict):
                    continue

                command = message.get(
                    "command"
                )

                if command is not None:

                    handle_dashboard_command(
                        conn,
                        command
                    )

    except OSError:

        pass

    finally:

        with clients_lock:

            if conn in clients:
                clients.remove(conn)

        try:

            conn.close()

        except OSError:

            pass

        print(
            f"Dashboard disconnected: {addr}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    # ========================================================================
    # Start existing Pi ROM + EMG server in the background.
    #
    # Its curses dashboard is disabled because this process is being
    # used as the Windows dashboard data source.
    # ========================================================================

    pi_rom_server.curses.wrapper = None

    print(
        "Starting existing IMU + EMG server..."
    )

    # ========================================================================
    # UDP receiver
    # ========================================================================

    udp_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    udp_sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    udp_sock.bind(
        ("0.0.0.0", 5005)
    )

    receiver_thread = threading.Thread(
        target=pi_rom_server.udp_receiver,
        args=(udp_sock,),
        daemon=True
    )

    receiver_thread.start()

    # ========================================================================
    # TCP server for Windows MyoTrack
    # ========================================================================

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind(
        (HOST, PORT)
    )

    server.listen(5)

    print(
        f"Dashboard TCP server listening on {PORT}"
    )

    print(
        "Waiting for Windows MyoTrack connection..."
    )

    accept_thread = threading.Thread(
        target=accept_loop,
        args=(server,),
        daemon=True
    )

    accept_thread.start()

    # ========================================================================
    # Dashboard update loop
    #
    # 20 updates/sec.
    #
    # Each update contains:
    #
    #   - IMU angle
    #   - ROM
    #   - calibration status
    #   - EMG statistics
    #   - newest 100 raw EMG samples
    # ========================================================================

    try:

        while True:

            message = get_dashboard_data()

            with clients_lock:

                has_clients = (
                    len(clients) > 0
                )

            if has_clients:

                send_to_all(message)

            time.sleep(
                0.05
            )

    except KeyboardInterrupt:

        print(
            "\nStopping..."
        )

    finally:

        try:

            udp_sock.close()

        except OSError:

            pass

        try:

            server.close()

        except OSError:

            pass


# ============================================================================
# ACCEPT LOOP
# ============================================================================

def accept_loop(server):

    """Accept Windows dashboard TCP connections."""

    while True:

        try:

            conn, addr = server.accept()

            threading.Thread(
                target=client_handler,
                args=(
                    conn,
                    addr
                ),
                daemon=True
            ).start()

        except OSError:

            break


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()
