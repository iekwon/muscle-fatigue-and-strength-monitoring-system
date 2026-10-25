#!/usr/bin/env python3

import socket
import struct
import time
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", port))
sock.settimeout(0.5)

exp_sn = None
exp_seq = None

tot = 0
pk = 0
gap_samples = 0
gap_pkts = 0
bad = 0

last_ts = None
dtmin = 10**9
dtmax = 0

t0 = time.time()
w_samples = 0
w_pkts = 0

print(f"Listening on UDP {port}")

while True:
    try:
        data, _ = sock.recvfrom(2048)
    except socket.timeout:
        data = None

    if data and data[:1] == b"E" and len(data) >= 12:

        _, cid, seq, ts, n = struct.unpack_from(
            "<BBIIH", data, 0
        )

        if len(data) != 12 + 10 * n:
            bad += 1

        else:
            if exp_seq is not None and seq != exp_seq:
                gap_pkts += (seq - exp_seq) & 0xFFFFFFFF

            exp_seq = (seq + 1) & 0xFFFFFFFF

            pk += 1
            w_pkts += 1

            for i in range(n):

                sn, sts, adc = struct.unpack_from(
                    "<IIH",
                    data,
                    12 + 10 * i
                )

                if exp_sn is not None and sn != exp_sn:
                    gap_samples += (sn - exp_sn) & 0xFFFFFFFF

                exp_sn = (sn + 1) & 0xFFFFFFFF

                if last_ts is not None:
                    d = (sts - last_ts) & 0xFFFFFFFF
                    dtmin = min(dtmin, d)
                    dtmax = max(dtmax, d)

                last_ts = sts

                tot += 1
                w_samples += 1

    now = time.time()

    if now - t0 >= 1.0:

        dt = now - t0

        print(
            f"{w_samples/dt:7.1f} samp/s  "
            f"{w_pkts/dt:5.1f} pkt/s | "
            f"total={tot} "
            f"sn_gaps={gap_samples} "
            f"seq_gaps={gap_pkts} "
            f"bad_len={bad} | "
            f"ts_dt_us=[{dtmin}..{dtmax}]"
        )

        t0 = now
        w_samples = 0
        w_pkts = 0
        dtmin = 10**9
        dtmax = 0
