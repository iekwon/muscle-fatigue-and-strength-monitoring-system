"""
Standalone DAQ reader -- deliberately has NO PyQt5 import anywhere in this
file, and is meant to be launched as a genuinely separate interpreter process
(via QProcess / subprocess, NOT via multiprocessing.Process on Windows, which
re-imports the launching script's module -- including any PyQt5 import at
the top of that file -- defeating the isolation).

Run standalone to sanity check it on its own:
    python daq_reader.py

Prints one line per RMS sample to stdout, e.g.:
    SAMPLE 0.01234
or on failure:
    ERROR OSError: exception: access violation reading 0x0000000000000000
"""
import numpy as np
from scipy import signal as sg
import nidaqmx

CHANNEL = "Dev1/ai0"
FS = 2500
CHUNK = 100
WIN = int(FS * 0.2)


def main():
    nyq = FS / 2
    bp_sos = sg.butter(4, [20 / nyq, 450 / nyq], btype='bandpass', output='sos')
    notch_b, notch_a = sg.iirnotch(50 / nyq, 30)
    bp_zi = sg.sosfilt_zi(bp_sos)
    notch_zi = sg.lfilter_zi(notch_b, notch_a)
    buffer = np.empty(0)

    try:
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(CHANNEL)
            task.timing.cfg_samp_clk_timing(
                rate=FS, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            task.start()
            while True:
                chunk = np.array(task.read(number_of_samples_per_channel=CHUNK))
                filtered, bp_zi = sg.sosfilt(bp_sos, chunk, zi=bp_zi)
                filtered, notch_zi = sg.lfilter(notch_b, notch_a, filtered, zi=notch_zi)
                buffer = np.concatenate([buffer, filtered])
                if len(buffer) >= WIN:
                    w = buffer[:WIN]
                    buffer = buffer[WIN:]
                    rms = float(np.sqrt(np.mean(w ** 2)))
                    print(f"SAMPLE {rms}", flush=True)
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()