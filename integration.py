import nidaqmx
import numpy as np
from scipy import signal as sg
import time

CHANNEL = "Dev1/ai0"
FS = 2500
CHUNK = 100          # samples per read (~40ms)
WIN = int(FS * 0.2)  # 200ms window for RMS

nyq = FS / 2
bp_sos = sg.butter(4, [20/nyq, 450/nyq], btype='bandpass', output='sos')
notch_b, notch_a = sg.iirnotch(50/nyq, 30)

bp_zi = sg.sosfilt_zi(bp_sos)
notch_zi = sg.lfilter_zi(notch_b, notch_a)
buffer = np.empty(0)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(CHANNEL)
    task.timing.cfg_samp_clk_timing(rate=FS, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
    task.start()

    print("Live filtered RMS — flex your muscle...")
    start = time.time()
    while time.time() - start < 10:
        chunk = np.array(task.read(number_of_samples_per_channel=CHUNK))

        filtered, bp_zi = sg.sosfilt(bp_sos, chunk, zi=bp_zi)
        filtered, notch_zi = sg.lfilter(notch_b, notch_a, filtered, zi=notch_zi)
        buffer = np.concatenate([buffer, filtered])

        if len(buffer) >= WIN:
            w = buffer[:WIN]
            buffer = buffer[WIN:]
            rms = float(np.sqrt(np.mean(w**2)))
            print(f"t={time.time()-start:.1f}s  RMS={rms:.3f}")