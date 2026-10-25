"""
GUI half of the isolation test. Launches daq_reader.py as a genuinely
separate process via QProcess (not multiprocessing) and reads its stdout
line by line. This process never imports nidaqmx at all -- only the child
process (running daq_reader.py, a plain script with zero Qt imports) touches
the DAQ.

Run with both files in the same folder:
    python daq_pyqt_repro_qprocess.py
"""
import sys
from PyQt5 import QtWidgets, QtCore


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QProcess DAQ isolation test")
        self.label = QtWidgets.QLabel("starting...")
        self.label.setStyleSheet("font-size: 20px; padding: 20px;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        self.resize(420, 140)

        self.process = QtCore.QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(["daq_reader.py"])
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.start()

    def _on_stdout(self):
        while self.process.canReadLine():
            line = bytes(self.process.readLine()).decode(errors="replace").strip()
            if line.startswith("SAMPLE"):
                rms = float(line.split()[1])
                self.label.setText(f"RMS: {rms:.5f}")
            elif line.startswith("ERROR"):
                self.label.setText(line)

    def _on_stderr(self):
        err = bytes(self.process.readAllStandardError()).decode(errors="replace")
        if err.strip():
            self.label.setText(f"STDERR:\n{err[-300:]}")

    def closeEvent(self, event):
        self.process.terminate()
        self.process.waitForFinished(1000)
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.process.kill()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()