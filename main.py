# UI/__init__.py or main app file
import sys

from sensors.ECG import ECG
from sensors.HRV import HRV
from UI.pages.dashboard import Dashboard
from PySide6.QtWidgets import QApplication, QMainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ecg = ECG()
    hrv = HRV()
    window = Dashboard(ecg=ecg, hrv=hrv)
    window.show()
    sys.exit(app.exec())
