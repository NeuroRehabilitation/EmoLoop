import sys
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QFileDialog,
    QSizePolicy,
    QGridLayout,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class Dashboard(QMainWindow):
    def __init__(self, ecg, hrv):
        super().__init__()
        self.ecg = ecg
        self.hrv = hrv

        self.signal = None
        self.time = None
        self.ecg_data = None
        self.hrv_data = None

        self.setWindowTitle("EmoLoop ECG Dashboard")
        self.resize(1400, 800)

        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        self.metrics_panel = QFrame()
        self.metrics_panel.setFrameShape(QFrame.StyledPanel)
        self.metrics_panel.setFixedWidth(280)

        metrics_layout = QVBoxLayout(self.metrics_panel)

        self.load_btn = QPushButton("Load ECG Data")
        self.process_btn = QPushButton("Process HRV")
        self.load_btn.clicked.connect(self.load_ecg)
        self.process_btn.clicked.connect(self.process)

        metrics_layout.addWidget(self.load_btn)
        metrics_layout.addWidget(self.process_btn)

        metrics_layout.addSpacing(10)

        self.avg_hr_var = QLabel("Avg HR: --")
        self.min_hr_var = QLabel("Min HR: --")
        self.max_hr_var = QLabel("Max HR: --")
        self.hr_std_var = QLabel("Std HR: --")
        self.sdnn_var = QLabel("SDNN: --")
        self.rmssd_var = QLabel("RMSSD: --")
        self.nn50_var = QLabel("NN50: --")
        self.pnn50_var = QLabel("pNN50: --")
        self.nn20_var = QLabel("NN20: --")
        self.pnn20_var = QLabel("pNN20: --")
        self.vlf_power_var = QLabel("VLF Power: --")
        self.lf_power_var = QLabel("LF Power: --")
        self.hf_power_var = QLabel("HF Power: --")
        self.total_power_var = QLabel("Total Power: --")
        self.lf_norm_var = QLabel("LF (nu): --")
        self.hf_norm_var = QLabel("HF (nu): --")
        self.lf_hf_var = QLabel("LF/HF: --")
        self.std_var = QLabel("STD: --")
        self.sdsd_var = QLabel("SDSD: --")
        self.sd2_var = QLabel("SD2: --")
        self.sd1_var = QLabel("SD1: --")
        self.sd2_sd1_var = QLabel("SD2/SD1: --")
        self.rpeaks_var = QLabel("R-peaks: --")

        self.metric_labels = [
            self.avg_hr_var,
            self.min_hr_var,
            self.max_hr_var,
            self.hr_std_var,
            self.sdnn_var,
            self.rmssd_var,
            self.nn50_var,
            self.pnn50_var,
            self.nn20_var,
            self.pnn20_var,
            self.vlf_power_var,
            self.lf_power_var,
            self.hf_power_var,
            self.total_power_var,
            self.lf_norm_var,
            self.hf_norm_var,
            self.lf_hf_var,
            self.std_var,
            self.sdsd_var,
            self.sd2_var,
            self.sd1_var,
            self.sd2_sd1_var,
            self.rpeaks_var,
        ]

        for lbl in self.metric_labels:
            lbl.setAlignment(Qt.AlignLeft)
            metrics_layout.addWidget(lbl)

        metrics_layout.addStretch(1)

        self.status_label = QLabel("Ready")
        self.status_label.setFrameShape(QFrame.Box)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        left_container = QVBoxLayout()
        left_container.addWidget(self.metrics_panel)

        left_widget = QWidget()
        left_widget.setLayout(left_container)
        left_widget.setMaximumWidth(320)

        self.plot_panel = QFrame()
        self.plot_panel.setFrameShape(QFrame.StyledPanel)
        plot_layout = QVBoxLayout(self.plot_panel)

        self.canvas = FigureCanvas(Figure(figsize=(8, 4), dpi=100))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.canvas.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.status_label)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.plot_panel, 1)

    def _build_plot(self):
        self.ax.set_title("ECG Signal")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)

        (self.ecg_line,) = self.ax.plot([], [], lw=1, label="ECG")
        (self.peak_line,) = self.ax.plot([], [], "ro", ms=4, label="R-peaks")
        self.ax.legend(loc="upper right")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def load_ecg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ECG data", "", "TXT Files (*.txt);;All Files (*.*)"
        )
        if not path:
            return

        try:
            self.data = np.loadtxt(path)
            self.signal = self.data[:, 2]
            self.time = np.arange(len(self.signal)) / self.ecg.sampling_rate
            self._update_plot(self.signal, None)
            self.status_label.setText(f"Loaded {len(self.signal)} samples")
        except Exception as e:
            self.status_label.setText(f"Load error: {e}")

    def process(self):
        if self.signal is None:
            self.status_label.setText("No ECG data loaded")
            return

        try:
            self.ecg_data = self.ecg.process(self.signal)

            self.hrv_data = self.hrv.process(
                rr_intervals=self.ecg_data["rr_intervals"],
                rr_time=self.ecg_data["rr_time"],
            )

            filtered = self.ecg_data["filtered_signal"]
            r_peaks = self.ecg_data["r_peaks"]
            hr = self.hrv_data["heart_rate"]

            hrv_time_domain = self.hrv_data["time_features"]
            hrv_freq_domain = self.hrv_data["frequency_features"]
            hrv_nonlinear = self.hrv_data["nonlinear_features"]

            self.avg_hr_var.setText(f"Avg HR: {hr.get('Avg HR', np.nan):.2f} bpm")
            self.min_hr_var.setText(f"Min HR: {hr.get('Min HR', np.nan):.2f} bpm")
            self.max_hr_var.setText(f"Max HR: {hr.get('Max HR', np.nan):.2f} bpm")
            self.hr_std_var.setText(f"Std HR: {hr.get('SD HR', np.nan):.2f} bpm")

            self.sdnn_var.setText(f"SDNN: {hrv_time_domain.get('SDNN', np.nan):.4f}")
            self.rmssd_var.setText(f"RMSSD: {hrv_time_domain.get('RMSSD', np.nan):.4f}")
            self.nn50_var.setText(f"NN50: {hrv_time_domain.get('NN50', np.nan)}")
            self.pnn50_var.setText(f"pNN50: {hrv_time_domain.get('pNN50', np.nan):.2f}")
            self.nn20_var.setText(f"NN20: {hrv_time_domain.get('NN20', np.nan)}")
            self.pnn20_var.setText(f"pNN20: {hrv_time_domain.get('pNN20', np.nan):.2f}")

            self.vlf_power_var.setText(
                f"VLF Power: {hrv_freq_domain.get('VLF_Power', np.nan):.4f}"
            )
            self.lf_power_var.setText(
                f"LF Power: {hrv_freq_domain.get('LF_Power', np.nan):.4f}"
            )
            self.hf_power_var.setText(
                f"HF Power: {hrv_freq_domain.get('HF_Power', np.nan):.4f}"
            )
            self.total_power_var.setText(
                f"Total Power: {hrv_freq_domain.get('Total_Power', np.nan):.4f}"
            )
            self.lf_norm_var.setText(
                f"LF (nu): {hrv_freq_domain.get('LF_(nu)', np.nan):.2f}"
            )
            self.hf_norm_var.setText(
                f"HF (nu): {hrv_freq_domain.get('HF_(nu)', np.nan):.2f}"
            )
            self.lf_hf_var.setText(f"LF/HF: {hrv_freq_domain.get('LF/HF', np.nan):.2f}")

            self.std_var.setText(f"STD: {hrv_nonlinear.get('STD', np.nan):.4f}")
            self.sdsd_var.setText(f"SDSD: {hrv_nonlinear.get('SDSD', np.nan):.4f}")
            self.sd2_var.setText(f"SD2: {hrv_nonlinear.get('SD2', np.nan):.4f}")
            self.sd1_var.setText(f"SD1: {hrv_nonlinear.get('SD1', np.nan):.4f}")
            self.sd2_sd1_var.setText(
                f"SD2/SD1: {hrv_nonlinear.get('SD2/SD1', np.nan):.4f}"
            )

            self.rpeaks_var.setText(f"R-peaks: {len(r_peaks)}")
            self.status_label.setText("ECG processed successfully")

            self._update_plot(filtered, r_peaks)

        except Exception as e:
            self.status_label.setText(f"Processing error: {e}")

    def _update_plot(self, signal, r_peaks):
        if signal is None:
            return

        t = np.arange(len(signal)) / self.ecg.sampling_rate

        self.ax.clear()
        self.ax.plot(t, signal, lw=1, label="ECG")

        if r_peaks is not None and len(r_peaks) > 0:
            self.ax.plot(t[r_peaks], signal[r_peaks], "ro", ms=4, label="R-peaks")

        self.ax.set_title("ECG Signal")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right")
        self.canvas.figure.tight_layout()
        self.canvas.draw()
