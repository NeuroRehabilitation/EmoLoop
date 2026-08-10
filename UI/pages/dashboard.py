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
from matplotlib.patches import Ellipse


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
        self.resize(1400, 1000)

        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )
        main_layout.setSpacing(10)

        # ==================================================
        # Left metrics panel
        # ==================================================
        self.metrics_panel = QFrame()
        self.metrics_panel.setFrameShape(QFrame.StyledPanel)
        self.metrics_panel.setFixedWidth(280)

        metrics_layout = QVBoxLayout(self.metrics_panel)
        metrics_layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )
        metrics_layout.setSpacing(4)

        self.load_btn = QPushButton("Load ECG Data")

        self.process_btn = QPushButton("Process HRV")

        self.load_btn.clicked.connect(self.load_ecg)

        self.process_btn.clicked.connect(self.process)

        metrics_layout.addWidget(self.load_btn)

        metrics_layout.addWidget(self.process_btn)

        metrics_layout.addSpacing(10)

        # Heart-rate metrics
        self.avg_hr_var = QLabel("Avg HR: --")
        self.min_hr_var = QLabel("Min HR: --")
        self.max_hr_var = QLabel("Max HR: --")
        self.hr_std_var = QLabel("Std HR: --")

        # Time-domain metrics
        self.avg_rr_var = QLabel("Avg RR: --")
        self.min_rr_var = QLabel("Min RR: --")
        self.max_rr_var = QLabel("Max RR: --")
        self.sd_rr_var = QLabel("SD RR: --")
        self.sdnn_var = QLabel("SDNN: --")
        self.rmssd_var = QLabel("RMSSD: --")
        self.nn50_var = QLabel("NN50: --")
        self.pnn50_var = QLabel("pNN50: --")
        self.nn20_var = QLabel("NN20: --")
        self.pnn20_var = QLabel("pNN20: --")

        # Frequency-domain metrics
        self.vlf_power_var = QLabel("VLF Power: --")
        self.lf_power_var = QLabel("LF Power: --")
        self.hf_power_var = QLabel("HF Power: --")
        self.total_power_var = QLabel("Total Power: --")
        self.lf_norm_var = QLabel("LF (nu): --")
        self.hf_norm_var = QLabel("HF (nu): --")
        self.lf_hf_var = QLabel("LF/HF: --")

        # Nonlinear metrics
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
            self.avg_rr_var,
            self.min_rr_var,
            self.max_rr_var,
            self.sd_rr_var,
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

        for label in self.metric_labels:
            label.setAlignment(Qt.AlignLeft)
            metrics_layout.addWidget(label)

        metrics_layout.addStretch(1)

        left_container = QVBoxLayout()
        left_container.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        left_container.addWidget(self.metrics_panel)

        left_widget = QWidget()
        left_widget.setLayout(left_container)
        left_widget.setMaximumWidth(320)

        # ==================================================
        # Main plot panel
        # ==================================================
        self.plot_panel = QFrame()
        self.plot_panel.setFrameShape(QFrame.StyledPanel)

        plot_layout = QVBoxLayout(self.plot_panel)
        plot_layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )
        plot_layout.setSpacing(10)

        # ==================================================
        # ECG figure
        # ==================================================
        self.ecg_figure = Figure(
            figsize=(10, 4),
            dpi=100,
        )

        self.ecg_canvas = FigureCanvas(self.ecg_figure)

        self.ecg_canvas.setFixedSize(
            1200,
            400,
        )

        self.ecg_toolbar = NavigationToolbar(
            self.ecg_canvas,
            self,
        )

        self.ecg_toolbar.setFixedWidth(self.ecg_canvas.width())

        self.ecg_ax = self.ecg_figure.add_subplot(111)

        ecg_title = QLabel("ECG Signal")
        ecg_title.setAlignment(Qt.AlignHCenter)

        plot_layout.addWidget(
            ecg_title,
            alignment=Qt.AlignHCenter,
        )

        plot_layout.addWidget(
            self.ecg_toolbar,
            alignment=Qt.AlignHCenter,
        )

        plot_layout.addWidget(
            self.ecg_canvas,
            alignment=Qt.AlignHCenter,
        )

        # ==================================================
        # Poincaré figure
        # ==================================================
        self.poincare_figure = Figure(
            figsize=(6, 5),
            dpi=100,
        )

        self.poincare_canvas = FigureCanvas(self.poincare_figure)

        self.poincare_canvas.setFixedSize(
            580,
            500,
        )

        self.poincare_toolbar = NavigationToolbar(
            self.poincare_canvas,
            self,
        )

        self.poincare_toolbar.setFixedWidth(self.poincare_canvas.width())

        self.poincare_ax = self.poincare_figure.add_subplot(111)

        # ==================================================
        # Frequency-power figure
        # ==================================================
        self.frequency_figure = Figure(
            figsize=(6, 5),
            dpi=100,
        )

        self.frequency_canvas = FigureCanvas(self.frequency_figure)

        self.frequency_canvas.setFixedSize(
            580,
            500,
        )

        self.frequency_toolbar = NavigationToolbar(
            self.frequency_canvas,
            self,
        )

        self.frequency_toolbar.setFixedWidth(self.frequency_canvas.width())

        self.frequency_ax = self.frequency_figure.add_subplot(111)

        # ==================================================
        # Side-by-side Poincaré/frequency layout
        # ==================================================
        side_by_side_layout = QHBoxLayout()
        side_by_side_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        side_by_side_layout.setSpacing(20)

        # Poincaré container
        poincare_widget = QWidget()

        poincare_layout = QVBoxLayout(poincare_widget)
        poincare_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        poincare_layout.setSpacing(4)

        poincare_title = QLabel("Poincaré Plot")
        poincare_title.setAlignment(Qt.AlignHCenter)

        poincare_layout.addWidget(
            poincare_title,
            alignment=Qt.AlignHCenter,
        )

        poincare_layout.addWidget(
            self.poincare_toolbar,
            alignment=Qt.AlignHCenter,
        )

        poincare_layout.addWidget(
            self.poincare_canvas,
            alignment=Qt.AlignHCenter,
        )

        # Frequency container
        frequency_widget = QWidget()

        frequency_layout = QVBoxLayout(frequency_widget)
        frequency_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        frequency_layout.setSpacing(4)

        frequency_title = QLabel("HRV Frequency-Band Power")
        frequency_title.setAlignment(Qt.AlignHCenter)

        frequency_layout.addWidget(
            frequency_title,
            alignment=Qt.AlignHCenter,
        )

        frequency_layout.addWidget(
            self.frequency_toolbar,
            alignment=Qt.AlignHCenter,
        )

        frequency_layout.addWidget(
            self.frequency_canvas,
            alignment=Qt.AlignHCenter,
        )

        # Add both containers horizontally
        side_by_side_layout.addWidget(poincare_widget)

        side_by_side_layout.addWidget(frequency_widget)

        plot_layout.addLayout(side_by_side_layout)

        # ==================================================
        # Status label
        # ==================================================
        self.status_label = QLabel("Ready")

        self.status_label.setFrameShape(QFrame.Box)

        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        plot_layout.addWidget(self.status_label)

        # ==================================================
        # Main layout
        # ==================================================
        main_layout.addWidget(left_widget)

        main_layout.addWidget(self.plot_panel)

    def _build_plot(self):
        # ==================================================
        # Initial ECG figure
        # ==================================================
        self.ecg_ax.set_title("ECG Signal")
        self.ecg_ax.set_xlabel("Time (s)")
        self.ecg_ax.set_ylabel("Amplitude")
        self.ecg_ax.grid(True, alpha=0.3)

        self.ecg_figure.tight_layout()
        self.ecg_canvas.draw()

        # ==================================================
        # Initial Poincaré figure
        # ==================================================
        self.poincare_ax.set_title("Poincaré Plot")
        self.poincare_ax.set_xlabel(r"$RR_n$ (s)")
        self.poincare_ax.set_ylabel(r"$RR_{n+1}$ (s)")
        self.poincare_ax.grid(True, alpha=0.3)

        self.poincare_figure.tight_layout()
        self.poincare_canvas.draw()

        # ==================================================
        # Frequency-band power figure
        # ==================================================
        labels = [
            "VLF (0.0033-0.04 Hz)",
            "LF (0.04-0.15 Hz)",
            "HF (0.15-0.4 Hz)",
        ]

        # Initially empty bars
        initial_values = [
            0.0,
            0.0,
            0.0,
        ]

        self.frequency_bars = (
            self.frequency_ax.bar(
                labels,
                initial_values,
                width=0.6,
                color=[
                    "tab:blue",
                    "tab:red",
                    "tab:green",
                ],
                alpha=0.65,
            )
        )

        self.frequency_ax.set_title(
            "HRV Frequency-Band Power"
        )

        self.frequency_ax.set_xlabel(
            "Frequency band (Hz)"
        )

        self.frequency_ax.set_ylabel(
            r"Power (ms$^2$)"
        )

        self.frequency_ax.grid(
            axis="y",
            alpha=0.3,
        )

        self.frequency_ax.set_axisbelow(
            True
        )

        # Make long labels readable
        self.frequency_ax.tick_params(
            axis="x",
            labelrotation=20,
        )

        self.frequency_figure.tight_layout()
        self.frequency_canvas.draw()

    def load_ecg(self):
        # path, _ = QFileDialog.getOpenFileName(
        #     self, "Open ECG data", "", "TXT Files (*.txt);;All Files (*.*)"
        # )
        # if not path:
        #     return

        try:
            self.data = np.loadtxt(
                r"C:\Users\Rodrigo\Desktop\PhD\EmoLoop\notebooks\signal_samples\SampleECG.txt"
            )
            # self.data = np.loadtxt(path)
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

            self.avg_rr_var.setText(f"Avg RR: {hrv_time_domain.get('Avg RR', np.nan):.2f} ms")
            self.min_rr_var.setText(f"Min RR: {hrv_time_domain.get('Min RR', np.nan):.2f} ms")
            self.max_rr_var.setText(f"Max RR: {hrv_time_domain.get('Max RR', np.nan):.2f} ms")
            self.sd_rr_var.setText(f"SD RR: {hrv_time_domain.get('SD RR', np.nan):.2f} ms")
            self.sdnn_var.setText(f"SDNN: {hrv_time_domain.get('SDNN', np.nan):.2f} ms")
            self.rmssd_var.setText(f"RMSSD: {hrv_time_domain.get('RMSSD', np.nan):.2f} ms")
            self.nn50_var.setText(f"NN50: {hrv_time_domain.get('NN50', np.nan)}")
            self.pnn50_var.setText(f"pNN50: {hrv_time_domain.get('pNN50', np.nan):.2f} %")
            self.nn20_var.setText(f"NN20: {hrv_time_domain.get('NN20', np.nan)}")
            self.pnn20_var.setText(f"pNN20: {hrv_time_domain.get('pNN20', np.nan):.2f} %")

            self.vlf_power_var.setText(
                f"VLF Power: {hrv_freq_domain.get('VLF_Power', np.nan):.2f} ms²"
            )
            self.lf_power_var.setText(
                f"LF Power: {hrv_freq_domain.get('LF_Power', np.nan):.2f} ms²"
            )
            self.hf_power_var.setText(
                f"HF Power: {hrv_freq_domain.get('HF_Power', np.nan):.2f} ms²"
            )
            self.total_power_var.setText(
                f"Total Power: {hrv_freq_domain.get('Total_Power', np.nan):.2f} ms²"
            )
            self.lf_norm_var.setText(
                f"LF (nu): {hrv_freq_domain.get('LF_(nu)', np.nan):.2f}"
            )
            self.hf_norm_var.setText(
                f"HF (nu): {hrv_freq_domain.get('HF_(nu)', np.nan):.2f}"
            )
            self.lf_hf_var.setText(f"LF/HF: {hrv_freq_domain.get('LF/HF', np.nan):.2f}")

            self.std_var.setText(f"STD: {hrv_nonlinear.get('STD', np.nan):.2f} s")
            self.sdsd_var.setText(f"SDSD: {hrv_nonlinear.get('SDSD', np.nan):.2f} s")
            self.sd2_var.setText(f"SD2: {hrv_nonlinear.get('SD2', np.nan):.2f} ms")
            self.sd1_var.setText(f"SD1: {hrv_nonlinear.get('SD1', np.nan):.2f} ms")
            self.sd2_sd1_var.setText(
                f"SD2/SD1: {hrv_nonlinear.get('SD2/SD1', np.nan):.2f}"
            )

            self.rpeaks_var.setText(f"R-peaks: {len(r_peaks)}")
            self.status_label.setText("ECG processed successfully")

            self._update_plot(
                filtered, r_peaks, rr_intervals=self.ecg_data["rr_intervals"]
            )
            self._update_frequency_plot()

        except Exception as e:
            self.status_label.setText(f"Processing error: {e}")

    def _add_poincare_ellipse(self, rr_intervals):
        if rr_intervals is None:
            return

        rr = np.asarray(
            rr_intervals,
            dtype=float,
        )

        rr = rr[np.isfinite(rr)]

        if len(rr) < 2:
            return

        # Convert milliseconds to seconds if necessary.
        # Remove this conversion if your data is always in seconds.
        if np.nanmedian(rr) > 10:
            rr = rr / 1000.0

        # Consecutive RR intervals
        rr_n = rr[:-1]
        rr_next = rr[1:]

        # Retrieve the already-computed values
        nonlinear = self.hrv_data.get(
            "nonlinear_features",
            {},
        )

        sd1 = nonlinear.get("SD1", np.nan)
        sd2 = nonlinear.get("SD2", np.nan)

        if not np.isfinite(sd1) or not np.isfinite(sd2):
            return

        if sd1 <= 0 or sd2 <= 0:
            return

        # Convert SD1 and SD2 to seconds if they are in milliseconds.
        # This must match the units used by rr_n and rr_next.
        if sd1 > 10:
            sd1 = sd1 / 1000.0

        if sd2 > 10:
            sd2 = sd2 / 1000.0

        # Center of the ellipse
        mean_rr = np.mean(rr)

        # The SD2 axis is parallel to y = x.
        # The SD1 axis is perpendicular to y = x.
        ellipse = Ellipse(
            xy=(mean_rr, mean_rr),
            # Matplotlib expects full diameters
            width=2.0 * sd2,
            height=2.0 * sd1,
            angle=45.0,
            facecolor="tab:orange",
            edgecolor="tab:red",
            alpha=0.25,
            linewidth=2.0,
            label=(f"SD1={sd1:.4f} s, " f"SD2={sd2:.4f} s"),
        )

        self.poincare_ax.add_patch(ellipse)

        self.poincare_ax.plot(
            mean_rr,
            mean_rr,
            marker="+",
            markersize=10,
            markeredgewidth=2,
            color="black",
            label="Mean RR",
        )

        return rr_n, rr_next

    def _update_plot(
        self,
        signal,
        r_peaks=None,
        rr_intervals=None,
    ):
        if signal is None:
            return

        # ==================================================
        # Update ECG figure
        # ==================================================
        t = np.arange(len(signal)) / self.ecg.sampling_rate

        self.ecg_ax.clear()

        self.ecg_ax.plot(
            t,
            signal,
            lw=1,
            label="ECG",
        )

        if r_peaks is not None and len(r_peaks) > 0:
            r_peaks = np.asarray(
                r_peaks,
                dtype=int,
            )

            valid_peaks = (r_peaks >= 0) & (r_peaks < len(signal))

            r_peaks = r_peaks[valid_peaks]

            self.ecg_ax.plot(
                t[r_peaks],
                signal[r_peaks],
                "ro",
                ms=4,
                label="R-peaks",
            )

        self.ecg_ax.set_title("ECG Signal")
        self.ecg_ax.set_xlabel("Time (s)")
        self.ecg_ax.set_ylabel("Amplitude")
        self.ecg_ax.grid(True, alpha=0.3)
        self.ecg_ax.legend(loc="upper right")

        self.ecg_figure.tight_layout()
        self.ecg_canvas.draw()

        # ==================================================
        # Update Poincaré figure
        # ==================================================
        self.poincare_ax.clear()

        if rr_intervals is not None:
            rr = np.asarray(
                rr_intervals,
                dtype=float,
            )

            rr = rr[np.isfinite(rr)]

            if len(rr) > 0 and np.nanmedian(rr) > 10:
                rr = rr / 1000.0

            if len(rr) >= 2:
                rr_n = rr[:-1]
                rr_next = rr[1:]

                self.poincare_ax.scatter(
                    rr_n,
                    rr_next,
                    s=25,
                    alpha=0.7,
                    color="tab:blue",
                    label="RR intervals",
                )

                # Only rr_intervals is passed
                self._add_poincare_ellipse(rr_intervals)

                all_values = np.concatenate([rr_n, rr_next])

                lower = np.min(all_values)
                upper = np.max(all_values)
                data_range = upper - lower

                if data_range == 0:
                    data_range = 0.01

                margin = 0.15 * data_range
                plot_lower = lower - margin
                plot_upper = upper + margin

                self.poincare_ax.plot(
                    [plot_lower, plot_upper],
                    [plot_lower, plot_upper],
                    "k--",
                    linewidth=1,
                    label="Identity line",
                )

                self.poincare_ax.set_xlim(
                    plot_lower,
                    plot_upper,
                )

                self.poincare_ax.set_ylim(
                    plot_lower,
                    plot_upper,
                )

        self.poincare_ax.set_title("Poincaré Plot with Ellipse")

        self.poincare_ax.set_xlabel(r"$RR_n$ (s)")

        self.poincare_ax.set_ylabel(r"$RR_{n+1}$ (s)")

        self.poincare_ax.grid(True, alpha=0.3)

        handles, labels = self.poincare_ax.get_legend_handles_labels()

        if handles:
            self.poincare_ax.legend(loc="upper left")

        self.poincare_figure.tight_layout()
        self.poincare_canvas.draw()

    def _update_frequency_plot(self):
        self.frequency_ax.clear()

        frequency_features = self.hrv_data.get(
            "frequency_features",
            {},
        )

        labels = [
            "VLF (0.0033-0.04 Hz)",
            "LF (0.04-0.15 Hz)",
            "HF (0.15-0.4 Hz)",
        ]

        values = np.asarray(
            [
                frequency_features.get(
                    "VLF_Power",
                    np.nan,
                ),
                frequency_features.get(
                    "LF_Power",
                    np.nan,
                ),
                frequency_features.get(
                    "HF_Power",
                    np.nan,
                ),
            ],
            dtype=float,
        )

        plot_values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        bars = self.frequency_ax.bar(
            labels,
            plot_values,
            width=0.6,
            color=[
                "tab:purple",
                "tab:orange",
                "tab:green",
            ],
            alpha=0.85,
        )

        self.frequency_ax.bar_label(
            bars,
            labels=[
                "nan" if not np.isfinite(value) else f"{value:.4f}" for value in values
            ],
            padding=3,
        )

        self.frequency_ax.set_title("HRV Frequency-Band Power")

        self.frequency_ax.set_xlabel("Frequency band (Hz)")

        self.frequency_ax.set_ylabel("Power (ms²)")

        self.frequency_ax.grid(
            axis="y",
            alpha=0.3,
        )

        self.frequency_ax.set_axisbelow(True)

        self.frequency_figure.tight_layout()
        self.frequency_canvas.draw()
