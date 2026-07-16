import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Dashboard:
    def __init__(self, root, ecg):
        self.root = root
        self.ecg = ecg
        self.root.title("EmoLoop ECG Dashboard")
        self.root.geometry("1200x800")

        self.signal = None
        self.time = None
        self.results = None

        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Load ECG Data", command=self.load_ecg).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(top, text="Process R Peaks", command=self.process_ecg).pack(
            side=tk.LEFT, padx=5
        )

        self.mean_hr_var = tk.StringVar(value="Mean HR: --")
        self.hr_std_var = tk.StringVar(value="HR Std: --")
        self.hrv_var = tk.StringVar(value="HRV: --")
        self.rpeaks_var = tk.StringVar(value="R-peaks: --")
        self.status_var = tk.StringVar(value="Ready")

        metrics = ttk.Frame(self.root, padding=10)
        metrics.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(metrics, textvariable=self.mean_hr_var).pack(anchor="w", pady=2)
        ttk.Label(metrics, textvariable=self.hr_std_var).pack(anchor="w", pady=2)
        ttk.Label(metrics, textvariable=self.hrv_var).pack(anchor="w", pady=2)
        ttk.Label(metrics, textvariable=self.rpeaks_var).pack(anchor="w", pady=2)

        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    def _build_plot(self):
        plot_frame = ttk.Frame(self.root, padding=10)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(9, 5), dpi=100)
        self.ax.set_title("ECG Signal")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)

        (self.ecg_line,) = self.ax.plot([], [], lw=1, label="ECG")
        (self.peak_line,) = self.ax.plot([], [], "ro", label="R-peaks")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_ecg(self):
        path = filedialog.askopenfilename(
            title="Open ECG data",
            filetypes=[("TXT Files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        print(path)
        try:
            self.data = np.loadtxt(path)
            self.signal = self.data[:, 2]
            self.time = np.arange(len(self.signal)) / self.ecg.sampling_rate
            self._update_plot(self.signal, None)
            self.status_var.set(f"Loaded {len(self.signal)} samples")
        except Exception as e:
            self.status_var.set(f"Load error: {e}")

    def process_ecg(self):
        if self.signal is None:
            self.status_var.set("No ECG data loaded")
            return

        try:
            self.results = self.ecg.process(self.signal)

            filtered = self.results["filtered_signal"]
            r_peaks = self.results["r_peaks"]
            hr = self.results["heart_rate"]

            self.mean_hr_var.set(f"Mean HR: {np.mean(hr["HR"]):.2f} bpm")
            # self.hr_std_var.set(f"HR Std: {hr['hr_std']:.2f} bpm")
            # self.hrv_var.set(f"HRV: {hr['hrv']:.4f} s")
            self.rpeaks_var.set(f"R-peaks: {len(r_peaks)}")
            self.status_var.set("ECG processed successfully")

            self._update_plot(filtered, r_peaks)

        except Exception as e:
            self.status_var.set(f"Processing error: {e}")

    def _update_plot(self, signal, r_peaks):
        if signal is None:
            return

        t = np.arange(len(signal)) / self.ecg.sampling_rate
        self.ecg_line.set_data(t, signal)

        if r_peaks is not None and len(r_peaks) > 0:
            self.peak_line.set_data(t[r_peaks], signal[r_peaks])
        else:
            self.peak_line.set_data([], [])

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
