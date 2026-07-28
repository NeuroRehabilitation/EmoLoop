from typing import Dict, Any

from sensors.HRV.base import HRV_base
from sensors.HRV.config import HRV_Config

import numpy as np
import scipy as sc


class HRVAlgorithm(HRV_base):
    def __init__(self, config: HRV_Config):
        self.config = config

    def remove_ectopy_beats(self, rr_intervals: np.ndarray) -> np.ndarray:
        if rr_intervals.size < 2:
            return rr_intervals.copy()

        keep = np.ones(len(rr_intervals), dtype=bool)

        for i in range(1, len(rr_intervals)):
            prev = rr_intervals[i - 1]
            if prev <= 0:
                continue
            if abs(rr_intervals[i] - prev) / prev > self.config.ectopy_threshold:
                keep[i] = False
                if i + 1 < len(rr_intervals):
                    keep[i + 1] = False

        return rr_intervals[keep]

    def rr_intervals(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        if rr_intervals.size == 0:
            return {
                "Avg RR": np.nan,
                "Min RR": np.nan,
                "Max RR": np.nan,
                "SD RR": np.nan,
            }

        return {
            "Avg RR": float(np.nanmean(rr_intervals)),
            "Min RR": float(np.nanmin(rr_intervals)),
            "Max RR": float(np.nanmax(rr_intervals)),
            "SD RR": float(np.nanstd(rr_intervals)),
        }

    def heart_rate(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        if rr_intervals.size == 0:
            return {
                "Avg HR": np.nan,
                "Min HR": np.nan,
                "Max HR": np.nan,
                "SD HR": np.nan,
            }

        heart_rate = 60 / rr_intervals

        return {
            "Avg HR": float(np.nanmean(heart_rate)),
            "Min HR": float(np.nanmin(heart_rate)),
            "Max HR": float(np.nanmax(heart_rate)),
            "SD HR": float(np.nanstd(heart_rate)),
        }

    def time_domain_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        if rr_intervals.size == 0:
            return {
                "SDNN": np.nan,
                "RMSSD": np.nan,
                "NN50": np.nan,
                "pNN50": np.nan,
                "NN20": np.nan,
                "pNN20": np.nan,
            }

        nn50 = self.NN50(rr_intervals)
        nn20 = self.NN20(rr_intervals)

        return {
            "SDNN": self.SDNN(rr_intervals),
            "RMSSD": self.RMSSD(rr_intervals),
            "NN50": nn50,
            "pNN50": self.pNN50(nn50, rr_intervals),
            "NN20": nn20,
            "pNN20": self.pNN20(nn20, rr_intervals),
        }

    @staticmethod
    def SDNN(rr_intervals: np.ndarray) -> float:
        if rr_intervals.size == 0:
            return np.nan

        return float(round(np.std(rr_intervals) * 1000, 4))

    @staticmethod
    def RMSSD(rr_intervals: np.ndarray) -> float:
        if rr_intervals.size < 2:
            return np.nan

        return float(
            round(
                np.sqrt(np.sum((np.diff(rr_intervals)) ** 2) / (len(rr_intervals) - 1))
                * 1000,
                4,
            )
        )

    @staticmethod
    def NN50(rr_intervals: np.ndarray) -> int:
        if rr_intervals.size < 2:
            return np.nan

        rr_interval_diff = np.diff(rr_intervals)
        rr_interval_abs = np.abs(rr_interval_diff)

        return sum(1 for i in rr_interval_abs if i > 0.05)

    @staticmethod
    def pNN50(nn50: float, rr_intervals: np.ndarray) -> float:
        if np.isnan(nn50):
            return np.nan
        return float(round((float(nn50) / len(rr_intervals)) * 100, 4))

    @staticmethod
    def NN20(rr_intervals: np.ndarray) -> int:
        if rr_intervals.size < 2:
            return np.nan

        rr_interval_diff = np.diff(rr_intervals)
        rr_interval_abs = np.abs(rr_interval_diff)

        return sum(1 for i in rr_interval_abs if i > 0.02)

    @staticmethod
    def pNN20(nn20: float, rr_intervals: np.ndarray) -> float:
        if np.isnan(nn20):
            return np.nan
        return float(round((float(nn20) / len(rr_intervals)) * 100, 4))

    @staticmethod
    def frequencyAnalysis(self,rr_intervals: np.ndarray, rr_time:np.ndarray):
        if len(rr_time) < 4 or len(rr_intervals) < 4:
            return np.array([]), np.array([])

        t_new = np.arange(rr_time[0], rr_intervals[-1], 1.0 / self.config.interpolation_rate)

        tck = sc.interpolate.splrep(rr_time, rr_intervals, s=0)
        rr_even = sc.interpolate.splev(t_new, tck)
        rr_even = rr_even - np.mean(rr_even)

        freq_axis, power_axis = sc.signal.welch(
            rr_even,
            fs=self.config.interpolation_rate,
            window=sc.signal.get_window(self.config.window, min(len(rr_even), 1000)),
            nperseg=min(len(rr_even), 1000),
        )

        mask = freq_axis < 0.5
        return freq_axis[mask], power_axis[mask]


    def frequency_domain_features(self,freqs: np.ndarray,power: np.ndarray) -> Dict[str, Any]:
        """
        Compute HRV band powers and normalized units.

        :param freqs: frequency axis
        :param power: power spectral density
        :return: dictionary of HRV features
        """

        def band_power(fmin, fmax):
            idx = (freqs >= fmin) & (freqs < fmax)
            return sc.integrate.trapz(power[idx], freqs[idx]) if np.any(idx) else np.nan

        vlf = band_power(0.0033, 0.04)
        lf = band_power(0.04, 0.15)
        hf = band_power(0.15, 0.4)
        total_power = band_power(0.0033, 0.4)

        if np.isfinite(total_power) and (total_power - vlf) > 0:
            lf_norm = lf / (total_power - vlf) * 100
            hf_norm = hf / (total_power - vlf) * 100
        else:
            lf_norm = np.nan
            hf_norm = np.nan

        ratio = lf_norm / hf_norm if np.isfinite(lf_norm) and np.isfinite(hf_norm) and hf_norm > 0 else np.nan

        return {
            "VLF_Power": [vlf],
            "LF_Power": [lf],
            "HF_Power": [hf],
            "Total_Power": [total_power],
            "LF_(nu)": [lf_norm],
            "HF_(nu)": [hf_norm],
            "VLF/HF": [ratio],
        }

