from typing import Dict, Any

from sensors.HRV.base import HRV_base
from sensors.HRV.config import HRV_Config

import numpy as np

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

    def rr_intervals(self,rr_intervals: np.ndarray) ->Dict[str,float]:
        if rr_intervals.size == 0:
            return {"Avg RR": np.nan, "Min RR": np.nan, "Max RR": np.nan, "SD RR": np.nan}

        return {"Avg RR": float(np.nanmean(rr_intervals)),
                "Min RR": float(np.nanmin(rr_intervals)),
                "Max RR": float(np.nanmax(rr_intervals)),
                "SD RR": float(np.nanstd(rr_intervals))}

    def heart_rate(self, rr_intervals: np.ndarray) -> Dict[str,float]:
        if rr_intervals.size == 0:
            return {"Avg HR": np.nan, "Min HR": np.nan, "Max HR": np.nan, "SD HR": np.nan}

        heart_rate = 60/rr_intervals

        return {"Avg HR": float(np.nanmean(heart_rate)),
                "Min HR": float(np.nanmin(heart_rate)),
                "Max HR": float(np.nanmax(heart_rate)),
                "SD HR": float(np.nanstd(heart_rate))}

    def time_domain_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        if rr_intervals.size == 0:
            return {"SDNN": np.nan, "RMSSD": np.nan, "NN50": np.nan, "pNN50": np.nan, "NN20": np.nan, "pNN20": np.nan}

        nn50 = self.NN50(rr_intervals)
        nn20 = self.NN20(rr_intervals)

        return {
            "SDNN": self.SDNN(rr_intervals),
            "RMSSD": self.RMSSD(rr_intervals),
            "NN50": nn50,
            "pNN50": self.pNN50(nn50, rr_intervals),
            "NN20": nn20,
            "pNN20": self.pNN20(nn20, rr_intervals)
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

        return float(round(np.sqrt(np.sum((np.diff(rr_intervals)) ** 2) / (len(rr_intervals) - 1))* 1000,4,))

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

