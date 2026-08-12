from abc import abstractmethod
from typing import Tuple, Dict, Any

import numpy as np
import scipy
from sensors.EDA.base import EDA_base
from sensors.EDA.config import EDA_Config
import neurokit2 as nk


class EDAAlgorithm(EDA_base):
    def __init__(self):
        self.config = EDA_Config()  # Initialize with default config

    def convertEDA(self, signal: np.ndarray) -> np.ndarray:
        VCC = self.config.VCC
        resolution = self.config.resolution
        signal_microS = (signal / pow(2, resolution)) * VCC / 0.12
        signal_S = signal_microS * pow(10, -6)

        return signal_S

    def filter(self, signal: np.ndarray) -> np.ndarray:
        sos = scipy.signal.butter(
            self.config.lowpass_butter_order,
            [self.config.lowpass_freq, self.config.highpass_freq],
            btype=self.config.filter_type,
            fs=self.config.sampling_rate,
            output="sos",
        )

        return scipy.signal.sosfiltfilt(sos, signal)

    def get_componentsEDA(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eda_components = nk.eda_phasic(
            signal,
            sampling_rate=self.config.sampling_rate,
            method=self.config.get_component_method,
        )

        eda_phasic = eda_components["EDA_Phasic"].values
        eda_tonic = eda_components["EDA_Tonic"].values

        return eda_phasic, eda_tonic

    def getSCRfeatures(self, phasic_component: np.ndarray) -> Dict[str, Any]:
        signals, peaks = nk.eda_peaks(
            phasic_component,
            sampling_rate=self.config.sampling_rate,
            method=self.config.method,
        )

        SCR_Amplitude = peaks.get("SCR_Amplitude", None)
        SCR_RiseTime = peaks.get("SCR_RiseTime", None)
        SCR_RecoveryTime = peaks.get("SCR_RecoveryTime", None)

        # Convert to np.nan if None or empty
        def _to_array_or_nan(x):
            if x is None:
                return np.nan
            x = np.asarray(x)
            return x if x.size > 0 else np.nan

        SCR_Amplitude = _to_array_or_nan(SCR_Amplitude)
        SCR_RiseTime = _to_array_or_nan(SCR_RiseTime)
        SCR_RecoveryTime = _to_array_or_nan(SCR_RecoveryTime)

        return {
            "SCR_Amplitude": SCR_Amplitude,
            "SCR_RiseTime": SCR_RiseTime,
            "SCR_RecoveryTime": SCR_RecoveryTime,
            "SCR_Avg_Amplitude": (
                np.nanmean(SCR_Amplitude)
                if not np.isnan(SCR_Amplitude).all()
                else np.nan
            ),
            "SCR_Avg_RiseTime": (
                np.nanmean(SCR_RiseTime) if not np.isnan(SCR_RiseTime).all() else np.nan
            ),
            "SCR_Avg_RecoveryTime": (
                np.nanmean(SCR_RecoveryTime)
                if not np.isnan(SCR_RecoveryTime).all()
                else np.nan
            ),
            "SCR_STD_Amplitude": (
                np.nanstd(SCR_Amplitude)
                if not np.isnan(SCR_Amplitude).all()
                else np.nan
            ),
            "SCR_STD_RiseTime": (
                np.nanstd(SCR_RiseTime) if not np.isnan(SCR_RiseTime).all() else np.nan
            ),
            "SCR_STD_RecoveryTime": (
                np.nanstd(SCR_RecoveryTime)
                if not np.isnan(SCR_RecoveryTime).all()
                else np.nan
            ),
            "SCR_Max_Amplitude": (
                np.nanmax(SCR_Amplitude)
                if not np.isnan(SCR_Amplitude).all()
                else np.nan
            ),
            "SCR_Max_RiseTime": (
                np.nanmax(SCR_RiseTime) if not np.isnan(SCR_RiseTime).all() else np.nan
            ),
            "SCR_Max_RecoveryTime": (
                np.nanmax(SCR_RecoveryTime)
                if not np.isnan(SCR_RecoveryTime).all()
                else np.nan
            ),
            "SCR_Min_Amplitude": (
                np.nanmin(SCR_Amplitude)
                if not np.isnan(SCR_Amplitude).all()
                else np.nan
            ),
            "SCR_Min_RiseTime": (
                np.nanmin(SCR_RiseTime) if not np.isnan(SCR_RiseTime).all() else np.nan
            ),
            "SCR_Min_RecoveryTime": (
                np.nanmin(SCR_RecoveryTime)
                if not np.isnan(SCR_RecoveryTime).all()
                else np.nan
            ),
        }

    def getSCLfeatures(self, tonic_component: np.ndarray) -> Dict[str, Any]:
        SCL_AVG = np.nanmean(tonic_component) if tonic_component.size > 0 else np.nan
        SCL_STD = np.nanstd(tonic_component) if tonic_component.size > 0 else np.nan
        SCL_MAX = np.nanmax(tonic_component) if tonic_component.size > 0 else np.nan
        SCL_MIN = np.nanmin(tonic_component) if tonic_component.size > 0 else np.nan

        return {
            "SCL_AVG": SCL_AVG,
            "SCL_STD": SCL_STD,
            "SCL_MAX": SCL_MAX,
            "SCL_MIN": SCL_MIN,
        }

    def frequencyAnalysis(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        factor = 1000
        fs_new = self.config.sampling_rate / factor

        downsampled1 = scipy.signal.decimate(signal, q=10, n=8)
        downsampled2 = scipy.signal.decimate(downsampled1, q=10, n=8)
        downsampled3 = scipy.signal.decimate(downsampled2, q=10, n=8)

        sos = scipy.signal.butter(
            8,
            0.01,
            btype="highpass",
            fs=self.config.sampling_rate,
            output="sos",
        )

        filtered_signal = scipy.signal.sosfiltfilt(sos, downsampled3)

        freqs, power = scipy.signal.welch(
            filtered_signal,
            fs=fs_new,
            nperseg=self.config.nperseg,
            window=self.config.window,
            noverlap=64,
        )

        return freqs, power
