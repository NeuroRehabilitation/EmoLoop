"""
Module: sensors.EDA.algorithms
Description: Implementation of EDA (Electrodermal Activity) signal processing algorithms.

This module provides the concrete implementation of the EDA_base abstract class,
utilizing the neurokit2 library and scipy for signal processing tasks including
filtering, component decomposition, feature extraction, and frequency analysis.
"""

from typing import Tuple, Dict, Any

import numpy as np
import scipy
from sensors.EDA.config import EDA_Config
from sensors.EDA.base import EDA_base
import neurokit2 as nk


class EDAAlgorithm(EDA_base):
    """
    Concrete implementation of EDA signal processing algorithms.

    This class provides a complete EDA processing pipeline using neurokit2 and scipy
    for signal filtering, component decomposition, and feature extraction. It implements
    all abstract methods from the EDA_base class.

    Attributes
    ----------
    config : EDA_Config
        Configuration object containing all parameters for signal processing,
        including sampling rate, filter parameters, and feature extraction settings.

    Methods
    -------
    convertEDA(signal)
        Convert raw digital EDA values to physical units (Siemens).
    filter(signal)
        Apply butterworth bandpass filtering to the EDA signal.
    get_componentsEDA(signal)
        Decompose EDA signal into phasic and tonic components.
    getSCRfeatures(phasic_component)
        Extract skin conductance response (SCR) features from phasic component.
    getSCLfeatures(tonic_component)
        Extract skin conductance level (SCL) features from tonic component.
    frequencyAnalysis(signal)
        Perform power spectral density analysis on the EDA signal.
    frequency_domain_features(freqs, power)
        Extract frequency-domain features from power spectral density.
    get_config()
        Return the current configuration parameters.
    """

    def __init__(self):
        """
        Initialize the EDAAlgorithm with default configuration.

        Creates a new instance with default EDA_Config parameters. These parameters
        control all aspects of signal processing including filter cutoff frequencies,
        decomposition methods, and feature extraction settings.
        """
        self.config = EDA_Config()  # Initialize with default config

    def convertEDA(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert raw digital EDA values to physical units (Siemens).

        This method performs voltage-to-conductance conversion from the raw
        digital values acquired by the sensor, accounting for the ADC resolution,
        reference voltage, and the sensor's 0.12 ohm resistor.

        Parameters
        ----------
        signal : np.ndarray
            Raw digital EDA signal values from the sensor.
            Expected shape: (n_samples,)

        Returns
        -------
        np.ndarray
            EDA signal converted to Siemens (S), which represents electrical conductivity.
            Same shape as input signal.

        Notes
        -----
        Conversion formula:
        signal_microS = (signal / 2^resolution) * VCC / 0.12
        signal_S = signal_microS * 10^-6

        Uses configuration parameters:
        - VCC: Reference voltage (typically 3.3V or 5V)
        - resolution: ADC resolution in bits (typically 12-16 bits)
        """
        VCC = self.config.VCC
        resolution = self.config.resolution
        signal_microS = (signal / pow(2, resolution)) * VCC / 0.12
        signal_S = np.asarray(signal_microS * pow(10, -6))

        return signal_S

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply butterworth bandpass filtering to the raw EDA signal.

        This method performs IIR filtering using a butterworth filter design
        in second-order sections (SOS) format for numerical stability, with
        forward-backward filtering (filtfilt) to achieve zero phase distortion.

        Parameters
        ----------
        signal : np.ndarray
            Raw EDA signal to be filtered.
            Expected shape: (n_samples,)

        Returns
        -------
        np.ndarray
            Filtered EDA signal with same shape as input.
            Phase distortion is eliminated due to the use of filtfilt.

        Notes
        -----
        Uses configuration parameters:
        - lowpass_butter_order: Filter order (typically 4-5)
        - lowpass_freq: High-pass cutoff frequency (Hz)
        - highpass_freq: Low-pass cutoff frequency (Hz)
        - filter_type: Filter type, typically 'bandpass'
        - sampling_rate: Sampling rate of the signal (Hz)

        The SOS (second-order sections) format is used for better numerical
        stability compared to direct IIR coefficients, especially for higher
        order filters.
        """
        sos = scipy.signal.butter(
            self.config.lowpass_butter_order,
            self.config.highpass_freq,
            btype=self.config.filter_type,
            fs=self.config.sampling_rate,
            output="sos",
        )

        return scipy.signal.sosfiltfilt(sos, signal)

    def get_componentsEDA(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose the EDA signal into phasic and tonic components.

        This method uses neurokit2's cvxEDA (or other configurable) decomposition
        algorithm to separate the EDA signal into:
        - Phasic: Rapid, discrete skin conductance responses (SCRs)
        - Tonic: Slow, continuous baseline skin conductance level (SCL)

        Parameters
        ----------
        signal : np.ndarray
            Filtered EDA signal to be decomposed.
            Expected shape: (n_samples,)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - eda_phasic (np.ndarray): Phasic component, shape (n_samples,)
            - eda_tonic (np.ndarray): Tonic component, shape (n_samples,)
            Both components sum approximately to the input signal.

        Notes
        -----
        Uses configuration parameters:
        - sampling_rate: Sampling rate of the signal (Hz)
        - get_component_method: Decomposition method (e.g., 'cvxeda', 'highpass')

        The decomposition assumes that EDA = Tonic + Phasic, which is the
        standard psychophysiological model of EDA signal structure.

        References
        ----------
        Decomposition is performed using neurokit2.eda_phasic() function.
        """
        eda_components = nk.eda_phasic(
            signal,
            sampling_rate=self.config.sampling_rate,
            method=self.config.get_component_method,
        )

        eda_phasic = eda_components["EDA_Phasic"].values
        eda_tonic = eda_components["EDA_Tonic"].values

        return eda_phasic, eda_tonic

    def getSCRfeatures(self, phasic_component: np.ndarray) -> Dict[str, Any]:
        """
        Extract skin conductance response (SCR) features from the phasic component.

        This method identifies SCR peaks in the phasic component and computes
        various temporal and amplitude-based features including rise time,
        recovery time, and amplitude statistics.

        Parameters
        ----------
        phasic_component : np.ndarray
            Phasic component of the EDA signal.
            Expected shape: (n_samples,)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the following SCR features:

            Raw measurements (arrays or np.nan):
            - 'SCR_Amplitude': Array of individual SCR peak amplitudes (µS)
            - 'SCR_RiseTime': Array of individual SCR rise times (seconds)
            - 'SCR_RecoveryTime': Array of individual SCR recovery times (seconds)

            Aggregate statistics:
            - 'SCR_Avg_Amplitude': Mean of SCR amplitudes (µS)
            - 'SCR_Avg_RiseTime': Mean of SCR rise times (seconds)
            - 'SCR_Avg_RecoveryTime': Mean of SCR recovery times (seconds)
            - 'SCR_STD_Amplitude': Standard deviation of SCR amplitudes
            - 'SCR_STD_RiseTime': Standard deviation of SCR rise times
            - 'SCR_STD_RecoveryTime': Standard deviation of SCR recovery times
            - 'SCR_Max_Amplitude': Maximum SCR amplitude
            - 'SCR_Max_RiseTime': Maximum SCR rise time
            - 'SCR_Max_RecoveryTime': Maximum SCR recovery time
            - 'SCR_Min_Amplitude': Minimum SCR amplitude
            - 'SCR_Min_RiseTime': Minimum SCR rise time
            - 'SCR_Min_RecoveryTime': Minimum SCR recovery time

        Notes
        -----
        Uses configuration parameters:
        - sampling_rate: Sampling rate of the signal (Hz)
        - method: Peak detection method (e.g., 'neurokit', 'vanhalem')

        Returns np.nan for features when no valid SCRs are detected. All
        measurements are robust to missing or invalid data using nanmean,
        nanstd, nanmin, nanmax functions.

        References
        ----------
        Peak detection is performed using neurokit2.eda_peaks() function.
        """
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
            """Convert None or empty array to np.nan."""
            if x is None:
                return np.nan
            x = np.asarray(x)
            return x if x.size > 0 else np.nan

        SCR_Amplitude = _to_array_or_nan(SCR_Amplitude)
        SCR_RiseTime = _to_array_or_nan(SCR_RiseTime)
        SCR_RecoveryTime = _to_array_or_nan(SCR_RecoveryTime)

        return {
            "SCR_Amplitude": (
                SCR_Amplitude if not np.isnan(SCR_Amplitude).all() else np.nan
            ),
            "SCR_RiseTime": (
                SCR_RiseTime if not np.isnan(SCR_RiseTime).all() else np.nan
            ),
            "SCR_RecoveryTime": (
                SCR_RecoveryTime
                if not np.isnan(SCR_RecoveryTime).all()
                else np.nan
            ),
            "SCR_Avg_Amplitude": (
                float(np.nanmean(SCR_Amplitude))
                if not np.isnan(SCR_Amplitude).all()
                else np.nan
            ),
            "SCR_Avg_RiseTime": (
                float(np.nanmean(SCR_RiseTime))
                if not np.isnan(SCR_RiseTime).all()
                else np.nan
            ),
            "SCR_Avg_RecoveryTime": (
                float(np.nanmean(SCR_RecoveryTime))
                if not np.isnan(SCR_RecoveryTime).all()
                else np.nan
            ),
            "SCR_STD_Amplitude": (
                float(np.nanstd(SCR_Amplitude))
                if not np.isnan(SCR_Amplitude).all()
                else np.nan
            ),
            "SCR_STD_RiseTime": (
                float(np.nanstd(SCR_RiseTime))
                if not np.isnan(SCR_RiseTime).all()
                else np.nan
            ),
            "SCR_STD_RecoveryTime": (
                float(np.nanstd(SCR_RecoveryTime))
                if not np.isnan(SCR_RecoveryTime).all()
                else np.nan
            ),
            "SCR_Max_Amplitude": (
                float(np.nanmax(SCR_Amplitude))
                if not np.isnan(SCR_Amplitude).all()
                else np.nan
            ),
            "SCR_Max_RiseTime": (
                float(np.nanmax(SCR_RiseTime))
                if not np.isnan(SCR_RiseTime).all()
                else np.nan
            ),
            "SCR_Max_RecoveryTime": (
                float(np.nanmax(SCR_RecoveryTime))
                if not np.isnan(SCR_RecoveryTime).all()
                else np.nan
            ),
            "SCR_Min_Amplitude": (
                float(np.nanmin(SCR_Amplitude))
                if not np.isnan(SCR_Amplitude).all()
                else np.nan
            ),
            "SCR_Min_RiseTime": (
                float(np.nanmin(SCR_RiseTime))
                if not np.isnan(SCR_RiseTime).all()
                else np.nan
            ),
            "SCR_Min_RecoveryTime": (
                float(np.nanmin(SCR_RecoveryTime))
                if not np.isnan(SCR_RecoveryTime).all()
                else np.nan
            ),
        }

    def getSCLfeatures(self, tonic_component: np.ndarray) -> Dict[str, Any]:
        """
        Extract skin conductance level (SCL) features from the tonic component.

        This method computes basic descriptive statistics of the tonic (baseline)
        component of the EDA signal, representing sustained arousal/stress levels.

        Parameters
        ----------
        tonic_component : np.ndarray
            Tonic component of the EDA signal.
            Expected shape: (n_samples,)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the following SCL features (all in µS):

            - 'SCL_AVG': Mean skin conductance level (baseline)
            - 'SCL_STD': Standard deviation of skin conductance level
            - 'SCL_MAX': Maximum skin conductance level
            - 'SCL_MIN': Minimum skin conductance level

        Notes
        -----
        All statistics are computed using nan-safe functions (nanmean, nanstd,
        nanmax, nanmin) to handle potential missing or invalid values.

        Returns np.nan for each statistic if the tonic component is empty or
        contains only NaN values.

        The SCL represents the slowly-changing baseline of the EDA signal and
        is associated with sustained emotional states, fatigue, and overall
        arousal level.
        """
        SCL_AVG = float(
            np.nanmean(tonic_component) if tonic_component.size > 0 else np.nan
        )
        SCL_STD = float(
            np.nanstd(tonic_component) if tonic_component.size > 0 else np.nan
        )
        SCL_MAX = float(
            np.nanmax(tonic_component) if tonic_component.size > 0 else np.nan
        )
        SCL_MIN = float(
            np.nanmin(tonic_component) if tonic_component.size > 0 else np.nan
        )

        return {
            "SCL_AVG": SCL_AVG,
            "SCL_STD": SCL_STD,
            "SCL_MAX": SCL_MAX,
            "SCL_MIN": SCL_MIN,
        }

    def frequencyAnalysis(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform power spectral density (PSD) analysis on the EDA signal.

        This method computes the power spectral density using Welch's method after
        aggressive downsampling and high-pass filtering to focus on physiologically
        relevant frequency content and improve spectral resolution.

        Parameters
        ----------
        signal : np.ndarray
            Filtered EDA signal for frequency analysis.
            Expected shape: (n_samples,)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - freqs (np.ndarray): Frequency vector in Hz, shape (n_freqs,)
            - power (np.ndarray): Power spectral density, shape (n_freqs,)

        Notes
        -----
        Processing pipeline:
        1. Downsample by 10 three times (total factor of 1000), reducing from
           original sampling rate to fs_new = sampling_rate / 1000
        2. Apply high-pass butterworth filter at 0.01 Hz to remove low-frequency drift
        3. Compute Welch's PSD with configurable window and segment length

        Uses configuration parameters:
        - sampling_rate: Original sampling rate (Hz)
        - nperseg: Welch segment length (samples)
        - window: Window function for Welch's method (e.g., 'hann', 'hamming')

        The aggressive downsampling is necessary due to the low frequency content
        of EDA signals (typically < 0.5 Hz), which would otherwise require very
        large segment lengths for adequate frequency resolution.

        References
        ----------
        Uses scipy.signal.welch() with noverlap=64 samples for 50% overlap.
        """
        factor = 1000
        fs_new = self.config.sampling_rate / factor

        # Downsample in three stages to reduce computational load
        downsampled1 = scipy.signal.decimate(signal, q=10, n=8)
        downsampled2 = scipy.signal.decimate(downsampled1, q=10, n=8)
        downsampled3 = scipy.signal.decimate(downsampled2, q=10, n=8)

        # Apply high-pass filter to remove low-frequency drift
        sos = scipy.signal.butter(
            8,
            0.01,
            btype="highpass",
            fs=self.config.sampling_rate,
            output="sos",
        )

        filtered_signal = scipy.signal.sosfiltfilt(sos, downsampled3)

        # Compute power spectral density using Welch's method
        freqs, power = scipy.signal.welch(
            filtered_signal,
            fs=fs_new,
            nperseg=self.config.nperseg,
            window=self.config.window,
            noverlap=64,
        )

        return freqs, power

    def frequency_domain_features(
        self, freqs: np.ndarray, power: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract frequency-domain features from the power spectral density.

        This method computes features in predefined frequency bands (VLF, LF, HF)
        and derives normalized measures and ratios commonly used in autonomic
        nervous system analysis.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency vector from PSD analysis (Hz).
            Expected shape: (n_freqs,)
        power : np.ndarray
            Power spectral density values.
            Expected shape: (n_freqs,), must correspond to freqs

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the following frequency-domain features:

            Absolute power (µS²/Hz-like units, scaled by 10^6):
            - 'VLF_Power': Very low frequency power (vlf_lfreq to vlf_hfreq Hz)
            - 'LF_Power': Low frequency power (lf_lfreq to lf_hfreq Hz)
            - 'HF_Power': High frequency power (hf_lfreq to hf_hfreq Hz)
            - 'Total_Power': Total power (VLF to HF)

            Normalized units (%, excluding VLF):
            - 'LF_(nu)': LF in normalized units (%)
            - 'HF_(nu)': HF in normalized units (%)

            Ratios:
            - 'LF/HF': Ratio of LF to HF power (sympatho-vagal balance indicator)

        Notes
        -----
        Uses configuration parameters:
        - vlf_lfreq, vlf_hfreq: VLF frequency band boundaries (Hz)
        - lf_lfreq, lf_hfreq: LF frequency band boundaries (Hz)
        - hf_lfreq, hf_hfreq: HF frequency band boundaries (Hz)

        Normalization excludes VLF to emphasize the balance between LF
        (associated with sympathetic activity) and HF (associated with
        parasympathetic activity).

        Returns np.nan for computed features when insufficient valid data
        is available (e.g., when total power is not finite or HF power is zero).

        References
        ----------
        Band power integration uses scipy.integrate.trapezoid() for numerical
        integration of the PSD curve using the trapezoidal rule.
        """

        def band_power(fmin, fmax):
            """
            Integrate PSD between fmin and fmax using trapezoidal rule.

            Parameters
            ----------
            fmin : float
                Minimum frequency boundary (Hz)
            fmax : float
                Maximum frequency boundary (Hz)

            Returns
            -------
            float
                Integrated power in the frequency band, or np.nan if no
                frequencies fall within the band.
            """
            idx = (freqs >= fmin) & (freqs < fmax)
            return (
                scipy.integrate.trapezoid(power[idx], freqs[idx])
                if np.any(idx)
                else np.nan
            )

        # Compute band powers. Scale by 10^6 for convention (µs^2/Hz-like units).
        vlf = float(
            band_power(self.config.vlf_lfreq, self.config.vlf_hfreq) * pow(10, 6)
        )
        lf = float(band_power(self.config.lf_lfreq, self.config.lf_hfreq) * pow(10, 6))
        hf = float(band_power(self.config.hf_lfreq, self.config.hf_hfreq) * pow(10, 6))
        total_power = float(
            band_power(self.config.vlf_lfreq, self.config.hf_hfreq) * pow(10, 6)
        )

        # Compute normalized units for LF and HF (exclude VLF from denominator).
        # This normalization emphasizes the balance between sympathetic and
        # parasympathetic nervous system activity.
        if np.isfinite(total_power) and (total_power - vlf) > 0:
            lf_norm = lf / (total_power - vlf) * 100
            hf_norm = hf / (total_power - vlf) * 100
        else:
            lf_norm = np.nan
            hf_norm = np.nan

        # LF/HF ratio in normalized units (guard against division by zero).
        # This ratio is often used as a marker of sympatho-vagal balance.
        ratio = (
            lf_norm / hf_norm
            if np.isfinite(lf_norm) and np.isfinite(hf_norm) and hf_norm > 0
            else np.nan
        )

        return {
            "VLF_Power": vlf,
            "LF_Power": lf,
            "HF_Power": hf,
            "Total_Power": total_power,
            "LF_(nu)": lf_norm,
            "HF_(nu)": hf_norm,
            "LF/HF": ratio,
        }

    def get_config(self) -> Dict[str, Any]:

        return self.config.__dict__
