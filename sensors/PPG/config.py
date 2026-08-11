"""
Configuration module for Photoplethysmography (PPG) signal processing.

This module defines the configuration parameters used for filtering and processing
PPG signals, including bandpass filter settings, sampling rate, and detection thresholds.
"""

from dataclasses import dataclass


@dataclass
class PPG_Config:
    """
    Configuration class for PPG (Photoplethysmography) signal processing parameters.

    This dataclass encapsulates all configurable parameters used in the PPG signal
    processing pipeline, including filter characteristics and peak detection settings.

    Attributes:
        lowpass_freq (float): Cutoff frequency for the low-pass filter in Hz.
            Default is 5 Hz, filtering out high-frequency noise.

        highpass_freq (float): Cutoff frequency for the high-pass filter in Hz.
            Default is 0.1 Hz, removing DC offset and very low-frequency drift.

        butter_order (int): Order of the Butterworth filter. Higher orders provide
            steeper rolloff but may introduce more phase distortion.
            Default is 2 (second-order filter).

        filter_type (str): Type of filter configuration to apply. Currently supports
            "bandpass" for combined high-pass and low-pass filtering.
            Default is "bandpass".

        sampling_rate (float): Sampling rate of the PPG signal in Hz.
            Default is 1000 Hz (1 kHz).

        threshold (float): Detection threshold for peak/pulse identification as a
            fraction of the signal amplitude. Range: 0.0 to 1.0.
            Default is 0.7 (70% of max amplitude).

        window (int): Window size for signal processing algorithms in seconds.
            Used in moving window analyses and smoothing operations.
            Default is 5 seconds.
    """

    lowpass_freq: float = 5
    highpass_freq: float = 0.1
    butter_order: int = 2
    filter_type: str = "bandpass"
    sampling_rate: float = 1000
    threshold: float = 0.7
    window: int = 5