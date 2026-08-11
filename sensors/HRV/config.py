"""
sensors.HRV.config

HRV Configuration - dataclass for centralized parameter management.

This module defines the `HRV_Config` dataclass, which encapsulates all tunable
parameters used by HRV algorithms (e.g., `HRVAlgorithm`). It provides sensible
defaults for:
  - Ectopy detection and removal thresholds
  - Signal processing parameters (window type, interpolation rate)
  - Frequency band boundaries for spectral analysis (VLF, LF, HF)

By centralizing these parameters in a configuration object, algorithms can be
easily reconfigured without modifying code, and configurations can be serialized,
loaded from files, or swapped for testing.
"""

from dataclasses import dataclass


@dataclass
class HRV_Config:
    """
    Configuration parameters for HRV processing algorithms.

    This dataclass provides a clean, typed container for all tunable settings
    required by HRV analysis pipelines. Instantiate with defaults or override
    specific parameters as needed:

        config = HRV_Config(ectopy_threshold=0.25, interpolation_rate=4)

    Attributes
    ----------
    ectopy_threshold : float
        Relative difference threshold (unitless) used in ectopic beat detection.
        An RR interval is flagged as ectopic if its relative difference from
        the previous interval exceeds this value.
        Default: 0.2 (20% relative change)
        Typical range: 0.15–0.3

    window : str
        Name of the window function to apply during spectral analysis (PSD
        computation). Passed directly to `scipy.signal.get_window()`.
        Common choices: "hann", "hamming", "blackman", "tukey"
        Default: "hann"

    interpolation_rate : int
        Sampling rate (Hz) used when resampling the irregularly-sampled RR
        interval series onto a uniform grid. Higher rates yield higher
        spectral resolution but require more computation.
        Default: 4 Hz
        Typical range: 2–10 Hz

    vlf_lfreq : float
        Very Low Frequency (VLF) band lower boundary (Hz).
        Default: 0.0033 Hz (≈ 300 second period)

    vlf_hfreq : float
        Very Low Frequency (VLF) band upper boundary (Hz).
        Default: 0.04 Hz (≈ 25 second period)

    lf_lfreq : float
        Low Frequency (LF) band lower boundary (Hz).
        Default: 0.04 Hz
        Typical for HRV: 0.04–0.15 Hz

    lf_hfreq : float
        Low Frequency (LF) band upper boundary (Hz).
        Default: 0.15 Hz

    hf_lfreq : float
        High Frequency (HF) band lower boundary (Hz).
        Default: 0.15 Hz
        Typical for HRV: 0.15–0.4 Hz (respiratory frequencies)

    hf_hfreq : float
        High Frequency (HF) band upper boundary (Hz).
        Default: 0.4 Hz

    Notes
    -----
    - Frequency band definitions (VLF/LF/HF) follow the ESC Task Force
      guideline recommendations for HRV analysis.
    - The bands should be non-overlapping and cover the full range of
      interest (typically 0.0033–0.4 Hz for standard HRV).
    - Parameters are stored as instance attributes and can be freely
      modified after instantiation or read by algorithm implementations.

    Examples
    --------
    Default configuration:
        config = HRV_Config()

    Custom configuration with modified ectopy threshold:
        config = HRV_Config(ectopy_threshold=0.25)

    Access parameters:
        print(config.lf_lfreq)  # 0.04
        print(config.interpolation_rate)  # 4
    """

    ectopy_threshold: float = 0.2
    """Relative difference threshold for ectopic beat detection (unitless)."""

    window: str = "hann"
    """Window function name for spectral analysis (passed to scipy.signal.get_window)."""

    interpolation_rate: int = 4
    """Sampling rate (Hz) for uniform resampling of RR intervals."""

    vlf_lfreq: float = 0.0033
    """Very Low Frequency band lower boundary (Hz)."""

    vlf_hfreq: float = 0.04
    """Very Low Frequency band upper boundary (Hz)."""

    lf_lfreq: float = 0.04
    """Low Frequency band lower boundary (Hz)."""

    lf_hfreq: float = 0.15
    """Low Frequency band upper boundary (Hz)."""

    hf_lfreq: float = 0.15
    """High Frequency band lower boundary (Hz)."""

    hf_hfreq: float = 0.4
    """High Frequency band upper boundary (Hz)."""
