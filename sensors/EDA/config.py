"""
sensors.EDA.config
-------------------

Configuration dataclass for Electrodermal Activity (EDA) signal processing.

This module defines the EDA_Config dataclass which centralizes all tunable
parameters used throughout the EDA processing pipeline (filter parameters,
sampling settings, PSD / frequency-band boundaries, windowing and method
selection).

Usage example:
    from sensors.EDA.config import EDA_Config
    cfg = EDA_Config(sampling_rate=1000, lowpass_freq=5.0)

    # Access values:
    print(cfg.sampling_rate)  # -> 1000

Notes:
- Frequencies are expressed in Hertz (Hz).
- Voltage / ADC related parameters (VCC, resolution) are present to support
  conversions from raw ADC (counts) to physical units when needed.
- Band definitions (vlf, lf, hf) follow common HRV/EDA frequency-band naming
  conventions and can be used when computing power in specific frequency bands.
"""

from dataclasses import dataclass


@dataclass
class EDA_Config:
    """
    Configuration class for Electrodermal Activity (EDA) signal processing.

    This dataclass encapsulates the configuration parameters required for EDA
    signal processing, including sampling frequency, ADC reference/resolution,
    filter parameters, method selection, and frequency-band definitions used for
    spectral analyses.

    Attributes:
        sampling_rate (float):
            Sampling frequency in Hz. Use the actual rate at which the EDA
            sensor is sampled (e.g., 1000.0 for 1 kHz).

        VCC (float):
            Reference voltage used for analog-to-digital conversion. This can be
            useful if raw ADC counts must be converted to microSiemens or volts.
            Units: Volts (or the unit the acquisition board uses). Default is 3 V.

        resolution (float):
            ADC resolution in bits. Used when converting raw integer ADC counts
            to a voltage value (e.g., 16-bit ADC). Default is 16.

        lowpass_butter_order (int):
            Order of the low-pass Butterworth filter applied to the signal. A
            higher order produces a steeper roll-off. Typical small integer values
            (2-6) are common.

        lowpass_freq (float):
            Cutoff frequency (Hz) for the low-pass filter. Frequencies above this
            value are attenuated.

        highpass_butter_order (int):
            Order of the high-pass Butterworth filter applied to the signal.

        highpass_freq (float):
            Cutoff frequency (Hz) for the high-pass filter. Frequencies below this
            value are attenuated (useful to remove slow drifts).

        method (str):
            Name of the EDA processing method to use. Example values:
            - "neurokit" : use NeuroKit2-based extraction routines
            - "custom"   : use a project-specific processing pipeline

        nperseg (int):
            Number of samples per segment for spectral methods such as Welch's
            method. Controls frequency resolution in PSD estimates.

        window (str):
            Window function name used for spectral estimation (e.g., "blackman",
            "hamming", "hann").

        vlf_lfreq, vlf_hfreq (float):
            Very Low Frequency (VLF) band lower and upper boundaries (Hz).
            Typical VLF band for autonomic signals: ~0.0033 - 0.04 Hz.

        lf_lfreq, lf_hfreq (float):
            Low Frequency (LF) band lower and upper boundaries (Hz).
            Typical LF band: ~0.04 - 0.15 Hz.

        hf_lfreq, hf_hfreq (float):
            High Frequency (HF) band lower and upper boundaries (Hz).
            Typical HF band: ~0.15 - 0.4 Hz.
    """

    sampling_rate: int = 1000  # Sampling frequency in Hz

    # ADC / hardware conversion parameters
    VCC: float = 3  # Reference voltage for analog-to-digital conversion (Volts)
    resolution: float = 16  # ADC resolution in bits

    # Filter parameters
    filter_type: str = (
        "lowpass"  # Type of filter to apply (e.g., "bandpass", "lowpass", "highpass")
    )
    lowpass_butter_order: int = 4  # Order of the low-pass Butterworth filter
    lowpass_freq: float = 3  # Low-pass filter cutoff frequency in Hz
    highpass_butter_order: int = 4  # Order of the high-pass Butterworth filter
    highpass_freq: float = 0.05  # High-pass filter cutoff frequency in Hz

    # Processing method and segment/window choices
    get_component_method = str = (
        "highpass"  # Method for EDA component extraction (e.g., "highpass", "cvxEDA","SparsEDA")
    )
    method: str = (
        "neurokit"  # Method for EDA signal processing (e.g., "neurokit", "custom")
    )
    nperseg: int = 128  # Number of samples per segment for spectral processing (Welch)
    window: str = (
        "blackman"  # Window function for spectral estimation (e.g., "blackman", "hamming")
    )

    # Frequency-band definitions for spectral feature extraction (Hz)
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
