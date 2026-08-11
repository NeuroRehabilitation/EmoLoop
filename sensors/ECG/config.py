
"""
sensors.ECG.config

ECG Configuration - dataclass for centralized parameter management.

This module defines the `ECG_Config` dataclass, which encapsulates all tunable
parameters used by ECG signal processing algorithms (e.g., `PanTompkinsAlgorithm`).
It provides sensible defaults for:
  - Analog-to-digital converter (ADC) parameters (VCC, gain, resolution)
  - Digital signal processing parameters (filter type, order, frequency bands)
  - R-peak detection parameters (peak height, distance, prominence thresholds)
  - Signal processing window discarding (initial transients)
  - Sampling rate and algorithm selection

By centralizing these parameters in a configuration object, algorithms can be
easily reconfigured without modifying code, and configurations can be serialized,
loaded from files, or swapped for testing.
"""

from dataclasses import dataclass


@dataclass
class ECG_Config:
    """
    Configuration parameters for ECG signal processing algorithms.

    This dataclass provides a clean, typed container for all tunable settings
    required by ECG processing pipelines (filtering, R-peak detection, etc.).
    Instantiate with defaults or override specific parameters as needed:

        config = ECG_Config(sampling_rate=250, butter_order=5)

    Attributes
    ----------
    VCC : int
        Reference voltage for analog-to-digital conversion (millivolts).
        Represents the full-scale input voltage range of the ADC.
        Default: 3000 mV (3.0 V)
        Typical range: 3000–5000 mV

    gain : int
        Amplifier gain (unitless) applied to the analog signal before ADC.
        Used to scale the digitized signal back to millivolts.
        Default: 1000
        Formula for conversion: (signal * 2^resolution - 1/2) * VCC / gain

    resolution : int
        ADC bit resolution (bits). Determines the total number of discrete
        levels the ADC can represent.
        Default: 16 bits
        Typical values: 8, 10, 12, 14, 16, 24 bits
        Affects the conversion formula: 2^resolution possible levels

    filter_type : str
        Type of digital filter to apply to the signal.
        Default: "bandpass"
        Typical options: "bandpass", "lowpass", "highpass", "bandstop"
        For ECG, "bandpass" is most common (isolates QRS complex).

    butter_order : int
        Order (degree) of the Butterworth filter.
        Higher order = steeper frequency response, but more phase distortion
        if not using zero-phase filtering (sosfiltfilt).
        Default: 4
        Typical range: 2–10
        Note: sosfiltfilt applies filter twice (forward-backward), effectively
              doubling the filter order for frequency response.

    lowpass_freq : int
        High-pass frequency cutoff (Hz) — the lower frequency boundary
        of the bandpass filter.
        Removes baseline wander and slow drift.
        Default: 5 Hz
        Typical range: 0.5–10 Hz
        Note: In bandpass terminology, this is the "low" frequency limit.

    highpass_freq : int
        Low-pass frequency cutoff (Hz) — the upper frequency boundary
        of the bandpass filter.
        Removes high-frequency noise and powerline interference.
        Default: 15 Hz
        Typical range: 10–100 Hz
        Note: In bandpass terminology, this is the "high" frequency limit.

    mph : int or None
        Minimum peak height (MPH) threshold for peak detection.
        Peaks below this amplitude are ignored.
        Default: None (disabled)
        If set, only peaks with amplitude >= mph are considered.
        Useful for rejecting noise and weak signals.

    mpd : int
        Minimum peak distance (MPD) in samples — the minimum separation
        between consecutive detected peaks.
        Prevents detection of multiple peaks in the same beat.
        Default: 35 samples
        Typical value at 1000 Hz sampling: 35 samples ≈ 0.035 seconds
        Set based on expected minimum heart rate (e.g., 35 samples for ~1.7 s minimum RR interval)

    threshold : int
        Minimum peak prominence threshold — the minimum vertical distance
        from a peak to the nearest local minimum.
        Eliminates small, shallow peaks (likely noise).
        Default: 0 (disabled)
        Typical range: 0–100 (algorithm-dependent)

    edge : str
        Type of edges to detect when searching for peaks.
        Default: "rising"
        Options:
          - "rising" : detect rising edges (peaks where derivative goes from + to -)
          - "falling" : detect falling edges (where derivative goes from - to +)
          - "both" : detect both rising and falling edges
        For ECG R-peaks, "rising" is typical (positive QRS deflection).

    kpsh : bool
        Keep peaks same height (KPSH) flag for peak clustering.
        If True, keeps multiple peaks with the same height when they cluster
        within the minimum peak distance (mpd).
        Default: False (keep only the highest peak in each cluster)
        Set True if you expect plateaus or multi-peaked QRS complexes.

    valley : bool
        If True, search for valleys (minima) instead of peaks (maxima).
        Useful for inverted ECG signals or other signal types.
        Default: False (search for peaks)
        For standard ECG, keep False.

    library : str
        Name of the algorithm library to use for processing.
        Default: "neurokit"
        This parameter may be used by selector classes or factory patterns
        to choose between different algorithm implementations.
        Typical options: "neurokit", "pan_tompkins", "custom"

    discard_window : float
        Initial time window to discard from the beginning of the signal (seconds).
        Removes unreliable transient data (filter initialization artifacts,
        signal startup transients).
        Default: 0.15 seconds
        Typical range: 0.1–0.5 seconds
        Converted to samples: discard_window * sampling_rate

    sampling_rate : int
        ECG signal sampling rate (Hz) — number of samples per second.
        Critical for converting between sample indices and time.
        Default: 1000 Hz
        Typical values: 250, 500, 1000 Hz (device-dependent)
        Must match the actual sampling rate of the input signal.

    Notes
    -----
    - All parameters are stored as instance attributes and can be freely
      modified after instantiation.
    - Different ECG devices may require different configurations (e.g.,
      different VCC, gain, sampling_rate).
    - For the Pan-Tompkins algorithm, the filter parameters (butter_order,
      lowpass_freq, highpass_freq) are particularly important.
    - Peak detection parameters (mph, mpd, threshold, edge, kpsh, valley)
      should be tuned based on signal quality and heart rate.

    Examples
    --------
    Default configuration (1000 Hz sampling, standard ECG):
        config = ECG_Config()

    Custom configuration for a different device (250 Hz sampling):
        config = ECG_Config(sampling_rate=250, mpd=9)

    Tighter peak detection (fewer false positives):
        config = ECG_Config(threshold=10, mpd=50)

    Access parameters:
        print(config.lowpass_freq)  # 5
        print(config.sampling_rate)  # 1000
    """

    VCC: int = 3000
    """Reference voltage for ADC (mV). Default: 3000"""

    gain: int = 1000
    """Amplifier gain (unitless). Default: 1000"""

    resolution: int = 16
    """ADC bit resolution (bits). Default: 16"""

    filter_type: str = "bandpass"
    """Digital filter type. Default: "bandpass" """

    butter_order: int = 4
    """Butterworth filter order. Default: 4"""

    lowpass_freq: int = 5
    """High-pass cutoff frequency (Hz). Default: 5"""

    highpass_freq: int = 15
    """Low-pass cutoff frequency (Hz). Default: 15"""

    mph: int = None
    """Minimum peak height threshold. Default: None (disabled)"""

    mpd: int = 35
    """Minimum peak distance (samples). Default: 35"""

    threshold: int = 0
    """Minimum peak prominence threshold. Default: 0 (disabled)"""

    edge: str = "rising"
    """Edge type for peak detection. Default: "rising" """

    kpsh: bool = False
    """Keep peaks same height flag. Default: False"""

    valley: bool = False
    """Detect valleys instead of peaks. Default: False"""

    library: str = "neurokit"
    """Algorithm library name. Default: "neurokit" """

    discard_window: float = 0.15
    """Initial window to discard (seconds). Default: 0.15"""

    sampling_rate: int = 1000
    """ECG sampling rate (Hz). Default: 1000"""