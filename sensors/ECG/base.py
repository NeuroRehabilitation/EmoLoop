
"""
sensors.ECG.base

ECG Base Interface - abstract base class defining the algorithm contract.

This module defines `ECG_base`, an abstract base class (ABC) that specifies the
interface that all ECG algorithm implementations must satisfy. By enforcing a
consistent interface, this allows different signal processing algorithms
(e.g., Pan-Tompkins, derivative-based methods, machine learning approaches)
to be swapped at runtime without changing caller code.

Any concrete ECG processing algorithm should inherit from `ECG_base` and
implement all abstract methods. This enables flexible, pluggable ECG signal
processing pipelines for R-peak detection and HRV analysis.

The typical workflow is:
  1. filter() - Apply bandpass filtering to clean the raw ECG signal
  2. detect_r_peaks() - Detect R-peaks (QRS complexes) in the filtered signal
  3. rr_intervals() - Calculate RR intervals and timestamps from R-peaks
  4. (Optionally) get_config() - Retrieve algorithm configuration
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class ECG_base(ABC):
    """
    Abstract base class defining the interface for ECG signal processing algorithms.

    This class specifies the contract that all ECG processing implementations must
    satisfy. Subclasses must implement methods for:
      - Signal filtering (noise removal, baseline correction)
      - R-peak detection (identification of the main QRS deflection)
      - RR interval calculation (for heart rate variability analysis)
      - Configuration retrieval

    The class is algorithm-agnostic: different implementations (Pan-Tompkins,
    template matching, machine learning, etc.) can be swapped as long as they
    conform to this interface.

    Notes
    -----
    - ECG signals are typically sampled at 250–1000 Hz depending on the device.
    - R-peaks are the positive deflections of the QRS complex, representing
      ventricular depolarization.
    - RR intervals (in seconds or milliseconds) are fundamental for HRV analysis.
    - All implementations should handle edge cases gracefully (short signals,
      noisy data, artifacts).

    Example
    -------
    Implementing a concrete algorithm:

        from sensors.ECG.base import ECG_base
        import numpy as np

        class MyECGAlgorithm(ECG_base):
            def filter(self, signal):
                # Apply bandpass filter
                return filtered_signal

            def detect_r_peaks(self, signal):
                # Detect R-peaks and return indices
                return r_peak_indices

            def rr_intervals(self, r_peaks):
                # Compute RR intervals and timestamps
                return rr_intervals, rr_time

            def get_config(self):
                # Return configuration dictionary
                return self.config
    """

    @abstractmethod
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Filter the raw ECG signal to remove noise and baseline drift.

        Applies signal processing filters (typically bandpass) to isolate the
        ECG waveform of interest (QRS complex frequency range) while removing:
          - Low-frequency baseline wander (< ~5 Hz)
          - High-frequency electrical noise (> ~40 Hz)
          - 50/60 Hz powerline interference

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal (in arbitrary ADC units or millivolts).
            Typically 1D array of shape (N,) where N is the number of samples.

        Returns
        -------
        np.ndarray
            Filtered ECG signal with the same shape as input. Typically a
            bandpass-filtered signal emphasizing the QRS complex.

        Notes
        -----
        - The exact filter type, order, and frequency bands are algorithm-
          and configuration-dependent.
        - Common implementations use Butterworth or Chebyshev filters.
        - Filter parameters should be read from a configuration object.
        - High-quality filtering is critical for accurate R-peak detection.

        Examples
        --------
        algo = MyECGAlgorithm(config)
        filtered = algo.filter(raw_ecg_signal)
        """
        pass

    @abstractmethod
    def detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks (QRS complexes) in a filtered ECG signal.

        Identifies the sample indices corresponding to R-peaks, which are the
        dominant positive deflections in the ECG signal's QRS complex. R-peaks
        represent ventricular depolarization and are critical for heart rate
        and heart rate variability (HRV) analysis.

        Parameters
        ----------
        signal : np.ndarray
            Filtered ECG signal (output from filter() method). Should be a
            1D array of shape (N,) where N is the number of samples.

        Returns
        -------
        np.ndarray
            Array of R-peak sample indices (int). These are 0-based indices
            into the input signal where R-peaks are located. Shape: (M,) where
            M is the number of detected R-peaks. Typically sorted in ascending
            order.

        Raises
        ------
        ValueError
            If no R-peaks are detected in the signal (e.g., signal too noisy,
            too short, or no cardiac activity).

        Notes
        -----
        - The algorithm is implementation-specific (Pan-Tompkins, template
          matching, machine learning, etc.).
        - Accuracy depends on signal quality, sampling rate, and algorithm
          parameters.
        - The returned indices should point to the exact sample of the R-peak
          maximum (or the nearest sample for discrete representation).
        - For HRV analysis, accurate R-peak detection is critical; even small
          errors propagate to RR interval calculations.

        Examples
        --------
        algo = MyECGAlgorithm(config)
        filtered = algo.filter(raw_ecg_signal)
        r_peak_indices = algo.detect_r_peaks(filtered)
        # r_peak_indices might be: array([1050, 1300, 1550, ...])
        """
        pass

    @abstractmethod
    def rr_intervals(self, r_peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate RR intervals and their timestamps from R-peak indices.

        Computes the time (in seconds) between consecutive R-peaks, along with
        the timestamps at which each interval ends. These RR intervals form the
        basis for heart rate and heart rate variability (HRV) analysis.

        Parameters
        ----------
        r_peaks : np.ndarray
            Array of R-peak sample indices (int) as returned by detect_r_peaks().
            Must have at least 2 elements to compute intervals. Shape: (M,) where
            M >= 2.

        Returns
        -------
        tuple (rr_intervals, rr_time)
            - rr_intervals : np.ndarray
                RR intervals in seconds (float). Computed as the time difference
                between consecutive R-peaks. Shape: (M-1,).
            - rr_time : np.ndarray
                Timestamps of RR interval endpoints in seconds (float).
                Corresponds to the sample time of peaks[1:] / sampling_rate.
                Shape: (M-1,).

        Notes
        -----
        - RR intervals are typically in the range 0.6–1.2 seconds for resting
          heart rates of 50–100 BPM.
        - Both output arrays have length M-1 (one fewer than the number of
          R-peaks).
        - If the input has fewer than 2 peaks, the output arrays will be empty.
        - The sampling_rate must be known to convert sample indices to seconds;
          this is typically stored in a configuration object.
        - These outputs are passed to HRV processing algorithms (e.g.,
          HRVAlgorithm) for further analysis (time-domain, frequency-domain,
          non-linear metrics).

        Examples
        --------
        algo = MyECGAlgorithm(config)
        filtered = algo.filter(raw_ecg_signal)
        r_peak_indices = algo.detect_r_peaks(filtered)
        rr_intervals, rr_time = algo.rr_intervals(r_peak_indices)
        # rr_intervals: array([1.05, 1.02, 1.03, ...])  # in seconds
        # rr_time: array([4.20, 5.25, 6.27, ...])  # in seconds
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Retrieve the configuration parameters used by the algorithm.

        Returns a dictionary containing all configuration settings used by the
        algorithm (e.g., sampling rate, filter parameters, peak detection
        thresholds). This allows callers to inspect how the algorithm is
        configured and serialize/persist the configuration if needed.

        Returns
        -------
        Dict[str, Any]
            Dictionary of configuration parameters. Keys and values are
            algorithm-specific but commonly include:
              - "sampling_rate" : int (Hz)
              - "filter_type" : str (e.g., "bandpass")
              - "lowpass_freq" : float (Hz)
              - "highpass_freq" : float (Hz)
              - ... (other algorithm-specific parameters)

        Notes
        -----
        - The structure and content of the returned dictionary depend on the
          algorithm implementation.
        - Typically wraps a configuration object (e.g., ECG_Config dataclass)
          and returns it as a dict (or its __dict__).
        - Useful for logging, validation, or passing configuration to downstream
          processors.

        Examples
        --------
        algo = MyECGAlgorithm(config)
        config_dict = algo.get_config()
        print(config_dict)  # {'sampling_rate': 250, 'filter_type': 'bandpass', ...}
        """
        pass