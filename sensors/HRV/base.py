
"""
sensors.HRV.base

HRV Base Interface - abstract base class defining the algorithm contract.

This module defines `HRV_base`, an abstract base class (ABC) that specifies the
interface that all HRV algorithm implementations must satisfy. By enforcing a
consistent interface, this allows different algorithm implementations (e.g.,
signal processing variants, libraries) to be swapped at runtime without
changing caller code.

Any concrete HRV algorithm should inherit from `HRV_base` and implement all
abstract methods to ensure compatibility with higher-level processors like
`sensors.HRV.processor.HRV`.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class HRV_base(ABC):
    """
    Abstract base class defining the interface for HRV algorithm implementations.

    This class specifies the contract that all HRV algorithms must satisfy.
    Subclasses must implement all abstract methods to compute various HRV metrics
    from RR interval series:

      - Ectopy removal (outlier/artifact detection and filtering)
      - Heart rate derivation from RR intervals
      - Time-domain HRV metrics (SDNN, RMSSD, NN50, etc.)
      - Frequency-domain analysis (PSD / spectral decomposition)
      - Frequency-domain metrics (VLF/LF/HF band powers, ratios)
      - Non-linear metrics (Poincaré plot features, entropy measures)

    Notes
    -----
    - RR intervals are typically provided in seconds (float).
    - All methods should handle edge cases gracefully (e.g., empty arrays,
      arrays with insufficient samples) by returning np.nan or empty containers.
    - Algorithm implementations can rely on a configuration object (e.g.,
      `HRV_Config`) to define parameters like frequency bands, thresholds, etc.

    Example
    -------
    Implementing a concrete algorithm:

        from sensors.HRV.base import HRV_base
        import numpy as np

        class MyHRVAlgorithm(HRV_base):
            def remove_ectopy_beats(self, rr_intervals):
                # Custom ectopy removal logic
                return rr_intervals

            def heart_rate(self, rr_intervals):
                # Compute heart rate from RR intervals
                return {"Avg HR": 60.0 / np.mean(rr_intervals)}

            # ... implement all other abstract methods
    """

    @abstractmethod
    def remove_ectopy_beats(self, rr_intervals: np.ndarray) -> np.ndarray:
        """
        Detect and remove ectopic beats (artifacts) from RR interval series.

        Ectopic beats are premature heartbeats that interrupt the normal sinus
        rhythm. They introduce irregularities that can distort HRV metrics.
        This method identifies suspect intervals (typically using a relative
        difference threshold or statistical test) and returns a filtered series.

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals (in seconds). Shape: (N,) for N intervals.

        Returns
        -------
        np.ndarray
            Filtered RR intervals with ectopic beats and their associated
            artifacts removed. Typically a subset of the input array.
            Shape: (M,) where M <= N.

        Notes
        -----
        - The exact criterion for detecting ectopy is algorithm-dependent.
        - Common heuristics include relative difference thresholds or
          statistical outlier detection.
        - Implementations should handle edge cases (empty input, single sample)
          by returning the input unchanged or an empty array.
        """
        pass

    @abstractmethod
    def rr_intervals(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Compute basic summary statistics of RR intervals.

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals (in seconds).

        Returns
        -------
        Dict[str, float]
            Dictionary with RR interval statistics. Common keys include:
              - "Avg RR": mean RR interval
              - "Min RR": minimum RR interval
              - "Max RR": maximum RR interval
              - "SD RR": standard deviation of RR intervals
            (Exact keys are algorithm-specific.)

        Notes
        -----
        - All values should be in the same units as the input (typically seconds).
        - If input is empty, return a dict with keys mapped to np.nan.
        """
        pass

    @abstractmethod
    def heart_rate(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Derive heart rate metrics from RR intervals.

        Converts RR intervals to instantaneous heart rate (beats per minute)
        and computes summary statistics (mean, min, max, standard deviation).

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals (in seconds).

        Returns
        -------
        Dict[str, float]
            Dictionary with heart rate statistics (in beats per minute). 
            Common keys include:
              - "Avg HR": mean heart rate
              - "Min HR": minimum heart rate
              - "Max HR": maximum heart rate
              - "SD HR": standard deviation of heart rate
            (Exact keys are algorithm-specific.)

        Notes
        -----
        - Heart rate is derived as: HR = 60 / RR (BPM).
        - If input is empty, return a dict with keys mapped to np.nan.
        """
        pass

    @abstractmethod
    def time_domain_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """
        Compute time-domain HRV metrics from RR intervals.

        Time-domain metrics summarize variability and irregularity in the
        RR interval sequence (e.g., standard deviation, root-mean-square of
        successive differences, counts of large interval differences).
        These metrics do not require frequency analysis and are fast to compute.

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals (in seconds).

        Returns
        -------
        Dict[str, Any]
            Dictionary of time-domain HRV features. Common keys include:
              - "SDNN": standard deviation of NN intervals (ms)
              - "RMSSD": root-mean-square of successive differences (ms)
              - "NN50": count of successive differences > 50 ms
              - "pNN50": percentage of NN50 (%)
              - "NN20": count of successive differences > 20 ms
              - "pNN20": percentage of NN20 (%)
              - ... other measures (algorithm-specific)
            (Exact keys and units depend on the implementation.)

        Notes
        -----
        - Many time-domain metrics are reported in milliseconds by convention.
        - If input is empty or too small, return a dict with keys mapped to np.nan.
        """
        pass

    @abstractmethod
    def frequencyAnalysis(
        self, rr_intervals: np.ndarray, rr_time: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform frequency-domain analysis on RR intervals (PSD computation).

        Converts the irregularly-sampled RR interval time series into a
        power spectral density (PSD) estimate via interpolation and spectral
        methods (e.g., FFT, Welch's method).

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals (in seconds). Shape: (N,).
        rr_time : np.ndarray
            Timestamps corresponding to `rr_intervals` (in seconds). Shape: (N,).
            Used to establish the irregular sampling grid for interpolation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple (freqs, power) where:
              - freqs: 1D array of frequency bin centres (Hz).
              - power: 1D array of power spectral density values
                corresponding to each frequency.
            Both arrays should have the same length. If input is insufficient
            for reliable PSD estimation, return two empty arrays.

        Notes
        -----
        - Algorithms typically use spline or linear interpolation to create
          a uniformly-sampled series, then apply a spectral method.
        - The frequency resolution depends on the analysis window and
          interpolation rate.
        - If input is too short (< 4 samples), return empty arrays.
        - Common frequency range for HRV: 0–0.5 Hz.
        """
        pass

    @abstractmethod
    def frequency_domain_features(
        self, freqs: np.ndarray, power: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute frequency-domain HRV metrics from power spectral density.

        Integrates PSD over standard HRV frequency bands (VLF, LF, HF) to
        quantify power in each physiological range. Also computes normalized
        units and band ratios (e.g., LF/HF).

        Parameters
        ----------
        freqs : np.ndarray
            Frequency axis (Hz). Shape: (M,).
        power : np.ndarray
            Power spectral density values corresponding to `freqs`. Shape: (M,).

        Returns
        -------
        Dict[str, Any]
            Dictionary of frequency-domain HRV features. Common keys include:
              - "VLF_Power": Very Low Frequency band power (< 0.04 Hz)
              - "LF_Power": Low Frequency band power (0.04–0.15 Hz)
              - "HF_Power": High Frequency band power (0.15–0.4 Hz)
              - "Total_Power": sum of all band powers
              - "LF_(nu)": LF power in normalized units (%)
              - "HF_(nu)": HF power in normalized units (%)
              - "LF/HF": ratio of LF to HF (normalized)
              - ... other measures (algorithm-specific)
            (Exact keys and units are algorithm-specific.)

        Notes
        -----
        - Power is typically integrated using trapezoidal rule.
        - Normalized units are computed as: power / (Total - VLF) * 100.
        - If frequency/power arrays are empty, return a dict with keys
          mapped to np.nan.
        - Band limits (VLF/LF/HF boundaries) are typically defined in config.
        """
        pass

    @abstractmethod
    def non_linear_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """
        Compute non-linear HRV metrics from RR intervals.

        Non-linear methods capture complex, chaotic, or scaling properties of
        the RR sequence that may not be evident from time/frequency analysis.
        Common approaches include Poincaré plot geometry, entropy measures,
        fractal exponents, etc.

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals (in seconds).

        Returns
        -------
        Dict[str, Any]
            Dictionary of non-linear HRV features. Common keys include:
              - "SD1": Poincaré short-axis scatter (ms)
              - "SD2": Poincaré long-axis scatter (ms)
              - "SD2/SD1": ratio of long to short axis
              - "Entropy": approximate or sample entropy
              - ... other measures (algorithm-specific)
            (Exact keys and units depend on the implementation.)

        Notes
        -----
        - Poincaré plot features (SD1/SD2) are derived from the 2D scatter of
          (RR_n, RR_{n+1}) points.
        - Many non-linear measures are sensitive to input size and quality.
        - If input is too small or invalid, return a dict with keys mapped to np.nan.
        """
        pass