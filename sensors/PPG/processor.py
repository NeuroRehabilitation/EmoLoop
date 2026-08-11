
"""
sensors.PPG.processor
---------------------

High-level interface for processing photoplethysmography (PPG) signals.

This module exposes the `PPG` class which wraps a selectable algorithm (implementing
the `sensors.PPG.base.PPG_base` interface) to provide a consistent, high-level API
for:

- filtering raw PPG signals,
- detecting and validating peaks,
- pairing peaks to compute inter-beat (RR) intervals and their timestamps,
- exposing processed results and convenience accessors.

The `PPG` class intentionally does not implement signal processing itself; it delegates
work to the selected algorithm (`PPGAlgorithm` by default) so different algorithm
implementations can be plugged in for testing or device-specific behavior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray

from sensors.PPG.config import PPG_Config
from sensors.PPG.algorithms import PPGAlgorithm
from sensors.PPG.base import PPG_base


class PPG:
    """
    High-level PPG processing interface.

    This wrapper coordinates an algorithm instance (conforming to
    `sensors.PPG.base.PPG_base`) and provides a stable API for consumers
    of PPG processing functionality. The selected algorithm instance is
    responsible for the actual signal-processing steps such as filtering,
    peak detection, and RR-interval computation.

    Responsibilities delegated to the algorithm:
    1. Filtering the PPG signal.
    2. Detecting PPG peaks and valleys.
    3. Pairing peaks with preceding valleys.
    4. Validating peaks using pulse amplitudes.
    5. Calculating inter-beat (RR) intervals and their timestamps.

    Attributes
    ----------
    config : PPG_Config
        Configuration object containing algorithm parameters (sampling rate,
        thresholds, filter settings, etc.). If not provided, a default
        `PPG_Config()` is created.
    sampling_rate : float
        Convenience reference to `config.sampling_rate`.
    algorithm : PPG_base
        The concrete algorithm instance used to do processing. By default a
        `PPGAlgorithm` is instantiated with the provided config.
    _raw_signal : Optional[np.ndarray]
        Most recently processed raw signal (kept for inspection).
    _filtered_signal : Optional[np.ndarray]
        Most recently filtered signal returned by the algorithm.
    _peaks_amplitude : Optional[np.ndarray]
        Amplitudes of validated peaks identified by the algorithm.
    _peaks_index : Optional[np.ndarray]
        Indices (in samples) of validated peaks.
    _rr_intervals : Optional[np.ndarray]
        RR intervals (in seconds) between consecutive validated peaks.
    _rr_time : Optional[np.ndarray]
        Timestamps (seconds) associated with RR intervals (e.g., time of beat).
    """

    def __init__(
        self,
        algorithm: Optional[PPG_base] = None,
        config: Optional[PPG_Config] = None,
    ) -> None:
        """
        Initialize the PPG processing wrapper.

        Parameters
        ----------
        algorithm : Optional[PPG_base], optional
            A pre-instantiated algorithm that implements the `PPG_base`
            interface. If None, `PPGAlgorithm` is constructed with `config`.
        config : Optional[PPG_Config], optional
            Configuration object describing sampling rate and algorithm
            parameters. If None, a default `PPG_Config()` is used.

        Notes
        -----
        The wrapper stores the latest raw/filtered signals and peak/interval
        results as attributes for later access via getter methods.
        """
        self.config = config or PPG_Config()
        self.sampling_rate = self.config.sampling_rate

        if algorithm is not None:
            self.algorithm = algorithm
        else:
            # Default to the project's PPGAlgorithm implementation.
            self.algorithm = PPGAlgorithm(self.config)

        # Internal state that will be filled after calling `process`.
        self._raw_signal: Optional[np.ndarray] = None
        self._filtered_signal: Optional[np.ndarray] = None
        self._peaks_amplitude: Optional[np.ndarray] = None
        self._peaks_index: Optional[np.ndarray] = None
        self._rr_intervals: Optional[np.ndarray] = None
        self._rr_time: Optional[np.ndarray] = None

    def process(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Process a PPG signal end-to-end.

        This method runs filtering, peak detection/validation, and RR-interval
        computation. Results are stored on the instance and also returned as
        a dictionary for convenience.

        Parameters
        ----------
        signal : np.ndarray
            Raw PPG signal. Peak indices in results are expressed in samples
            (i.e., indices into this array).

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - "raw_signal": np.ndarray, the input signal (converted to ndarray).
            - "filtered_signal": np.ndarray, the filtered signal returned by
              the algorithm (may be None if algorithm returns None).
            - "peaks_amplitude": np.ndarray, amplitudes of validated peaks.
            - "peaks_index": np.ndarray, sample indices of validated peaks.
            - "rr_intervals": np.ndarray, RR intervals in seconds between peaks.
            - "rr_time": np.ndarray, timestamps (seconds) for intervals or beats.
        """
        if signal is None:
            raise ValueError("Signal cannot be None.")

        # Ensure we operate on a numpy array for consistent downstream behavior.
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)

        if signal.size == 0:
            raise ValueError("Signal must be non-empty.")

        self._raw_signal = signal

        # Filter the raw PPG signal using the selected algorithm.
        self._filtered_signal = self.algorithm.filter(signal)

        # Detect and validate PPG peaks. The algorithm is expected to return
        # a mapping with keys "PeaksAmp" and "PeaksIndex".
        peak_data = self.algorithm.detect_peaks(
            self._raw_signal
        )

        # Store validated peak amplitudes and their sample indices for later access.
        self._peaks_amplitude = peak_data["PeaksAmp"]
        self._peaks_index = peak_data["PeaksIndex"]

        # Calculate intervals (in seconds) and timestamps from validated peak indices.
        self._rr_intervals, self._rr_time = (
            self.algorithm.rr_intervals(
                self._peaks_index
            )
        )

        return {
            "raw_signal": self._raw_signal,
            "filtered_signal": self._filtered_signal,
            "peaks_amplitude": self._peaks_amplitude,
            "peaks_index": self._peaks_index,
            "rr_intervals": self._rr_intervals,
            "rr_time": self._rr_time,
        }

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply the algorithm's filter to a signal.

        This is a convenience wrapper that forwards to the algorithm's `filter`
        method. Consumers may call this directly to examine the filtered output
        without performing peak detection.

        Parameters
        ----------
        signal : np.ndarray
            Raw PPG signal to filter.

        Returns
        -------
        np.ndarray
            Filtered signal as returned by the algorithm.
        """
        return self.algorithm.filter(signal)

    def detect_peaks(
        self,
        signal: np.ndarray,
    ) -> ndarray:
        """
        Detect and validate PPG peaks using the selected algorithm.

        Parameters
        ----------
        signal : np.ndarray
            Raw or preprocessed signal to run peak detection on.

        Returns
        -------
        ndarray
            Algorithm-specific peak detection output. In this codebase the
            convention is for `detect_peaks` to return a dictionary-like object
            containing keys such as "PeaksAmp" and "PeaksIndex". This wrapper
            exposes the exact return value for callers that need access to the
            underlying algorithm's full output.

        Notes
        -----
        Use `process()` for the full processing pipeline which will update
        the instance attributes (`_peaks_index`, `_peaks_amplitude`, etc.).
        """
        return self.algorithm.detect_peaks(signal)

    def get_rr_intervals(
        self,
        peaks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute RR intervals and associated timestamps from peak indices.

        Parameters
        ----------
        peaks : np.ndarray
            Array of peak indices (sample positions). These are typically
            the validated peaks produced by the algorithm.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple (rr_intervals, rr_time) where:
            - rr_intervals: numpy array of inter-beat intervals (seconds).
            - rr_time: numpy array of timestamps (seconds) associated to each interval.

        """
        return self.algorithm.rr_intervals(peaks)

    def get_raw_signal(self) -> Optional[np.ndarray]:
        """
        Return the most recently processed raw signal.

        Returns
        -------
        Optional[np.ndarray]
            Raw numpy array passed to `process`, or None if `process` has not
            been called since the last algorithm change.
        """
        return self._raw_signal

    def get_filtered_signal(self) -> Optional[np.ndarray]:
        """
        Return the most recently filtered signal.

        Returns
        -------
        Optional[np.ndarray]
            Filtered signal as returned by the algorithm during the most recent
            `process()` call, or None if not available.
        """
        return self._filtered_signal

    def get_peaks_amplitude(self) -> Optional[np.ndarray]:
        """
        Return amplitudes of the most recently validated peaks.

        Returns
        -------
        Optional[np.ndarray]
            Array of peak amplitudes, or None if peaks are not available.
        """
        return self._peaks_amplitude

    def get_peaks_index(self) -> Optional[np.ndarray]:
        """
        Return indices of the most recently validated peaks.

        Returns
        -------
        Optional[np.ndarray]
            Array of sample indices for validated peaks, or None if not available.
        """
        return self._peaks_index

    def get_config(self) -> Dict[str, Any]:
        """
        Return the algorithm configuration as a dictionary-like mapping.

        Returns
        -------
        Dict[str, Any]
            The configuration used by the currently selected algorithm. The
            exact structure depends on `PPG_Config` and the algorithm's
            `get_config()` implementation.
        """
        return self.algorithm.get_config()

    def set_algorithm(self, algorithm: PPG_base) -> None:
        """
        Replace the PPG algorithm and clear previous results.

        Parameters
        ----------
        algorithm : PPG_base
            New algorithm instance conforming to `PPG_base`. After setting,
            previously-stored signals, peaks, and intervals are cleared to
            avoid mixing results from different algorithm instances.

        Notes
        -----
        This operation is destructive with regard to prior processing results.
        """
        self.algorithm = algorithm

        # Clear cached processing results to avoid stale data mixing.
        self._raw_signal = None
        self._filtered_signal = None
        self._peaks_amplitude = None
        self._peaks_index = None
        self._rr_intervals = None
        self._rr_time = None