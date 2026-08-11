"""
PPG base interface.

This module defines the abstract base class used by PPG sensor implementations
in the project. Concrete PPG processors should subclass `PPG_base` and provide
implementations for the abstract methods declared here.

The expectations for implementations:
- Input signals are 1-D numpy arrays (shape: (n_samples,)).
- `filter` should return a filtered 1-D numpy array with the same length as input.
- `detect_peaks` should return an array of peak indices (integer indices into the
  filtered signal) or any other suitable representation agreed on by downstream
  components.
- `rr_intervals` should compute inter-beat (RR) intervals based on detected
  peaks and return a tuple of two numpy arrays (see method docstring below).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class PPG_base(ABC):
    """
    Abstract base class for Photoplethysmography (PPG) signal processing.

    Subclasses must implement the following methods to provide a complete PPG
    processing pipeline for this project:
      - filter: apply preprocessing and bandpass/denoising to raw PPG signal
      - detect_peaks: locate pulse peaks (or characteristic points) in the filtered signal
      - rr_intervals: compute RR (inter-beat) intervals from detected peak indices
      - get_config: return the runtime/configuration parameters being used

    Implementations should work with numpy arrays and avoid side-effects where
    possible (i.e., return new arrays rather than mutating inputs), unless
    documented otherwise by the concrete subclass.
    """

    @abstractmethod
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Filter the raw PPG signal.

        This method should perform any signal preprocessing required by the
        downstream algorithms (e.g., detrending, band-pass Butterworth filter,
        smoothing, resampling). Implementations must accept a 1-D numpy array
        and return a 1-D numpy array of the same length representing the
        filtered signal.

        Parameters
        ----------
        signal : np.ndarray
            Raw PPG signal samples as a 1-D numpy array of floats. Expected
            shape is (n_samples,). NaN handling, clipping and scaling policy
            should be documented by the concrete implementation.

        Returns
        -------
        np.ndarray
            Filtered PPG signal (1-D numpy array). Should have the same length
            as `signal` unless the implementation documents a different contract.
        """
        pass

    @abstractmethod
    def detect_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect peaks in a filtered PPG signal.

        The returned value is typically an array of integer indices pointing to
        the positions of detected peaks in `signal`. Alternatively, implementations
        may return timestamps or boolean masks if that is the agreed convention;
        however, indices are the default expectation in this codebase.

        Parameters
        ----------
        signal : np.ndarray
            Filtered PPG signal (1-D numpy array) on which to perform peak detection.

        Returns
        -------
        np.ndarray
            Array of detected peak indices (dtype int) or another agreed-upon
            representation. If no peaks are found, return an empty numpy array.
        """
        pass

    @abstractmethod
    def rr_intervals(self, peaksIndex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate RR (inter-beat) intervals from peak indices.

        Given an ordered sequence of peak indices (or times), compute the
        inter-beat intervals. The exact units of the returned intervals depend on
        whether `peaksIndex` represents sample indices (in which case the
        implementation will typically convert to seconds using the sampling rate)
        or time values (in seconds).

        Parameters
        ----------
        peaksIndex : np.ndarray
            Ordered array of detected peak positions. Typically integer sample
            indices into the signal array, but the concrete implementation may
            accept timestamps (floats). It is the caller's responsibility to
            follow the convention documented by the implementation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple (rr_intervals, rr_times) where:
              - rr_intervals : np.ndarray
                  1-D array of inter-beat intervals (e.g., in seconds). If N peaks
                  are provided, this array typically has length N-1.
              - rr_times : np.ndarray
                  1-D array of time points associated with the intervals (e.g.,
                  the time of the later peak in each interval or the midpoint).
                  Its length must match that of `rr_intervals`.

            If there are fewer than two peaks, both arrays should be empty.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration used by this PPG processor.

        This method should return a dictionary or mapping with configuration
        parameters relevant for processing (e.g., sampling rate, filter
        cutoff frequencies, detection thresholds). The exact keys are defined
        by concrete implementations but commonly include items like:
          - "sampling_rate"
          - "lowpass_freq"
          - "highpass_freq"
          - "filter_type"
          - "butter_order"
          - "threshold"
          - "window"

        Returns
        -------
        Dict[str, Any]
            Mapping of configuration parameter names to their values.

        """
        pass
