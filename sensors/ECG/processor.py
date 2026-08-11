"""
sensors.ECG.processor

High-level ECG signal processing wrapper.

This module provides the ECG class, which acts as a sensor-agnostic processor for
electrocardiogram (ECG) analysis. It coordinates an algorithm implementation
(conforming to the ECG_base interface) and exposes a simple API to:
  - Filter raw ECG signals to remove noise and baseline drift
  - Detect R-peaks (main QRS deflections) in the ECG waveform
  - Calculate RR intervals for heart rate variability (HRV) analysis
  - Extract heart rate metrics

The ECG class itself is algorithm-independent: provide a custom algorithm that
implements the methods defined by `sensors.ECG.base.ECG_base` and the processor
will delegate the heavy lifting to that algorithm.

Notes
-----
- Raw ECG signals are typically in arbitrary ADC units; conversion to millivolts
  is handled by the algorithm's filter() method.
- The sampling_rate and other parameters are read from config (not separate parameters).
- The processor stores results of the last process() call in internal attributes
  and exposes getters to retrieve them.
- R-peak detection is critical for accurate HRV analysis; even small errors
  propagate to downstream HRV metrics.
"""

from typing import Dict, Any, Optional
import numpy as np
from sensors.ECG.config import ECG_Config
from sensors.ECG.algorithms import PanTompkinsAlgorithm
from sensors.ECG.base import ECG_base


class ECG:
    """
    High-level ECG processor that works with any algorithm implementing ECG_base.

    Responsibilities
    -----------------
    - Validate inputs for processing.
    - Delegate computation of ECG metrics to the provided algorithm instance.
    - Store the most recent results and expose convenient getters.
    - Provide small helper wrappers to call common algorithm methods.

    This class is intentionally lightweight — it aims to be the glue between
    caller code (UI, data pipelines) and the algorithm implementations.

    Processing pipeline (in process() method):
      1. Filter raw ECG signal to isolate QRS complex (algorithm.filter)
      2. Detect R-peaks in filtered signal (algorithm.detect_r_peaks)
      3. Calculate RR intervals from R-peaks (algorithm.rr_intervals)
      4. Store results internally for later retrieval

    Attributes
    ----------
    config : ECG_Config
        Configuration object containing algorithm parameters:
          - sampling_rate: ECG sampling frequency (Hz, typically 250–1000)
          - filter settings: bandpass cutoff frequencies, filter order
          - peak detection settings: thresholds, distance, prominence
          - ADC conversion parameters: VCC, gain, resolution
          - ... (all config attributes from ECG_Config)

    sampling_rate : int
        Sampling rate of the ECG signal in Hz. Derived from config.sampling_rate.
        Used to convert between sample indices and time (seconds).

    algorithm : ECG_base
        Instance of an algorithm implementing the `ECG_base` interface.
        The algorithm is responsible for actual ECG computation routines
        (filtering, R-peak detection, RR interval calculation).

    _raw_signal : Optional[np.ndarray]
        Stored raw ECG signal from the last process() call (if any).
        None until process() is invoked.

    _filtered_signal : Optional[np.ndarray]
        Stored filtered ECG signal from the last process() call.
        None until process() is invoked.

    _r_peaks : Optional[np.ndarray]
        Stored R-peak indices from the last process() call.
        None until process() is invoked.

    _rr_intervals : Optional[np.ndarray]
        Stored RR intervals (in seconds) from the last process() call.
        None until process() is invoked.

    _rr_time : Optional[np.ndarray]
        Stored RR interval timestamps (in seconds) from the last process() call.
        None until process() is invoked.

    Notes on algorithm swapping
    ---------------------------
    You can instantiate ECG with a custom algorithm that implements `ECG_base`.
    If no algorithm is provided, a default `PanTompkinsAlgorithm` implementation
    will be instantiated using `config`. Call set_algorithm() to swap algorithms
    at runtime.
    """

    def __init__(
        self, algorithm: Optional[ECG_base] = None, config: Optional[ECG_Config] = None
    ) -> None:
        """
        Initialize the ECG processor.

        Parameters
        ----------
        algorithm : Optional[ECG_base]
            Custom algorithm instance to use for processing. If None, the
            processor will instantiate the default algorithm `PanTompkinsAlgorithm`
            using the provided config. The provided instance must implement
            the `ECG_base` interface.

        config : Optional[ECG_Config]
            Configuration object with algorithm/sensor parameters. If None,
            a default `ECG_Config()` instance will be created. Configuration
            contains parameters such as sampling_rate, filter settings,
            peak detection thresholds, and ADC conversion parameters.

        Notes
        -----
        - The configuration is required by many algorithms (sampling rate,
          filter cutoff frequencies, peak detection parameters, etc.).
          Ensure `config` contains expected fields for the chosen algorithm.
        - The processor does not itself perform ADC conversion or signal
          conditioning — it assumes the input signal is raw or minimally
          preprocessed when provided to process().
        """
        # Initialize config first (always needed)
        self.config = config or ECG_Config()

        # Get sampling_rate from config
        self.sampling_rate = self.config.sampling_rate

        # Instantiate algorithm (default: PanTompkinsAlgorithm)
        if algorithm is not None:
            self.algorithm = algorithm
        else:
            self.algorithm = PanTompkinsAlgorithm(self.config)

        # Store processing results (None until process() is called)
        self._raw_signal: Optional[np.ndarray] = None
        self._filtered_signal: Optional[np.ndarray] = None
        self._r_peaks: Optional[np.ndarray] = None
        self._rr_intervals: Optional[np.ndarray] = None
        self._rr_time: Optional[np.ndarray] = None

    def process(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Process raw ECG signal and extract heart rate metrics.

        Full processing pipeline:
          - Filter raw ECG to remove noise and baseline drift
          - Detect R-peaks (QRS complex maxima) in filtered signal
          - Calculate RR intervals and timestamps
          - Store results in instance attributes and return a consolidated dict

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal as a 1D array. Typically in arbitrary ADC units
            or millivolts (device-dependent). Must be non-empty.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
              - "raw_signal": np.ndarray (input signal, copy or reference)
              - "filtered_signal": np.ndarray (output from algorithm.filter)
              - "r_peaks": np.ndarray (R-peak sample indices)
              - "rr_intervals": np.ndarray (RR intervals in seconds)
              - "rr_time": np.ndarray (RR interval timestamps in seconds)

        Raises
        ------
        ValueError
            If `signal` is None or empty.
        ValueError
            If the algorithm cannot detect any R-peaks (e.g., signal too noisy
            or too short).

        Notes
        -----
        - Input signal is automatically converted to np.ndarray if not already.
        - Results are cached in instance attributes (_raw_signal, _filtered_signal,
          etc.) for later retrieval via getter methods.
        - The algorithm's implementation determines the exact processing details
          (e.g., Pan-Tompkins vs. alternative R-peak detection methods).
        - RR intervals are computed as differences between consecutive R-peak
          times (in seconds).

        Examples
        --------
        ecg = ECG(config=my_config)
        result = ecg.process(raw_signal_array)
        print(f"Detected {len(result['r_peaks'])} R-peaks")
        print(f"Mean RR interval: {np.mean(result['rr_intervals']) * 1000:.1f} ms")
        """
        # Validate input
        if signal is None or len(signal) == 0:
            raise ValueError("Signal must be a non-empty numpy array")

        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        # Store raw signal
        self._raw_signal = signal

        # Step 1: Filter the signal
        filtered_signal = self.algorithm.filter(signal)
        self._filtered_signal = filtered_signal

        # Step 2: Detect R-peaks
        r_peaks = self.algorithm.detect_r_peaks(filtered_signal)
        self._r_peaks = r_peaks

        # Step 3: Calculate RR intervals
        rr_intervals, rr_time = self.algorithm.rr_intervals(r_peaks)
        self._rr_intervals = rr_intervals
        self._rr_time = rr_time

        # Return full results
        return {
            "raw_signal": signal,
            "filtered_signal": filtered_signal,
            "r_peaks": r_peaks,
            "rr_intervals": rr_intervals,
            "rr_time": rr_time,
        }

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Filter raw ECG signal only (skip R-peak detection and RR calculation).

        Applies the algorithm's filter to remove noise and baseline drift,
        isolating the QRS complex frequency range. Useful when you only need
        the filtered signal without full R-peak detection.

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal (in arbitrary ADC units or millivolts).

        Returns
        -------
        np.ndarray
            Filtered ECG signal with the same shape as input. Typically a
            bandpass-filtered signal emphasizing the QRS complex.

        Notes
        -----
        - This is a convenience wrapper around algorithm.filter().
        - Does not cache results in instance attributes.
        - For full processing (including R-peak detection), use process() instead.
        - Filter parameters are determined by the configuration.

        Examples
        --------
        ecg = ECG(config=my_config)
        filtered = ecg.filter(raw_signal)
        """
        return self.algorithm.filter(signal)

    def detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks only (skip filtering and RR interval calculation).

        Identifies R-peak (QRS complex) locations in a signal. Typically called
        on a pre-filtered signal (output from filter()). Useful when filtering
        is performed elsewhere or when you only need peak locations.

        Parameters
        ----------
        signal : np.ndarray
            Filtered ECG signal (typically output from filter()).

        Returns
        -------
        np.ndarray
            Array of R-peak sample indices (int). Sorted in ascending order.

        Raises
        ------
        ValueError
            If no R-peaks are detected in the signal.

        Notes
        -----
        - This is a convenience wrapper around algorithm.detect_r_peaks().
        - Input should ideally be pre-filtered (e.g., from filter()).
        - Does not cache results in instance attributes.
        - For full processing, use process() instead.

        Examples
        --------
        ecg = ECG(config=my_config)
        filtered = ecg.filter(raw_signal)
        r_peaks = ecg.detect_r_peaks(filtered)
        """
        return self.algorithm.detect_r_peaks(signal)

    def get_rr_intervals(self, r_peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate RR intervals and timestamps from R-peak indices.

        Computes the time (in seconds) between consecutive R-peaks and the
        timestamps at which each interval ends. These intervals are fundamental
        for heart rate and HRV analysis.

        Parameters
        ----------
        r_peaks : np.ndarray
            Array of R-peak sample indices (int) as returned by detect_r_peaks().

        Returns
        -------
        tuple (rr_intervals, rr_time)
            - rr_intervals : np.ndarray
                RR intervals in seconds (float). Shape: (M-1,) where M is the
                number of R-peaks.
            - rr_time : np.ndarray
                Timestamps of RR interval endpoints in seconds. Shape: (M-1,).

        Notes
        -----
        - This is a convenience wrapper around algorithm.rr_intervals().
        - Requires self.sampling_rate to be correctly set (from config).
        - Both output arrays have length one less than the number of R-peaks.
        - Does not cache results in instance attributes.
        - For full processing, use process() instead.

        Examples
        --------
        ecg = ECG(config=my_config)
        filtered = ecg.filter(raw_signal)
        r_peaks = ecg.detect_r_peaks(filtered)
        rr_intervals, rr_time = ecg.get_rr_intervals(r_peaks)
        """
        return self.algorithm.rr_intervals(r_peaks)

    # Getter methods for stored results
    # These return cached results from the last process() call.

    def get_raw_signal(self) -> Optional[np.ndarray]:
        """
        Get the raw ECG signal from the last process() call.

        Returns
        -------
        Optional[np.ndarray]
            Raw ECG signal (input to process()), or None if process() has not
            been called yet.
        """
        return self._raw_signal

    def get_filtered_signal(self) -> Optional[np.ndarray]:
        """
        Get the filtered ECG signal from the last process() call.

        Returns
        -------
        Optional[np.ndarray]
            Filtered ECG signal (output from algorithm.filter()), or None if
            process() has not been called yet.
        """
        return self._filtered_signal

    def get_r_peaks(self) -> Optional[np.ndarray]:
        """
        Get the detected R-peaks from the last process() call.

        Returns
        -------
        Optional[np.ndarray]
            Array of R-peak sample indices, or None if process() has not been
            called yet or if R-peak detection failed.
        """
        return self._r_peaks

    def get_config(self) -> Dict[str, Any]:
        """
        Get a dictionary representation of the ECG sensor configuration.

        This delegates to the algorithm's get_config() method, which returns
        a dictionary (or dataclass __dict__) of all configuration parameters
        used by the algorithm.

        Returns
        -------
        Dict[str, Any]
            Configuration parameters dictionary. Typical keys include:
              - "sampling_rate" : int (Hz)
              - "lowpass_freq" : int (Hz)
              - "highpass_freq" : int (Hz)
              - "butter_order" : int
              - "mpd" : int (minimum peak distance)
              - "discard_window" : float (seconds)
              - ... (other ECG_Config attributes)

        Notes
        -----
        - The structure depends on the algorithm implementation and its
          configuration object.
        - Useful for logging configuration or validating settings.
        - Does not modify any state; purely informational.

        Examples
        --------
        ecg = ECG(config=my_config)
        config = ecg.get_config()
        print(f"Sampling rate: {config['sampling_rate']} Hz")
        """
        return self.algorithm.get_config()

    def set_algorithm(self, algorithm: ECG_base) -> None:
        """
        Change the algorithm used for processing and reset cached results.

        Allows runtime swapping of algorithm implementations (e.g., from
        Pan-Tompkins to an alternative R-peak detection method). Resets
        cached results to avoid confusion from mixing outputs of different
        algorithms.

        Parameters
        ----------
        algorithm : ECG_base
            New algorithm instance (must implement the ECG_base interface).

        Notes
        -----
        - Cached results (_raw_signal, _filtered_signal, etc.) are cleared
          to avoid mixing outputs from different algorithms.
        - Callers should re-run process() after setting a new algorithm to
          generate fresh results.
        - The new algorithm's config may differ from the previous one; ensure
          compatibility if needed.

        Examples
        --------
        ecg = ECG(config=my_config)
        # ... process with default algorithm ...

        # Switch to a custom algorithm
        custom_algo = MyCustomECGAlgorithm(my_config)
        ecg.set_algorithm(custom_algo)

        # Re-process with the new algorithm
        result = ecg.process(new_signal)
        """
        self.algorithm = algorithm

        # Reset stored results to avoid mixing results produced by different
        # algorithm instances. Callers should re-run process() after setting a
        # new algorithm.
        self._raw_signal = None
        self._filtered_signal = None
        self._r_peaks = None
        self._rr_intervals = None
        self._rr_time = None
