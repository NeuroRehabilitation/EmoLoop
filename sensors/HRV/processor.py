"""
sensors.HRV.processor
High-level HRV processing wrapper.

This module provides the HRV class which acts as a sensor-agnostic processor for
heart rate variability (HRV) analysis. It coordinates an algorithm implementation
(conforming to the HRV_base interface) and exposes a simple API to compute:
 - heart rate estimates
 - time-domain HRV features
 - frequency-domain HRV features (via PSD / spectral analysis)
 - non-linear HRV features

The HRV class itself is algorithm-independent: provide a custom algorithm that
implements the methods defined by `sensors.HRV.base.HRV_base` and the processor
will delegate the heavy lifting to that algorithm.

Notes:
 - Inputs (RR intervals and timestamps) are expected in seconds (float).
 - Many algorithms may accept RR intervals as seconds; check the algorithm
   implementation if you use different units (e.g., milliseconds).
 - The processor stores results of the last `process()` call in internal
   attributes and exposes getters to retrieve them.
"""

from typing import Dict, Any, Optional
import numpy as np
from sensors.HRV.config import HRV_Config
from sensors.HRV.base import HRV_base

# from sensors.HRV.algorithms import HRVAlgorithm  # if you need a default


class HRV:
    """
    High-level HRV processor that works with any algorithm implementing HRV_base.

    Responsibilities:
    - Validate inputs for processing.
    - Delegate computation of HRV metrics to the provided algorithm instance.
    - Store the most recent results and expose convenient getters.
    - Provide small helper wrappers to call common algorithm methods.

    This class is intentionally lightweight — it aims to be the glue between
    caller code (UI, data pipelines) and the algorithm implementations.

    Attributes
    ----------
    config : HRV_Config
        Configuration object containing parameters required by algorithms,
        for example `sampling_rate`, window sizes, frequency bands, etc.
    algorithm : HRV_base
        Instance of an algorithm implementing the `HRV_base` interface.
        The algorithm is responsible for actual HRV computation routines.
    _heart_rate : Optional[Dict[str, float]]
        Stored heart rate summary from the last `process()` call (if any).
        Typically may contain keys such as 'mean' or 'instant' depending on
        algorithm implementation.
    _freq : Optional[np.ndarray]
        Frequency axis from the last frequency analysis (Hz).
    _power : Optional[np.ndarray]
        Power spectral density values corresponding to `_freq`.
    _time_features : Optional[Dict[str, Any]]
        Time-domain HRV features from the last `process()` call.
    _freq_features : Optional[Dict[str, Any]]
        Frequency-domain HRV features from the last `process()` call.
    _nonlinear_features : Optional[Dict[str, Any]]
        Non-linear HRV features (e.g., Poincaré metrics) from the last call.

    Notes on algorithm swapping
    --------------------------
    You can instantiate HRV with a custom algorithm that implements `HRV_base`.
    If no algorithm is provided, a default `HRVAlgorithm` implementation (if
    available in `sensors.HRV.algorithms`) will be instantiated using `config`.
    """

    def __init__(
        self,
        algorithm: Optional[HRV_base] = None,
        config: Optional[HRV_Config] = None,
    ) -> None:
        """
        Initialize the HRV processor.

        Parameters
        ----------
        algorithm : Optional[HRV_base]
            Custom algorithm instance to use for processing. If None, the
            processor will attempt to instantiate the default algorithm
            class `HRVAlgorithm` from `sensors.HRV.algorithms`.
            The provided instance must implement the `HRV_base` interface.
        config : Optional[HRV_Config]
            Configuration object with algorithm/sensor parameters. If None,
            a default `HRV_Config()` instance will be created.

        Notes
        -----
        - The configuration is required by many algorithms (sampling rate,
          resampling frequency, PSD parameters, etc.). Ensure `config`
          contains expected fields for the chosen algorithm.
        - The processor does not itself perform signal preprocessing (e.g.
          peak detection) — it assumes RR intervals are already computed when
          provided to `process()`.
        """
        # Initialize config first (always needed)
        self.config = config or HRV_Config()

        # Instantiate algorithm (default: HRVAlgorithm)
        if algorithm is not None:
            self.algorithm = algorithm
        else:
            # Lazy import to avoid import cycles / heavy imports at module load
            from sensors.HRV.algorithms import HRVAlgorithm

            self.algorithm = HRVAlgorithm(self.config)

        # Store processing results (None until process() is called)
        # Note: rr_intervals and rr_time are explicitly tracked by getters but
        # are set only during process(). They are documented here for clarity.
        self._heart_rate: Optional[Dict[str, float]] = None
        self._freq: Optional[np.ndarray] = None
        self._power: Optional[np.ndarray] = None
        self._time_features: Optional[Dict[str, Any]] = None
        self._freq_features: Optional[Dict[str, Any]] = None
        self._nonlinear_features: Optional[Dict[str, Any]] = None

        # The processor stores the last rr_intervals / rr_time if provided by
        # the caller. These attributes may be None until process() is invoked.
        # (They are referenced by the getters below.)
        # Note: not all existing code sets these attributes explicitly; callers
        # should rely on the return value of process() or the getters after
        # a successful process() call.
        self._rr_intervals: Optional[np.ndarray] = None
        self._rr_time: Optional[np.ndarray] = None

    def process(
        self,
        rr_intervals: Optional[np.ndarray] = None,
        rr_time: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Process RR intervals and extract HRV metrics.

        Full processing pipeline:
          - Validate that RR intervals are provided.
          - Compute heart rate summary via algorithm.heart_rate().
          - Compute time-domain HRV features via algorithm.time_domain_features().
          - Compute frequency-domain representation via algorithm.frequencyAnalysis(),
            and then extract features via algorithm.frequency_domain_features().
          - Compute non-linear HRV features via algorithm.non_linear_features().
          - Store results in instance attributes and return a consolidated dict.

        Parameters
        ----------
        rr_intervals : Optional[np.ndarray]
            RR intervals in seconds (floats). Must be provided — the processor
            will raise ValueError if None. Shape should be (N,) for N intervals.
        rr_time : Optional[np.ndarray]
            Timestamps associated with `rr_intervals`, in seconds. Some
            algorithms require timestamps to produce an accurate PSD; if the
            algorithm can operate without timestamps, this may be None.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
              - "rr_intervals": np.ndarray (input rr_intervals)
              - "rr_time": np.ndarray or None (input rr_time)
              - "heart_rate": dict with heart rate summary (algorithm-specific)
              - "time_features": dict of time-domain HRV features
              - "frequency_features": dict of frequency-domain HRV features
              - "nonlinear_features": dict of non-linear HRV features

        Raises
        ------
        ValueError
            If `rr_intervals` is None (no valid input to compute HRV).
        """
        # Validate input
        if rr_intervals is None:
            raise ValueError("rr_intervals must be provided")

        # Store inputs for later retrieval via getters
        # Store a copy to avoid accidental mutation by callers
        try:
            self._rr_intervals = np.asarray(rr_intervals, dtype=float).copy()
        except Exception:
            # Fall back to raw assignment if conversion fails; callers should
            # provide numeric RR intervals to get meaningful results.
            self._rr_intervals = rr_intervals

        if rr_time is not None:
            try:
                self._rr_time = np.asarray(rr_time, dtype=float).copy()
            except Exception:
                self._rr_time = rr_time
        else:
            self._rr_time = None

        # Heart rate summary (algorithm-dependent format)
        heart_rate = self.algorithm.heart_rate(rr_intervals)
        self._heart_rate = heart_rate

        # Time-domain features (e.g., meanNN, SDNN, RMSSD, pNN50)
        time_features = self.algorithm.time_domain_features(rr_intervals)
        self._time_features = time_features

        # Frequency-domain: compute PSD (freq, power) then extract features
        freq, power = self.algorithm.frequencyAnalysis(rr_intervals, rr_time)
        self._freq = freq
        self._power = power

        freq_features = self.algorithm.frequency_domain_features(freq, power)
        self._freq_features = freq_features

        # Non-linear features (e.g., Poincaré SD1/SD2, entropy measures)
        nonlinear_features = self.algorithm.non_linear_features(rr_intervals)
        self._nonlinear_features = nonlinear_features

        # Return full results in a single dictionary for convenience
        return {
            "rr_intervals": rr_intervals,
            "rr_time": rr_time,
            "heart_rate": heart_rate,
            "time_features": time_features,
            "frequency_features": freq_features,
            "nonlinear_features": nonlinear_features,
        }

    # Convenience wrappers that delegate to algorithm
    # These small helper methods are useful when callers need only a single
    # computation step without running the entire pipeline above.

    def remove_ectopy_beats(self, rr_intervals: np.ndarray) -> np.ndarray:
        """Remove ectopic beats from RR intervals using the algorithm's method.

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in seconds.

        Returns
        -------
        np.ndarray
            RR intervals after ectopy removal (algorithm-dependent).
        """
        return self.algorithm.remove_ectopy_beats(rr_intervals)

    def time_domain_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """Compute time-domain HRV features.

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in seconds.

        Returns
        -------
        Dict[str, Any]
            Dictionary of time-domain measures (algorithm-defined keys).
        """
        return self.algorithm.time_domain_features(rr_intervals)

    def frequency_analysis(
        self, rr_intervals: np.ndarray, rr_time: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density (PSD) from RR intervals.

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in seconds.
        rr_time : np.ndarray
            Timestamps corresponding to rr_intervals in seconds.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (freqs, power) where:
              - freqs: 1D array of frequency bin centres (Hz)
              - power: 1D array of power spectral density values
        """
        return self.algorithm.frequencyAnalysis(rr_intervals, rr_time)

    def frequency_domain_features(
        self, freqs: np.ndarray, power: np.ndarray
    ) -> Dict[str, Any]:
        """Compute frequency-domain HRV features from a PSD.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency axis in Hz.
        power : np.ndarray
            PSD values corresponding to `freqs`.

        Returns
        -------
        Dict[str, Any]
            Dictionary of spectral features (e.g., LF power, HF power, LF/HF).
        """
        return self.algorithm.frequency_domain_features(freqs, power)

    def non_linear_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """Compute non-linear HRV features (algorithm-specific).

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in seconds.

        Returns
        -------
        Dict[str, Any]
            Dictionary of non-linear measures (e.g., entropy, Poincaré).
        """
        return self.algorithm.non_linear_features(rr_intervals)

    # Getter methods for stored results
    # These return the cached results from the last call to `process()`.

    def get_rr_intervals(self) -> Optional[np.ndarray]:
        """Get the RR intervals from the last `process()` call.

        Returns
        -------
        Optional[np.ndarray]
            RR intervals (seconds) or None if `process()` has not been called.
        """
        return self._rr_intervals

    def get_rr_time(self) -> Optional[np.ndarray]:
        """Get the RR timestamps from the last `process()` call.

        Returns
        -------
        Optional[np.ndarray]
            RR timestamps (seconds) or None if not available.
        """
        return self._rr_time

    def get_freq_power(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the frequency axis and PSD from the last `process()` call.

        Returns
        -------
        tuple
            (freqs, power) where each element may be None if frequency analysis
            has not been performed yet.
        """
        return self._freq, self._power

    def get_time_features(self) -> Optional[Dict[str, Any]]:
        """Get the time-domain HRV features from the last `process()` call."""
        return self._time_features

    def get_freq_features(self) -> Optional[Dict[str, Any]]:
        """Get the frequency-domain HRV features from the last `process()` call."""
        return self._freq_features

    def get_nonlinear_features(self) -> Optional[Dict[str, Any]]:
        """Get the non-linear HRV features from the last `process()` call."""
        return self._nonlinear_features

    def get_config(self) -> Dict[str, Any]:
        """Get a dictionary representation of the HRV sensor configuration.

        This delegates to the algorithm's `get_config()` method. The structure of
        the returned dict depends on the algorithm/config implementation.

        Returns
        -------
        Dict[str, Any]
            Configuration parameters used by the algorithm.
        """
        return self.algorithm.get_config()

    def set_algorithm(self, algorithm: HRV_base) -> None:
        """
        Change the algorithm used for processing and reset cached results.

        Parameters
        ----------
        algorithm : HRV_base
            New algorithm instance (must implement the HRV_base interface).
        """
        self.algorithm = algorithm

        # Reset stored results to avoid mixing results produced by different
        # algorithm instances. Callers should re-run `process()` after setting a
        # new algorithm.
        self._rr_intervals = None
        self._rr_time = None
        self._freq = None
        self._power = None
        self._time_features = None
        self._freq_features = None
        self._nonlinear_features = None
