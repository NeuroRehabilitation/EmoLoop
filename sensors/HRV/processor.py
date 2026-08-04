from typing import Dict, Any, Optional
import numpy as np
from sensors.HRV.config import HRV_Config
from sensors.HRV.base import HRV_base

# from sensors.HRV.algorithms import HRVAlgorithm  # if you need a default


class HRV:
    """
    High-level HRV processor that works with any algorithm implementing HRV_base.

    Processes RR interval series and extracts time-, frequency-, and non-linear
    HRV metrics. Algorithm-independent: can swap HRVAlgorithm for custom
    implementations.

    Note:
    -----
    The sampling_rate and other parameters are read from config (not separate
    parameters).

    Attributes:
    -----------
    sampling_rate : int
        Sampling rate in Hz (from config.sampling_rate, default: 250)
    algorithm : HRV_base
        Algorithm instance for processing (default: HRVAlgorithm)
    config : HRV_Config
        Configuration including all sensor parameters
    """

    def __init__(
        self,
        algorithm: Optional[HRV_base] = None,
        config: Optional[HRV_Config] = None,
    ) -> None:
        """
        Initialize the HRV processor.

        Parameters:
        -----------
        algorithm : HRV_base, optional
            Custom algorithm instance. If None, uses HRVAlgorithm.
        config : HRV_Config, optional
            Configuration for algorithm including sampling_rate and HRV params.
            If None, uses default config.

        Note:
        -----
        The sampling_rate is read from config.sampling_rate (default: 250).
        """
        # Initialize config first (always needed)
        self.config = config or HRV_Config()

        # Instantiate algorithm (default: HRVAlgorithm)
        if algorithm is not None:
            self.algorithm = algorithm
        else:
            from sensors.HRV.algorithms import HRVAlgorithm

            self.algorithm = HRVAlgorithm(self.config)

        # Store processing results (None until process() is called)
        self._heart_rate: Optional[Dict[str, float]] = None
        self._freq: Optional[np.ndarray] = None
        self._power: Optional[np.ndarray] = None
        self._time_features: Optional[Dict[str, Any]] = None
        self._freq_features: Optional[Dict[str, Any]] = None
        self._nonlinear_features: Optional[Dict[str, Any]] = None

    def process(
        self,
        rr_intervals: Optional[np.ndarray] = None,
        rr_time: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Process RR intervals and extract HRV metrics.

        Runs full pipeline:
          - optional ectopy removal
          - time-domain features
          - frequency-domain (via algorithm.frequencyAnalysis + features)
          - non-linear features

        Parameters:
        -----------
        rr_intervals : np.ndarray, optional
            RR intervals in seconds. If None, computed from r_peaks.
        rr_time : np.ndarray, optional
            Timestamps of RR intervals in seconds. If None, computed from r_peaks.
        r_peaks : np.ndarray, optional
            R-peak sample indices. Used to compute rr_intervals and rr_time if not provided.
        remove_ectopy : bool
            Whether to remove ectopic beats before computing features.

        Returns:
        --------
        Dict[str, Any]
            Dictionary with:
            - "rr_intervals": RR intervals (np.ndarray)
            - "rr_time": RR timestamps (np.ndarray)
            - "time_features": time-domain HRV features dict
            - "freq_features": frequency-domain HRV features dict
            - "nonlinear_features": non-linear HRV features dict

        Raises:
        -------
        ValueError
            If no valid input is provided
        """

        # Validate input
        if rr_intervals is None:
            raise ValueError("rr_intervals must be provided")

        heart_rate = self.algorithm.heart_rate(rr_intervals)
        self._heart_rate = heart_rate

        # Time-domain features
        time_features = self.algorithm.time_domain_features(rr_intervals)
        self._time_features = time_features

        # Frequency-domain
        freq, power = self.algorithm.frequencyAnalysis(rr_intervals, rr_time)
        self._freq = freq
        self._power = power

        freq_features = self.algorithm.frequency_domain_features(freq, power)
        self._freq_features = freq_features

        # Non-linear features
        nonlinear_features = self.algorithm.non_linear_features(rr_intervals)
        self._nonlinear_features = nonlinear_features

        # Return full results
        return {
            "rr_intervals": rr_intervals,
            "rr_time": rr_time,
            "heart_rate": heart_rate,
            "time_features": time_features,
            "frequency_features": freq_features,
            "nonlinear_features": nonlinear_features,
        }

    # Convenience wrappers that delegate to algorithm

    def remove_ectopy_beats(self, rr_intervals: np.ndarray) -> np.ndarray:
        """Remove ectopic beats from RR intervals."""
        return self.algorithm.remove_ectopy_beats(rr_intervals)

    def time_domain_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """Compute time-domain HRV features."""
        return self.algorithm.time_domain_features(rr_intervals)

    def frequency_analysis(
        self, rr_intervals: np.ndarray, rr_time: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute PSD from RR intervals."""
        return self.algorithm.frequencyAnalysis(rr_intervals, rr_time)

    def frequency_domain_features(
        self, freqs: np.ndarray, power: np.ndarray
    ) -> Dict[str, Any]:
        """Compute frequency-domain HRV features from PSD."""
        return self.algorithm.frequency_domain_features(freqs, power)

    def non_linear_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """Compute non-linear HRV features."""
        return self.algorithm.non_linear_features(rr_intervals)

    # Getter methods for stored results

    def get_rr_intervals(self) -> Optional[np.ndarray]:
        """Get the RR intervals from last process() call."""
        return self._rr_intervals

    def get_rr_time(self) -> Optional[np.ndarray]:
        """Get the RR timestamps from last process() call."""
        return self._rr_time

    def get_freq_power(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the frequency axis and PSD from last process() call."""
        return self._freq, self._power

    def get_time_features(self) -> Optional[Dict[str, Any]]:
        """Get the time-domain HRV features from last process() call."""
        return self._time_features

    def get_freq_features(self) -> Optional[Dict[str, Any]]:
        """Get the frequency-domain HRV features from last process() call."""
        return self._freq_features

    def get_nonlinear_features(self) -> Optional[Dict[str, Any]]:
        """Get the non-linear HRV features from last process() call."""
        return self._nonlinear_features

    def get_config(self) -> Dict[str, Any]:
        """Get the HRV sensor configuration."""
        return self.algorithm.get_config()

    def set_algorithm(self, algorithm: HRV_base) -> None:
        """
        Change the algorithm used for processing.

        Parameters:
        -----------
        algorithm : HRV_base
            New algorithm instance (must implement HRV_base interface)
        """
        self.algorithm = algorithm

        # Reset stored results
        self._rr_intervals = None
        self._rr_time = None
        self._freq = None
        self._power = None
        self._time_features = None
        self._freq_features = None
        self._nonlinear_features = None
