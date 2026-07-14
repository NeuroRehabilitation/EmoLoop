from typing import Dict, Any, Optional
import numpy as np
from config import ECG_Config
from algorithms import PanTompkinsAlgorithm


class ECG:
    """
    High-level ECG processor that works with any algorithm implementing ECG_base.

    Processes raw ECG signals and extracts heart rate metrics.
    Algorithm-independent: can swap PanTompkins for custom implementations.

    Note:
    -----
    The sampling_rate is read from config.sampling_rate (not a separate parameter).

    Attributes:
    -----------
    sampling_rate : int
        Sampling rate in Hz (from config.sampling_rate, default: 250)
    algorithm : ECG_base
        Algorithm instance for processing (default: PanTompkinsAlgorithm)
    config : ECG_Config
        Configuration including all sensor parameters
    """

    def __init__(
        self,
        algorithm: Optional[ECG_base] = None,
        config: Optional[ECG_Config] = None
    ) -> None:
        """
        Initialize the ECG processor.

        Parameters:
        -----------
        algorithm : ECG_base, optional
            Custom algorithm instance. If None, uses PanTompkinsAlgorithm.
        config : ECG_Config, optional
            Configuration for algorithm including sampling_rate. If None, uses default config.

        Note:
        -----
        The sampling_rate is read from config.sampling_rate (default: 250).
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
        self._heart_rate: Optional[Dict[str, Any]] = None

    def process(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Process raw ECG signal and extract heart rate metrics.

        Runs full pipeline: filter → detect R-peaks → calculate heart rate.

        Parameters:
        -----------
        signal : np.ndarray
            Raw ECG signal array (1D)

        Returns:
        --------
        Dict[str, Any]
            Dictionary with:
            - "raw_signal": original signal (np.ndarray)
            - "filtered_signal": filtered signal (np.ndarray)
            - "r_peaks": R-peak indices (np.ndarray)
            - "heart_rate": HR metrics dict (mean_hr, hr_std, hrv)
            - "sampling_rate": sampling rate used (int, from config)

        Raises:
        -------
        ValueError
            If signal is None or empty
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

        # Step 3: Calculate heart rate
        heart_rate = self.algorithm.calculate_heart_rate(r_peaks)
        self._heart_rate = heart_rate

        # Return full results
        return {
            "raw_signal": signal,
            "filtered_signal": filtered_signal,
            "r_peaks": r_peaks,
            "heart_rate": heart_rate,
            "sampling_rate": self.sampling_rate,
        }

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Filter raw ECG signal only (skip R-peak detection and heart rate).

        Parameters:
        -----------
        signal : np.ndarray
            Raw ECG signal

        Returns:
        --------
        np.ndarray
            Filtered ECG signal
        """
        return self.algorithm.filter(signal)

    def detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks only (skip filtering and heart rate calculation).

        Note: Should be called on filtered signal.

        Parameters:
        -----------
        signal : np.ndarray
            Filtered ECG signal

        Returns:
        --------
        np.ndarray
            Array of R-peak indices
        """
        return self.algorithm.detect_r_peaks(signal)

    def calculate_heart_rate(self, r_peaks: np.ndarray) -> Dict[str, Any]:
        """
        Calculate heart rate from R-peaks only.

        Parameters:
        -----------
        r_peaks : np.ndarray
            Array of R-peak indices

        Returns:
        --------
        Dict[str, float]
            Heart rate metrics:
            - "mean_hr": mean heart rate (bpm)
            - "hr_std": standard deviation of HR (bpm)
            - "hrv": heart rate variability (RR interval std, seconds)
        """
        return self.algorithm.calculate_heart_rate(r_peaks)

    # Getter methods for stored results

    def get_raw_signal(self) -> Optional[np.ndarray]:
        """Get the raw ECG signal from last process() call."""
        return self._raw_signal

    def get_filtered_signal(self) -> Optional[np.ndarray]:
        """Get the filtered ECG signal from last process() call."""
        return self._filtered_signal

    def get_r_peaks(self) -> Optional[np.ndarray]:
        """Get the detected R-peaks from last process() call."""
        return self._r_peaks

    def get_heart_rate(self) -> Optional[Dict[str, float]]:
        """Get the heart rate metrics from last process() call."""
        return self._heart_rate

    def get_config(self) -> Dict[str, Any]:
        """Get the ECG sensor configuration."""
        return self.algorithm.get_config()

    def set_algorithm(self, algorithm: ECG_base) -> None:
        """
        Change the algorithm used for processing.

        Parameters:
        -----------
        algorithm : ECG_base
            New algorithm instance (must implement ECG_base interface)
        """
        self.algorithm = algorithm

        # Reset stored results
        self._raw_signal = None
        self._filtered_signal = None
        self._r_peaks = None
        self._heart_rate = None
