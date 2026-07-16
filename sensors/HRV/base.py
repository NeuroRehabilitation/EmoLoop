from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class HRV_base(ABC):
    @abstractmethod
    def remove_ectopy_beats(self, rr_intervals: np.ndarray) -> np.ndarray:
        """
        Abstract method to remove ectopic beats from RR intervals.
        Parameters:
        rr_intervals (np.ndarray): The RR intervals array.

        Returns:
        np.ndarray: The RR intervals array with ectopic beats removed.
        """
        pass

    @abstractmethod
    def heart_rate(self, rr_intervals: np.ndarray) -> float:
        """
        Abstract method to calculate heart rate from RR intervals.
        Parameters:
        rr_intervals (np.ndarray): The RR intervals array.

        Returns:
        float: The calculated heart rate in beats per minute (bpm).
        """
        pass

    @abstractmethod
    def time_domain_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """
        Abstract method to calculate time-domain features from RR intervals.
        Parameters:
        rr_intervals (np.ndarray): The RR intervals array.

        Returns:
        Dict[str, Any]: A dictionary containing time-domain features.
        """
        pass

    @abstractmethod
    def frequency_domain_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """
        Abstract method to calculate frequency-domain features from RR intervals.
        Parameters:
        rr_intervals (np.ndarray): The RR intervals array.

        Returns:
        Dict[str, Any]: A dictionary containing frequency-domain features.
        """
        pass

    @abstractmethod
    def non_linear_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """
        Abstract method to calculate non-linear features from RR intervals.
        Parameters:
        rr_intervals (np.ndarray): The RR intervals array.

        Returns:
        Dict[str, Any]: A dictionary containing non-linear features.
        """
        pass
