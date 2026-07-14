from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class ECG_base(ABC):

    @abstractmethod
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Abstract method to filter the ECG signal.

        Parameters:
        signal (np.ndarray): The raw ECG signal to be filtered.

        Returns:
        np.ndarray: The filtered ECG signal.
        """
        pass

    @abstractmethod
    def detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Detect R-peaks in filtered ECG signal."""
        pass

    @abstractmethod
    def calculate_heart_rate(self, r_peaks: np.ndarray) -> Dict[str, Any]:
        """Calculate the heart rate of the ECG signal."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the ECG sensor."""
        pass
