from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class ECG_base(ABC):
    @abstractmethod
    def filter(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """
        Abstract method to filter the ECG signal.

        Parameters:
        signal (np.ndarray): The raw ECG signal to be filtered.

        Returns:
        np.ndarray: The filtered ECG signal.
        """
        pass

    def detect_r_peaks(self, filtered_data: np.ndarray, fs: int) -> np.ndarray:
        """Detect R-peaks in filtered ECG signal."""
        pass

    def calculate_heart_rate(self, r_peaks: np.ndarray, fs: int) -> np.ndarray:
        """Calculate the heart rate of the ECG signal."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the ECG sensor."""
        pass
