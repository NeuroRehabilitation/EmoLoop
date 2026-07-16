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
    def rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """Calculate RR intervals from R-peaks."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the ECG sensor."""
        pass
