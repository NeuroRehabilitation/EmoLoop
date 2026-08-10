from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class PPG_base(ABC):

    @abstractmethod
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Abstract method to filter the PPG signal.

        Parameters:
        signal (np.ndarray): The raw PPG signal to be filtered.

        Returns:
        np.ndarray: The filtered PPG signal.
        """
        pass

    @abstractmethod
    def detect_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Detect peaks in filtered PPG signal."""
        pass

    @abstractmethod
    def rr_intervals(self, peaksIndex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate RR intervals from peaks."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the PPG sensor."""
        pass
