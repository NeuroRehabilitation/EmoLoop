from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class RESP_base(ABC):
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """

        """
        pass

    def getRESPfeatures(self,signal:np.ndarray) -> Dict[str,Any]:
        """

        """
        pass

    def RESP_RRV(self,signal:np.ndarray) -> Dict[str,Any]:
        pass