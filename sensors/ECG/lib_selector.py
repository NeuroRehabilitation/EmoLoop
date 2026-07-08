from typing import Dict, Any, Optional
from base import ECG_base
from algorithms import (
    PanTompkinsAlgorithm,
)

class ECGAlgorithmSelector:
    """Selects ECG processing algorithm based on library name."""

    ALGORITHMS: Dict[str, ECG_base] = {
        "pan_tompkins": PanTompkinsAlgorithm,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config

    def select(self, library_name: str) -> ECG_base:
        """Select ECG algorithm by library name.

        Args:
            library_name: Name of the algorithm library.

        Returns:
            ECG algorithm instance.

        Raises:
            ValueError: If library not supported.
        """
        if library_name not in self.ALGORITHMS:
            raise ValueError(f"Unsupported ECG library: {library_name}")

        return self.ALGORITHMS[library_name](config=self._config)

    def list_available(self) -> list:
        """List available algorithms."""
        return list(self.ALGORITHMS.keys())