"""
ECG Algorithm Selector - Factory pattern for selecting algorithms by name.
"""

from typing import Dict, Any, List, Optional

class ECGAlgorithmSelector:
    """
    Factory pattern for selecting ECG algorithms by name.

    Creates algorithm instances dynamically based on name.
    Useful for: UI dropdowns, config files, CLI arguments, testing.

    Attributes:
    -----------
    ALGORITHMS : Dict[str, ECG_base]
        Available algorithms (key: name, value: class)

    Example:
    --------
        # Basic usage
        selector = ECGAlgorithmSelector(config=my_config)
        algorithm = selector.select("pan_tompkins")
        ecg = ECG(sampling_rate=250, algorithm=algorithm)

        # List available algorithms
        available = selector.list_available()
        print(f"Available: {available}")  # ["pan_tompkins"]

        # Add custom algorithm
        from sensors.ECG.ECG_base import ECG_base
        class CustomAlgorithm(ECG_base):
            # Implement all methods
            pass

        selector.add_algorithm("custom", CustomAlgorithm)
        algorithm = selector.select("custom")
    """

    # Available algorithms (add more here)
    ALGORITHMS: Dict[str, ECG_base] = {
        "pan_tompkins": PanTompkinsAlgorithm,
        # "custom": CustomAlgorithm,  # Add custom algorithms here
    }

    def __init__(self, config: Optional[ECG_Config] = None) -> None:
        """
        Initialize the algorithm selector.

        Parameters:
        -----------
        config : ECG_Config, optional
            Configuration for algorithms. If None, uses default config.
        """
        self._config = config

    def select(self, library_name: str) -> ECG_base:
        """
        Select ECG algorithm by library name.

        Parameters:
        -----------
        library_name : str
            Name of the algorithm (e.g., "pan_tompkins")

        Returns:
        --------
        ECG_base
            Algorithm instance

        Raises:
        -------
        ValueError
            If library_name is not supported

        Example:
        --------
            selector = ECGAlgorithmSelector()
            algorithm = selector.select("pan_tompkins")
        """
        # Normalize to lowercase
        library_name_normalized = library_name.lower()

        if library_name_normalized not in self.ALGORITHMS:
            raise ValueError(
                f"Unsupported ECG library: {library_name}\n"
                f"Available algorithms: {self.list_available()}"
            )

        # Get algorithm class and instantiate
        algorithm_class = self.ALGORITHMS[library_name_normalized]
        return algorithm_class(config=self._config)

    def list_available(self) -> List[str]:
        """
        List available algorithm names.

        Returns:
        --------
        List[str]
            List of algorithm names

        Example:
        --------
            selector = ECGAlgorithmSelector()
            available = selector.list_available()
            print(available)  # ["pan_tompkins"]
        """
        return list(self.ALGORITHMS.keys())