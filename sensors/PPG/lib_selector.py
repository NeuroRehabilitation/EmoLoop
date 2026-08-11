"""
ECG Algorithm Selector - Factory pattern for selecting algorithms by name.
"""

from typing import Dict, Any, List, Optional

from sensors.PPG.algorithms import PPGAlgorithm
from sensors.PPG.base import PPG_base
from sensors.PPG.config import PPG_Config


class PPGAlgorithmSelector:
    """
    Factory pattern for selecting PPG algorithms by name.

    Creates algorithm instances dynamically based on name.
    Useful for: UI dropdowns, config files, CLI arguments, testing.

    Attributes:
    -----------
    ALGORITHMS : Dict[str, PPG_base]
        Available algorithms (key: name, value: class)

    Example:
    --------
        # Basic usage
        selector = PPGAlgorithmSelector(config=my_config)
        algorithm = selector.select("pan_tompkins")
        ppg = PPG(algorithm=algorithm)

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
    ALGORITHMS: Dict[str, PPG_base] = {
        "ppg_algorithm": PPGAlgorithm,
        # "custom": CustomAlgorithm,  # Add custom algorithms here
    }

    def __init__(self, config: Optional[PPG_Config] = None) -> None:
        """
        Initialize the algorithm selector.

        Parameters:
        -----------
        config : PPG_Config, optional
            Configuration for algorithms. If None, uses default config.
        """
        self._config = config

    def select(self, library_name: str) -> PPG_base:
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
            selector = PPGAlgorithmSelector()
            algorithm = selector.select("ppg_algorithm")
        """
        # Normalize to lowercase
        library_name_normalized = library_name.lower()

        if library_name_normalized not in self.ALGORITHMS:
            raise ValueError(
                f"Unsupported PPG library: {library_name}\n"
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
            selector = PPGAlgorithmSelector()
            available = selector.list_available()
            print(available)  # ["ppg_algorithm"]
        """
        return list(self.ALGORITHMS.keys())
