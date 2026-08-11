
"""
sensors.ECG.lib_selector

ECG Algorithm Selector - a factory for selecting ECG algorithm implementations by name.

This module provides `ECGAlgorithmSelector`, a convenience factory that maps short
string keys to algorithm classes and instantiates them with an optional `ECG_Config`.
It is intended for use by UI code, CLI tools, tests, or any place where a human-
readable algorithm name (or a config entry) should map to a concrete algorithm
implementation.

Usage examples:
    selector = ECGAlgorithmSelector(config=my_config)
    algorithm = selector.select("pan_tompkins")  # returns an instance of PanTompkinsAlgorithm
    available = selector.list_available()
    # -> ["pan_tompkins"]

This pattern centralizes algorithm registration and makes it simple to:
  - List available algorithms for UI dropdowns or help text
  - Choose an algorithm from a configuration file or CLI argument
  - Inject custom algorithm implementations for testing
  - Extend the system with new algorithms without modifying caller code
"""

from typing import Dict, Any, List, Optional
from sensors.ECG.algorithms import PanTompkinsAlgorithm
from sensors.ECG.base import ECG_base
from sensors.ECG.config import ECG_Config


class ECGAlgorithmSelector:
    """
    Factory for selecting ECG algorithm implementations by name.

    The selector keeps an internal mapping `ALGORITHMS` from a lowercased
    algorithm name to the algorithm class (constructor). When `select()` is
    called with a name, it instantiates the corresponding class, passing the
    optional `ECG_Config` that was provided to the selector.

    This pattern centralizes algorithm registration and makes it simple to:
     - List available algorithms for a UI dropdown or CLI help,
     - Choose an algorithm from a configuration file or CLI argument,
     - Inject custom algorithm implementations for testing,
     - Extend the system with new algorithms without changing caller code.

    Attributes
    ----------
    ALGORITHMS : Dict[str, ECG_base]
        Class-level mapping of available algorithm names to their implementing
        classes. Keys should be lowercase strings (algorithm names); values are
        class objects that can be instantiated with a config parameter.
        Add additional algorithms here to make them selectable by name.

    _config : Optional[ECG_Config]
        Instance-level configuration object passed to algorithm constructors
        when `select()` is called. May be None if default config is desired.

    Examples
    --------
    Basic usage:
        selector = ECGAlgorithmSelector(config=my_config)
        algorithm = selector.select("pan_tompkins")
        # `algorithm` is an instance of PanTompkinsAlgorithm configured with my_config

    Listing available algorithms:
        selector = ECGAlgorithmSelector()
        print(selector.list_available())  # ["pan_tompkins"]

    Registering a custom algorithm (edit this module to add to ALGORITHMS):
        from sensors.ECG.base import ECG_base
        class CustomECG(ECG_base):
            # implement required interface...
            pass
        ECGAlgorithmSelector.ALGORITHMS["custom"] = CustomECG

    Using in a UI or config loader:
        config_dict = load_config("config.yaml")
        algorithm_name = config_dict.get("algorithm", "pan_tompkins")
        selector = ECGAlgorithmSelector(config=ECG_Config(...))
        algorithm = selector.select(algorithm_name)
    """

    # Available algorithms (add more here). Keys should be lowercase names.
    ALGORITHMS: Dict[str, ECG_base] = {
        "pan_tompkins": PanTompkinsAlgorithm,
        # "custom": CustomAlgorithm,  # Add custom algorithms here
    }

    def __init__(self, config: Optional[ECG_Config] = None) -> None:
        """
        Initialize the selector with an optional configuration.

        Parameters
        ----------
        config : ECG_Config, optional
            Configuration instance that will be passed to algorithm constructors
            when `select()` is called. If `None`, algorithms should handle their
            own default configuration (or raise an error if config is required).

        Notes
        -----
        - The config is stored and reused for all algorithm instantiations.
        - Algorithms must accept a `config` parameter in their constructor
          (or have it as optional with a default).
        """
        self._config = config

    def select(self, library_name: str) -> ECG_base:
        """
        Instantiate and return an algorithm by its registered name.

        Looks up the algorithm class by name (case-insensitive), instantiates it
        with the stored configuration, and returns the instance.

        Parameters
        ----------
        library_name : str
            Name of the algorithm to select (case-insensitive). Must be one of
            the keys returned by `list_available()`. Examples: "pan_tompkins"

        Returns
        -------
        ECG_base
            An instance of the requested algorithm class, constructed with the
            selector's config (if provided).

        Raises
        ------
        ValueError
            If `library_name` is not found among registered algorithms. The error
            message includes a list of available algorithm names.

        Notes
        -----
        - Algorithm name matching is case-insensitive (normalized to lowercase).
        - The algorithm constructor must accept a `config` parameter.
        - If self._config is None, the algorithm should handle this gracefully
          (use defaults, raise an error, etc.).

        Examples
        --------
        selector = ECGAlgorithmSelector(config=my_config)
        algo = selector.select("pan_tompkins")
        # Returns an instance of PanTompkinsAlgorithm(config=my_config)

        Selecting with different case:
        algo = selector.select("PAN_TOMPKINS")  # Works (case-insensitive)
        """
        # Normalize to lowercase to make selection case-insensitive
        library_name_normalized = library_name.lower()

        if library_name_normalized not in self.ALGORITHMS:
            raise ValueError(
                f"Unsupported ECG library: {library_name}\n"
                f"Available algorithms: {self.list_available()}"
            )

        # Get algorithm class and instantiate it with the stored config
        algorithm_class = self.ALGORITHMS[library_name_normalized]
        return algorithm_class(config=self._config)

    def list_available(self) -> List[str]:
        """
        Return the list of registered algorithm names.

        Returns
        -------
        List[str]
            List of string keys that can be passed to `select()`. Names are
            lowercase and in arbitrary order (depends on dict iteration order).

        Notes
        -----
        - The returned list can be used to populate UI dropdowns or help text.
        - All strings in the list can be passed directly to `select()`.
        - To add new algorithms, update the class-level `ALGORITHMS` dict.

        Examples
        --------
        selector = ECGAlgorithmSelector()
        available = selector.list_available()
        print(available)  # ["pan_tompkins"]

        for algo_name in selector.list_available():
            print(f"  - {algo_name}")
        """
        return list(self.ALGORITHMS.keys())