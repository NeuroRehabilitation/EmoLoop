
"""
sensors.HRV.lib_selector

HRV Algorithm Selector - a small factory for selecting HRV algorithm implementations by name.

This module provides `HRVAlgorithmSelector`, a convenience factory that maps short
string keys to algorithm classes and instantiates them with an optional `HRV_Config`.
It is intended for use by UI code, CLI tools, tests or any place where a human-
readable algorithm name (or a config entry) should map to a concrete algorithm
implementation.

Usage examples:
    selector = HRVAlgorithmSelector(config=my_config)
    algorithm = selector.select("hrv_algorithm")  # returns an instance of HRVAlgorithm
    available = selector.list_available()
    # -> ["hrv_algorithm"]
"""

from typing import Dict, Any, List, Optional
from sensors.HRV.algorithms import HRVAlgorithm
from sensors.HRV.base import HRV_base
from sensors.HRV.config import HRV_Config


class HRVAlgorithmSelector:
    """
    Factory for selecting HRV algorithm implementations by name.

    The selector keeps an internal mapping `ALGORITHMS` from a lowercased
    algorithm name to the algorithm class (constructor). When `select()` is
    called with a name it instantiates the corresponding class, passing the
    optional `HRV_Config` that was provided to the selector.

    This pattern centralizes algorithm registration and makes it simple to:
     - list available algorithms for a UI dropdown,
     - choose an algorithm from a configuration file or CLI argument,
     - inject custom algorithm implementations for testing.

    Attributes
    ----------
    ALGORITHMS : Dict[str, HRV_base]
        Mapping of available algorithm names to their implementing classes.
        Add additional algorithms here to make them selectable by name.

    Examples
    --------
    Basic usage:
        selector = HRVAlgorithmSelector(config=my_config)
        algorithm = selector.select("hrv_algorithm")
        # `algorithm` is an instance of the class mapped to "hrv_algorithm"

    Listing available algorithms:
        selector = HRVAlgorithmSelector()
        print(selector.list_available())  # e.g. ["hrv_algorithm"]

    Registering a custom algorithm (edit this module to add it to ALGORITHMS):
        from sensors.HRV.base import HRV_base
        class CustomHRV(HRV_base):
            # implement required interface...
            pass
        HRVAlgorithmSelector.ALGORITHMS["custom"] = CustomHRV
    """

    # Available algorithms (add more here). Keys should be lowercase names.
    ALGORITHMS: Dict[str, HRV_base] = {
        "hrv_algorithm": HRVAlgorithm,
        # "custom": CustomHRV,  # Add custom algorithms here
    }

    def __init__(self, config: Optional[HRV_Config] = None) -> None:
        """
        Initialize the selector with an optional configuration.

        Parameters
        ----------
        config : HRV_Config, optional
            Configuration instance that will be passed to algorithm constructors
            when `select()` is called. If `None`, the algorithm constructors
            should create or use their own default config.
        """
        self._config = config

    def select(self, library_name: str) -> HRV_base:
        """
        Instantiate and return an algorithm by its registered name.

        Parameters
        ----------
        library_name : str
            Name of the algorithm to select (case-insensitive). Must be one of
            the keys returned by `list_available()`.

        Returns
        -------
        HRV_base
            An instance of the requested algorithm class, constructed with the
            selector's config.

        Raises
        ------
        ValueError
            If `library_name` is not found among registered algorithms.

        Examples
        --------
        selector = HRVAlgorithmSelector(config=my_config)
        algo = selector.select("hrv_algorithm")
        """
        # Normalize to lowercase to make selection case-insensitive
        library_name_normalized = library_name.lower()

        if library_name_normalized not in self.ALGORITHMS:
            raise ValueError(
                f"Unsupported HRV library: {library_name}\n"
                f"Available algorithms: {self.list_available()}"
            )

        # Get algorithm class and instantiate it, passing the stored config
        algorithm_class = self.ALGORITHMS[library_name_normalized]
        return algorithm_class(config=self._config)

    def list_available(self) -> List[str]:
        """
        Return the list of registered algorithm names.

        Returns
        -------
        List[str]
            Sorted or unsorted list of string keys that can be passed to `select()`.

        Example
        -------
        selector = HRVAlgorithmSelector()
        print(selector.list_available())  # ["hrv_algorithm"]
        """
        return list(self.ALGORITHMS.keys())