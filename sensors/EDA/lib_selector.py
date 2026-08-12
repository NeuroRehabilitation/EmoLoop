"""
sensors.EDA.lib_selector
------------------------

Selector utility for choosing an EDA (Electrodermal Activity) algorithm
implementation at runtime.

This module provides the EDAAlgorithmSelector class which centralizes logic for
selecting and instantiating a concrete EDA processing algorithm implementation
from a registry. The selector makes it easy to switch algorithms by name (case
insensitive) and to pass a shared configuration object to the chosen algorithm.

Intended usage:
    from sensors.EDA.lib_selector import EDAAlgorithmSelector
    from sensors.EDA.config import EDA_Config

    cfg = EDA_Config(sampling_rate=250)
    selector = EDAAlgorithmSelector(config=cfg)
    algo = selector.select("eda_algorithm")  # returns an instantiated algorithm
    available = selector.list_available()

Notes:
- Algorithm registry keys should be lowercase and descriptive.
- Registry values are expected to be classes implementing the EDA_base API and
  constructible with a single `config` keyword argument (see sensors.EDA.base).
"""

from typing import Dict, Any, List, Optional
from sensors.EDA.algorithms import EDAAlgorithm
from sensors.EDA.base import EDA_base
from sensors.EDA.config import EDA_Config


class EDAAlgorithmSelector:
    """
    A small factory/registry to select and instantiate EDA algorithm classes.

    Responsibilities:
    - Maintain a mapping of lowercase library/algorithm names to algorithm
      classes.
    - Provide a simple case-insensitive selection API.
    - Instantiate the selected algorithm, passing through the stored config.

    Attributes:
        ALGORITHMS (Dict[str, EDA_base]):
            Registry mapping algorithm name (lowercase) to the algorithm
            class. Each class should be a subclass of `EDA_base` and accept a
            `config` keyword argument in its constructor.

            Example:
                "eda_algorithm": EDAAlgorithm
                # "custom": CustomEDA  # If you add a custom implementation

        _config (Optional[EDA_Config]):
            Optional configuration object that will be forwarded to algorithm
            instances on selection. Having a central config ensures consistent
            settings across different algorithm implementations.
    """

    # Available algorithms (add more here). Keys should be lowercase names.
    # NOTE: Values are algorithm classes (types) not algorithm instances.
    ALGORITHMS: Dict[str, EDA_base] = {
        "eda_algorithm": EDAAlgorithm,
        # "custom": CustomEDA,  # Add custom algorithms here
    }

    def __init__(self, config: Optional[EDA_Config] = None) -> None:
        """
        Initialize the selector.

        Args:
            config: Optional EDA_Config instance to pass to created algorithm
                    instances. If omitted, algorithms are constructed with
                    None (or handle defaults internally).
        """
        self._config = config

    def select(self, library_name: str) -> EDA_base:
        """
        Select and instantiate an algorithm by name.

        Selection is case-insensitive. If the requested algorithm name is not
        registered, a ValueError is raised listing the available options.

        Args:
            library_name: Name of the algorithm to select (e.g., "eda_algorithm").

        Returns:
            An instance of the selected algorithm (subclass of EDA_base),
            constructed with the selector's stored config.

        Raises:
            ValueError: If `library_name` is not in the registry.

        Example:
            selector = EDAAlgorithmSelector(config=cfg)
            algorithm = selector.select("eda_algorithm")
        """
        # Normalize to lowercase to make selection case-insensitive
        library_name_normalized = library_name.lower()

        if library_name_normalized not in self.ALGORITHMS:
            raise ValueError(
                f"Unsupported EDA library: {library_name}\n"
                f"Available algorithms: {self.list_available()}"
            )

        # Get algorithm class and instantiate it, passing the stored config
        algorithm_class = self.ALGORITHMS[library_name_normalized]
        return algorithm_class(config=self._config)

    def list_available(self) -> List[str]:
        """
        Return a list of available algorithm keys.

        The returned keys match the registry keys (all lowercase). This method
        is useful for error messages, debugging, or exposing available options
        to a higher-level UI.

        Returns:
            List[str]: Registered algorithm names.
        """
        return list(self.ALGORITHMS.keys())
