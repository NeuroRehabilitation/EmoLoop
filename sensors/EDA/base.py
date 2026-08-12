from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class EDA_base(ABC):
    """
    Abstract base class for Electrodermal Activity (EDA) signal processing.

    Subclasses must implement the following methods to provide a complete EDA
    processing pipeline for this project:
      - filter: apply preprocessing and denoising to raw EDA signal
      - get_features: extract relevant features from the filtered signal
      - get_config: return the runtime/configuration parameters being used

    Implementations should work with numpy arrays and avoid side-effects where
    possible (i.e., return new arrays rather than mutating inputs), unless
    documented otherwise by the concrete subclass.
    """

    @abstractmethod
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Filter the raw EDA signal.

        This method should perform any signal preprocessing required by the
        downstream algorithms (e.g., detrending, smoothing, resampling). Implementations must accept a 1-D numpy array
        and return a 1-D numpy array of the same length representing the
        filtered signal.

        Parameters
        ----------
        signal : np.ndarray
            Raw EDA signal samples as a 1-D numpy array of floats. Expected
            shape is (n_samples,). NaN handling, clipping and scaling policy
            should be documented by the concrete implementation.

        Returns
        -------
        np.ndarray
            Filtered EDA signal (1-D numpy array). Should have the same length
            as `signal` unless the implementation documents a different contract.
        """
        pass

    def get_componentsEDA(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose the filtered EDA signal into phasic and tonic components.

        This method should implement a decomposition algorithm (e.g., cvxEDA,
        deconvolution, or other suitable methods) to separate the phasic and
        tonic components of the EDA signal. The output should be two 1-D numpy
        arrays representing the phasic and tonic components, respectively.

        Parameters
        ----------
        signal : np.ndarray
            Filtered EDA signal samples as a 1-D numpy array of floats. Expected
            shape is (n_samples,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two 1-D numpy arrays:
            - phasic_component: The phasic component of the EDA signal.
            - tonic_component: The tonic component of the EDA signal.
        """
        pass

    @abstractmethod
    def getSCRfeatures(self, phasic_component: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from the phasic component of the EDA signal.

        This method should compute relevant features from the phasic component,
        such as skin conductance responses (SCRs), amplitude, latency, rise time,
        and other characteristics. The output should be a dictionary containing
        these features.

        Parameters
        ----------
        phasic_component : np.ndarray
            Phasic component of the EDA signal as a 1-D numpy array of floats.
            Expected shape is (n_samples,).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing extracted features from the phasic component.
            The keys and values should be documented by the concrete implementation.
        """
        pass

    @abstractmethod
    def getSCLfeatures(self, tonic_component: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from the tonic component of the EDA signal.

        This method should compute relevant features from the tonic component,
        such as skin conductance level (SCL), baseline level, and other
        characteristics. The output should be a dictionary containing these
        features.

        Parameters
        ----------
        tonic_component : np.ndarray
            Tonic component of the EDA signal as a 1-D numpy array of floats.
            Expected shape is (n_samples,).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing extracted features from the tonic component.
            The keys and values should be documented by the concrete implementation.
        """
        pass

    @abstractmethod
    def frequencyAnalysis(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Perform frequency-domain analysis on the filtered EDA signal.

        This method should compute relevant frequency-domain features from the
        filtered EDA signal, such as power spectral density, frequency bands,
        and other spectral characteristics. The output should be a dictionary
        containing these features.

        Parameters
        ----------
        signal : np.ndarray
            Filtered EDA signal samples as a 1-D numpy array of floats. Expected
            shape is (n_samples,).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing extracted frequency-domain features from the EDA signal.
            The keys and values should be documented by the concrete implementation.
        """
        pass

    @abstractmethod
    def frequency_domain_features(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Extract frequency-domain features from a filtered EDA signal.

        This method should compute relevant frequency-domain features from the
        filtered EDA signal, such as power spectral density, frequency bands,
        and other spectral characteristics. The output should be a dictionary
        containing these features.

        Parameters
        ----------
        signal : np.ndarray
            Filtered EDA signal samples as a 1-D numpy array of floats. Expected
            shape is (n_samples,).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing extracted frequency-domain features from the EDA signal.
            The keys and values should be documented by the concrete implementation.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the runtime/configuration parameters.
        """
        pass
