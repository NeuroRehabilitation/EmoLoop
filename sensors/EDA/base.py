
"""
Module: sensors.EDA.base
Description: Abstract base class for Electrodermal Activity (EDA) signal processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class EDA_base(ABC):
    """
    Abstract base class for Electrodermal Activity (EDA) signal processing.

    This class defines the interface for EDA signal processing implementations.
    Subclasses must implement all abstract methods to provide a complete EDA
    processing pipeline for this project.

    Abstract Methods:
        - filter: Apply preprocessing and denoising to raw EDA signal
        - getSCRfeatures: Extract features from the phasic component
        - getSCLfeatures: Extract features from the tonic component
        - frequencyAnalysis: Perform frequency-domain analysis
        - frequency_domain_features: Extract frequency-domain features
        - get_config: Return runtime/configuration parameters

    Notes:
        Implementations should work with numpy arrays and avoid side-effects where
        possible (i.e., return new arrays rather than mutating inputs), unless
        documented otherwise by the concrete subclass.
    """

    @abstractmethod
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Filter the raw EDA signal.

        This method should perform any signal preprocessing required by the
        downstream algorithms (e.g., detrending, smoothing, resampling).

        Parameters
        ----------
        signal : np.ndarray
            Raw EDA signal samples as a 1-D numpy array of floats.
            Expected shape: (n_samples,)
            NaN handling, clipping and scaling policy should be documented
            by the concrete implementation.

        Returns
        -------
        np.ndarray
            Filtered EDA signal (1-D numpy array). Should have the same length
            as `signal` unless the implementation documents a different contract.

        Raises
        ------
        ValueError
            If input signal is not a 1-D array or contains invalid data.
        """
        pass

    def get_componentsEDA(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose the filtered EDA signal into phasic and tonic components.

        This method implements a decomposition algorithm (e.g., cvxEDA,
        deconvolution, or other suitable methods) to separate the phasic and
        tonic components of the EDA signal.

        Parameters
        ----------
        signal : np.ndarray
            Filtered EDA signal samples as a 1-D numpy array of floats.
            Expected shape: (n_samples,)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two 1-D numpy arrays:
            - phasic_component (np.ndarray): The phasic (rapid) component of the EDA signal
            - tonic_component (np.ndarray): The tonic (slow-changing) component of the EDA signal
            Both arrays have the same length as the input signal.

        Notes
        -----
        This method decomposes the EDA signal into its two main physiological components:
        - Phasic: Fast changes due to discrete skin conductance responses (SCRs)
        - Tonic: Slow changes representing baseline skin conductance level (SCL)
        """
        pass

    @abstractmethod
    def getSCRfeatures(self, phasic_component: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from the phasic component of the EDA signal.

        This method computes relevant features from the phasic component,
        such as skin conductance responses (SCRs), amplitude, latency, rise time,
        and other characteristics.

        Parameters
        ----------
        phasic_component : np.ndarray
            Phasic component of the EDA signal as a 1-D numpy array of floats.
            Expected shape: (n_samples,)

        Returns
        -------
        Dict[str, Any]
            A dictionary containing extracted features from the phasic component.
            Common keys may include:
            - 'scr_count': Number of skin conductance responses detected
            - 'scr_amplitude': Amplitude of SCRs
            - 'scr_latency': Latency of SCRs
            - 'scr_rise_time': Rise time of SCRs
            The specific keys and values should be documented by the concrete implementation.

        Notes
        -----
        The phasic component represents rapid changes in skin conductance,
        typically associated with emotional or cognitive responses.
        """
        pass

    @abstractmethod
    def getSCLfeatures(self, tonic_component: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from the tonic component of the EDA signal.

        This method computes relevant features from the tonic component,
        such as skin conductance level (SCL), baseline level, and other
        characteristics.

        Parameters
        ----------
        tonic_component : np.ndarray
            Tonic component of the EDA signal as a 1-D numpy array of floats.
            Expected shape: (n_samples,)

        Returns
        -------
        Dict[str, Any]
            A dictionary containing extracted features from the tonic component.
            Common keys may include:
            - 'scl_mean': Mean skin conductance level
            - 'scl_std': Standard deviation of SCL
            - 'scl_min': Minimum SCL value
            - 'scl_max': Maximum SCL value
            The specific keys and values should be documented by the concrete implementation.

        Notes
        -----
        The tonic component represents the slowly-changing baseline skin conductance,
        typically associated with sustained states of arousal or stress.
        """
        pass

    @abstractmethod
    def frequencyAnalysis(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform frequency-domain analysis on the filtered EDA signal.

        This method computes relevant frequency-domain features from the
        filtered EDA signal, such as power spectral density and frequency bands.

        Parameters
        ----------
        signal : np.ndarray
            Filtered EDA signal samples as a 1-D numpy array of floats.
            Expected shape: (n_samples,)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - freqs (np.ndarray): Frequency vector (Hz), shape: (n_freqs,)
            - power (np.ndarray): Power spectral density, shape: (n_freqs,)

        Notes
        -----
        The frequency analysis is typically performed using methods such as
        Fourier Transform or Welch's method to identify spectral characteristics
        of the EDA signal.
        """
        pass

    @abstractmethod
    def frequency_domain_features(
        self, freqs: np.ndarray, power: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract frequency-domain features from power spectral density.

        This method computes relevant frequency-domain features such as
        power in specific frequency bands, dominant frequencies, and other
        spectral characteristics.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency vector as a 1-D numpy array of floats.
            Expected shape: (n_freqs,)
            Units: Hz (Hertz)
        power : np.ndarray
            Power spectral density as a 1-D numpy array of floats.
            Expected shape: (n_freqs,)
            Should correspond to the frequency vector `freqs`.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing extracted frequency-domain features.
            Common keys may include:
            - 'power_low_freq': Power in low-frequency bands
            - 'power_high_freq': Power in high-frequency bands
            - 'freq_centroid': Centroid of the power spectrum
            - 'spectral_entropy': Entropy of the power spectrum
            The specific keys and values should be documented by the concrete implementation.

        Notes
        -----
        Frequency-domain features can provide insights into the spectral
        characteristics of EDA signals and may be useful for classification
        or pattern recognition tasks.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the runtime/configuration parameters used by this EDA processor.

        This method should return all configuration parameters and settings
        that were used to initialize and run the EDA processing pipeline.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the runtime/configuration parameters.
            Common keys may include:
            - 'sampling_rate': Sampling rate of the EDA signal (Hz)
            - 'filter_type': Type of filter applied
            - 'filter_params': Parameters for the filter
            - 'decomposition_method': Method used for component decomposition
            The specific keys and values should be documented by the concrete implementation.

        Notes
        -----
        This method is useful for reproducibility and debugging, as it allows
        external code to verify what settings were used to process the signal.
        """
        pass