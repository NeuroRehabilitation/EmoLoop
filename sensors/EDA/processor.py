"""
sensors.EDA.processor

High-level EDA signal processing wrapper.

This module provides the EDA class, which acts as a sensor-agnostic processor for
electrodermal activity (EDA) analysis. It coordinates an algorithm implementation
(conforming to the EDA_base interface) and exposes a simple API to:
  - Filter raw EDA signals to remove noise and baseline drift
  - Decompose EDA into tonic and phasic components
  - Detect skin conductance responses (SCR) in the phasic waveform
  - Calculate SCR time-domain metrics
  - Extract skin conductance level (SCL) metrics
  - Compute frequency-domain EDA features

The EDA class itself is algorithm-independent: provide a custom algorithm that
implements the methods defined by `sensors.EDA.base.EDA_base` and the processor
will delegate the heavy lifting to that algorithm.

Notes
-----
- Raw EDA signals are typically in arbitrary ADC units or microsiemens;
  conversion to physical units is handled by the algorithm's convertEDA() method.
- The sampling_rate and other parameters are read from config (not separate parameters).
- The processor stores results of the last process() call in internal attributes
  and exposes getters to retrieve them.
- SCR peak detection and tonic/phasic decomposition are central to EDA analysis.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from sensors.EDA.config import EDA_Config
from sensors.EDA.algorithms import EDAAlgorithm
from sensors.EDA.base import EDA_base


class EDA:
    """
    High-level EDA processor that works with any algorithm implementing EDA_base.

    This class provides a unified interface for processing electrodermal activity (EDA)
    signals through a multi-stage pipeline. It abstracts away algorithm-specific
    implementation details while maintaining full flexibility through pluggable algorithm
    instances.

    Attributes
    ----------
    config : EDA_Config
        Configuration object containing sampling rate and processing parameters.
    sampling_rate : float
        Sampling rate of the EDA signal in Hz, derived from config.
    algorithm : EDA_base
        The algorithm instance used to perform actual signal processing computations.

    Responsibilities
    -----------------
    - Validate inputs for processing.
    - Delegate computation of EDA metrics to the provided algorithm instance.
    - Store the most recent results and expose convenient getters.
    - Provide small helper wrappers to call common algorithm methods.

    Processing Pipeline (in process() method)
    ------------------------------------------
      1. Filter raw EDA signal to remove noise and baseline drift
      2. Decompose into tonic and phasic components
      3. Detect SCR peaks and extract SCR features
      4. Extract SCL features
      5. Compute frequency-domain features
      6. Store results internally for later retrieval
    """

    def __init__(
        self, algorithm: Optional[EDA_base] = None, config: Optional[EDA_Config] = None
    ) -> None:
        """
        Initialize the EDA processor.

        Parameters
        ----------
        algorithm : EDA_base, optional
            Custom algorithm implementing the EDA_base interface. If None, defaults to
            EDAAlgorithm(). The algorithm instance performs all signal processing operations.
        config : EDA_Config, optional
            Configuration object containing processing parameters (e.g., sampling_rate).
            If None, defaults to EDA_Config() with default parameters.

        Returns
        -------
        None

        Notes
        -----
        All internal signal storage attributes are initialized to None and will be
        populated when process() is called.
        """
        self.config = config or EDA_Config()
        self.sampling_rate = self.config.sampling_rate

        if algorithm is not None:
            self.algorithm = algorithm
        else:
            self.algorithm = EDAAlgorithm()

        # Internal storage for signal processing results
        self._raw_signal: Optional[np.ndarray] = None
        self._filtered_signal: Optional[np.ndarray] = None
        self._tonic: Optional[np.ndarray] = None
        self._phasic: Optional[np.ndarray] = None
        self._scr_features: Optional[Dict[str, Any]] = None
        self._scl_features: Optional[Dict[str, Any]] = None
        self._freqs: Optional[np.ndarray] = None
        self._power: Optional[np.ndarray] = None
        self._freq_features: Optional[Dict[str, Any]] = None

    def process(
        self, signal: np.ndarray, compute_frequency: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the complete EDA processing pipeline on a raw signal.

        This method applies the full processing pipeline: filtering, decomposition,
        SCR detection, SCL extraction, and optional frequency-domain analysis.
        Results are stored internally for later retrieval via getter methods.

        Parameters
        ----------
        signal : np.ndarray
            Raw EDA signal as a 1D numpy array. Typically in ADC units or microsiemens.
            Must be non-empty.
        compute_frequency : bool, default=True
            If True, performs frequency-domain analysis and computes frequency features.
            If False, skips frequency analysis for faster processing.

        Returns
        -------
        dict
            Dictionary containing all processing results with the following keys:
            - 'raw_signal' : np.ndarray
                The input signal as stored internally.
            - 'filtered_signal' : np.ndarray
                Noise-filtered EDA signal.
            - 'phasic' : np.ndarray
                Phasic (rapidly changing) component of EDA.
            - 'tonic' : np.ndarray
                Tonic (slowly changing) component of EDA.
            - 'scr_features' : dict
                Time-domain metrics computed from phasic component (e.g., peak amplitude,
                rise time, recovery time).
            - 'scl_features' : dict
                Metrics extracted from tonic component (e.g., mean level, variance).
            - 'freqs' : np.ndarray or None
                Frequency array from frequency-domain analysis. None if compute_frequency=False.
            - 'power' : np.ndarray or None
                Power spectral density values. None if compute_frequency=False.
            - 'freq_features' : dict or None
                Frequency-domain feature metrics. None if compute_frequency=False.

        Raises
        ------
        ValueError
            If signal is None or empty.

        See Also
        --------
        get_raw_signal, get_filtered_signal, get_phasic, get_tonic : Retrieve individual results
        """
        if signal is None or len(signal) == 0:
            raise ValueError("Signal must be a non-empty numpy array")

        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        self._raw_signal = signal

        filtered_signal = self.algorithm.filter(signal)
        self._filtered_signal = filtered_signal

        phasic, tonic = self.algorithm.get_componentsEDA(filtered_signal)
        self._phasic = phasic
        self._tonic = tonic

        scr_features = self.algorithm.getSCRfeatures(phasic)
        self._scr_features = scr_features

        scl_features = self.algorithm.getSCLfeatures(tonic)
        self._scl_features = scl_features

        if compute_frequency:
            freqs, power = self.algorithm.frequencyAnalysis(filtered_signal)
            self._freqs = freqs
            self._power = power
            freq_features = self.algorithm.frequency_domain_features(freqs, power)
            self._freq_features = freq_features
        else:
            freqs = None
            power = None
            freq_features = None
            self._freqs = None
            self._power = None
            self._freq_features = None

        return {
            "raw_signal": signal,
            "filtered_signal": filtered_signal,
            "phasic": phasic,
            "tonic": tonic,
            "scr_features": scr_features,
            "scl_features": scl_features,
            "freqs": freqs,
            "power": power,
            "freq_features": freq_features,
        }

    def convertEDA(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert raw EDA signal from ADC units to physical units (e.g., microsiemens).

        This method delegates to the algorithm's convertEDA() implementation, which
        applies the appropriate scaling and conversion based on hardware parameters.

        Parameters
        ----------
        signal : np.ndarray
            Raw EDA signal in ADC units.

        Returns
        -------
        np.ndarray
            EDA signal converted to physical units (e.g., microsiemens).

        See Also
        --------
        algorithm.convertEDA : The underlying algorithm implementation.
        """
        return self.algorithm.convertEDA(signal)

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply filtering to remove noise and baseline drift from raw EDA signal.

        This is a convenience wrapper around the algorithm's filter method.
        Typically applies lowpass filtering to remove high-frequency noise and
        may include detrending to remove slow baseline drift.

        Parameters
        ----------
        signal : np.ndarray
            Raw EDA signal to filter.

        Returns
        -------
        np.ndarray
            Filtered EDA signal with same length as input.

        See Also
        --------
        algorithm.filter : The underlying algorithm implementation.
        """
        return self.algorithm.filter(signal)

    def decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose EDA signal into tonic and phasic components.

        The phasic component represents rapid changes associated with skin conductance
        responses (SCRs), while the tonic component represents the slower baseline level.

        Parameters
        ----------
        signal : np.ndarray
            Filtered EDA signal to decompose.

        Returns
        -------
        tuple
            - phasic : np.ndarray
                Phasic (rapidly changing) component.
            - tonic : np.ndarray
                Tonic (slowly changing) baseline component.

        Notes
        -----
        The signal should typically be filtered before decomposition for best results.

        See Also
        --------
        algorithm.get_componentsEDA : The underlying algorithm implementation.
        """
        return self.algorithm.get_componentsEDA(signal)

    def detect_scr(self, phasic: np.ndarray) -> Dict[str, Any]:
        """
        Detect skin conductance responses (SCRs) and extract time-domain features.

        SCRs are the rapid peaks visible in the phasic component. This method detects
        these peaks and computes metrics such as peak amplitude, rise time, and
        recovery time.

        Parameters
        ----------
        phasic : np.ndarray
            Phasic component of EDA signal, typically obtained from decompose().

        Returns
        -------
        dict
            Dictionary of SCR features. Keys typically include:
            - 'peaks' : array of peak indices or values
            - 'amplitudes' : array of peak amplitudes
            - 'rise_times' : array of rise times
            - 'recovery_times' : array of recovery times
            Other keys depend on the algorithm implementation.

        See Also
        --------
        algorithm.getSCRfeatures : The underlying algorithm implementation.
        """
        return self.algorithm.getSCRfeatures(phasic)

    def extract_scl(self, tonic: np.ndarray) -> Dict[str, Any]:
        """
        Extract skin conductance level (SCL) metrics from the tonic component.

        SCL represents the baseline level of skin conductance and its variability.
        This method computes metrics like mean level, standard deviation, and other
        tonic component statistics.

        Parameters
        ----------
        tonic : np.ndarray
            Tonic component of EDA signal, typically obtained from decompose().

        Returns
        -------
        dict
            Dictionary of SCL features. Keys typically include:
            - 'mean' : mean skin conductance level
            - 'std' : standard deviation
            - 'min' : minimum value
            - 'max' : maximum value
            Other keys depend on the algorithm implementation.

        See Also
        --------
        algorithm.getSCLfeatures : The underlying algorithm implementation.
        """
        return self.algorithm.getSCLfeatures(tonic)

    def frequency_analysis(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform frequency-domain analysis on EDA signal.

        Computes the power spectral density (PSD) of the signal using methods such
        as Welch's method or FFT-based approaches.

        Parameters
        ----------
        signal : np.ndarray
            EDA signal (typically filtered) to analyze.

        Returns
        -------
        tuple
            - freqs : np.ndarray
                Frequency array in Hz.
            - power : np.ndarray
                Power spectral density corresponding to each frequency.

        See Also
        --------
        algorithm.frequencyAnalysis : The underlying algorithm implementation.
        """
        return self.algorithm.frequencyAnalysis(signal)

    def frequency_features(
        self, freqs: np.ndarray, power: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute frequency-domain feature metrics from power spectral density.

        Extracts features such as dominant frequency, spectral entropy, and band powers
        (e.g., power in typical EDA frequency bands).

        Parameters
        ----------
        freqs : np.ndarray
            Frequency array (Hz), typically from frequency_analysis().
        power : np.ndarray
            Power spectral density values, typically from frequency_analysis().

        Returns
        -------
        dict
            Dictionary of frequency-domain features. Keys may include:
            - 'dominant_freq' : frequency with highest power
            - 'spectral_entropy' : Shannon entropy of the spectrum
            - Band powers (e.g., 'low_freq_power', 'high_freq_power')
            Other keys depend on the algorithm implementation.

        See Also
        --------
        algorithm.frequency_domain_features : The underlying algorithm implementation.
        """
        return self.algorithm.frequency_domain_features(freqs, power)

    def get_raw_signal(self) -> Optional[np.ndarray]:
        """
        Retrieve the raw input signal from the last process() call.

        Returns
        -------
        np.ndarray or None
            The raw signal, or None if process() has not been called.
        """
        return self._raw_signal

    def get_filtered_signal(self) -> Optional[np.ndarray]:
        """
        Retrieve the filtered signal from the last process() call.

        Returns
        -------
        np.ndarray or None
            The filtered signal with noise and drift removed, or None if process()
            has not been called.
        """
        return self._filtered_signal

    def get_phasic(self) -> Optional[np.ndarray]:
        """
        Retrieve the phasic component from the last process() call.

        The phasic component contains rapid changes associated with skin conductance
        responses (SCRs).

        Returns
        -------
        np.ndarray or None
            The phasic component, or None if process() has not been called.
        """
        return self._phasic

    def get_tonic(self) -> Optional[np.ndarray]:
        """
        Retrieve the tonic component from the last process() call.

        The tonic component represents the slowly changing baseline level of
        skin conductance.

        Returns
        -------
        np.ndarray or None
            The tonic component, or None if process() has not been called.
        """
        return self._tonic

    def get_scr_features(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve skin conductance response (SCR) features from the last process() call.

        Returns
        -------
        dict or None
            Dictionary of SCR metrics (e.g., peak amplitude, rise time), or None
            if process() has not been called.
        """
        return self._scr_features

    def get_scl_features(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve skin conductance level (SCL) features from the last process() call.

        Returns
        -------
        dict or None
            Dictionary of SCL metrics (e.g., mean level, standard deviation), or None
            if process() has not been called.
        """
        return self._scl_features

    def get_freqs(self) -> Optional[np.ndarray]:
        """
        Retrieve frequency array from frequency-domain analysis.

        Returns
        -------
        np.ndarray or None
            Frequency array (Hz) from the last process() call, or None if
            compute_frequency=False or process() has not been called.
        """
        return self._freqs

    def get_power(self) -> Optional[np.ndarray]:
        """
        Retrieve power spectral density from frequency-domain analysis.

        Returns
        -------
        np.ndarray or None
            Power spectral density values from the last process() call, or None if
            compute_frequency=False or process() has not been called.
        """
        return self._power

    def get_freq_features(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve frequency-domain feature metrics from the last process() call.

        Returns
        -------
        dict or None
            Dictionary of frequency-domain features (e.g., dominant frequency, spectral
            entropy), or None if compute_frequency=False or process() has not been called.
        """
        return self._freq_features

    def get_config(self) -> Dict[str, Any]:
        """
        Retrieve the current configuration used by the algorithm.

        Returns
        -------
        dict
            Configuration dictionary containing algorithm parameters such as
            sampling rate and processing options.

        See Also
        --------
        algorithm.get_config : The underlying algorithm method.
        """
        return self.algorithm.get_config()

    def set_algorithm(self, algorithm: EDA_base) -> None:
        """
        Replace the current algorithm with a new one and reset stored results.

        This method allows switching between different EDA processing algorithms
        at runtime. All previously stored results are cleared to prevent mixing
        results from different algorithms.

        Parameters
        ----------
        algorithm : EDA_base
            New algorithm instance implementing the EDA_base interface.

        Returns
        -------
        None

        Notes
        -----
        All internal signal storage attributes are reset to None after algorithm
        replacement. Call process() to generate new results with the new algorithm.
        """
        self.algorithm = algorithm
        self._raw_signal = None
        self._filtered_signal = None
        self._tonic = None
        self._phasic = None
        self._scr_features = None
        self._scl_features = None
        self._freqs = None
        self._power = None
        self._freq_features = None
