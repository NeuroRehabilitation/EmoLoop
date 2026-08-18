"""
sensors.RESP.processor

High-level respiratory signal-processing wrapper.

This module provides the RESP class, which acts as a sensor-agnostic
processor for respiratory signal analysis. It coordinates an algorithm
implementation conforming to the RESP_base interface and exposes a simple API
to:

  - Convert raw respiratory ADC samples into voltage values.
  - Process respiratory signals using NeuroKit2.
  - Extract processed respiratory signals.
  - Calculate respiratory-rate metrics.
  - Calculate respiratory-amplitude metrics.
  - Calculate respiratory-volume-per-time (RVT) metrics.
  - Calculate respiratory-rate variability (RRV) metrics.
  - Calculate respiratory-amplitude variability (RAV) metrics.

The RESP class itself is algorithm-independent. A custom algorithm can be
provided as long as it implements the methods defined by sensors.RESP.base.RESP_base.

Notes
-----
- Raw respiratory signals may be represented in ADC units or another
  device-specific scale.
- ADC-to-voltage conversion is delegated to the algorithm implementation.
- The sampling rate and other processing parameters are obtained from the
  configuration object.
- The processor stores the results from the most recent process() call.
- NeuroKit2's rsp_process() returns sample-aligned respiratory signals and
  event metadata, including respiratory peaks and troughs.
"""

from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from sensors.RESP.config import RESP_Config
from sensors.RESP.algorithms import RESPAlgorithm
from sensors.RESP.base import RESP_base


class RESP:
    """
    High-level respiratory signal processor.

    The RESP class provides a lightweight interface between application code
    and a respiratory-processing algorithm. It delegates signal conversion,
    respiratory processing, and feature extraction to the configured algorithm.

    Responsibilities
    -----------------
    - Validate respiratory-signal inputs.
    - Delegate respiratory processing to the configured algorithm.
    - Store the most recent processing results.
    - Expose convenient getters for processed signals and extracted metrics.
    - Support runtime replacement of the respiratory-processing algorithm.

    Processing pipeline
    -------------------
    The default ``process()`` pipeline performs the following operations:

    1. Convert the raw respiratory signal using ``algorithm.convertRESP()``.
    2. Process the converted signal using ``algorithm.getRESPsignals()``.
    3. Extract respiratory-rate metrics using ``algorithm.getRespRate()``.
    4. Extract respiratory-amplitude metrics using
       ``algorithm.getRespAmplitude()``.
    5. Extract RVT metrics using ``algorithm.getRespRVT()``.
    6. Optionally calculate RRV and RAV metrics.

    Attributes
    ----------
    config : RESP_Config
        Configuration object containing respiratory sensor and processing
        parameters.

    sampling_rate : int or float
        Sampling frequency of the respiratory signal in Hz. This value is
        obtained from ``config.sampling_rate``.

    algorithm : RESP_base
        Algorithm instance responsible for respiratory signal conversion,
        processing, and feature extraction.

    _raw_signal : Optional[np.ndarray]
        Raw respiratory signal supplied during the most recent ``process()``
        call.

    _converted_signal : Optional[np.ndarray]
        Respiratory signal after ADC or sensor conversion.

    _signals : Optional[pd.DataFrame]
        Sample-aligned respiratory signals returned by NeuroKit2.

    _info : Optional[Dict[str, Any]]
        Metadata returned by NeuroKit2, including respiratory peak and trough
        sample indices.

    _resp_rate : Optional[Dict[str, Any]]
        Summary respiratory-rate metrics from the most recent processing call.

    _resp_amplitude : Optional[Dict[str, Any]]
        Summary respiratory-amplitude metrics from the most recent processing
        call.

    _resp_rvt : Optional[Dict[str, Any]]
        Summary RVT metrics from the most recent processing call.

    _rrv : Optional[pd.DataFrame]
        Respiratory-rate-variability metrics from the most recent processing
        call.

    _rav : Optional[pd.DataFrame]
        Respiratory-amplitude-variability metrics from the most recent
        processing call.

    Notes on algorithm swapping
    ---------------------------
    A custom algorithm implementing ``RESP_base`` can be passed during
    initialization or later through ``set_algorithm()``.

    If no algorithm is provided, a default ``RESPAlgorithm`` instance is
    created.
    """

    def __init__(
        self,
        algorithm: Optional[RESP_base] = None,
        config: Optional[RESP_Config] = None,
    ) -> None:
        """
        Initialize the RESP processor.

        Parameters
        ----------
        algorithm : Optional[RESP_base]
            Custom respiratory-processing algorithm. If ``None``, a default
            ``RESPAlgorithm`` instance is created.

        config : Optional[RESP_Config]
            Configuration object containing sensor-conversion parameters,
            sampling rate, and NeuroKit2 processing settings. If ``None``, a
            default ``RESP_Config`` instance is created.

        Notes
        -----
        When a custom algorithm is provided, its own configuration is used by
        the algorithm methods. The processor configuration is primarily used
        for the processor's sampling-rate attribute and default algorithm
        construction.
        """
        self.config = config or RESP_Config()
        self.sampling_rate = self.config.sampling_rate

        if algorithm is not None:
            self.algorithm = algorithm
        else:
            self.algorithm = RESPAlgorithm()

        self._raw_signal: Optional[np.ndarray] = None
        self._converted_signal: Optional[np.ndarray] = None
        self._signals: Optional[pd.DataFrame] = None
        self._info: Optional[Dict[str, Any]] = None
        self._resp_rate: Optional[Dict[str, Any]] = None
        self._resp_amplitude: Optional[Dict[str, Any]] = None
        self._resp_rvt: Optional[Dict[str, Any]] = None
        self._rrv: Optional[pd.DataFrame] = None
        self._rav: Optional[pd.DataFrame] = None

    def process(
        self,
        signal: np.ndarray,
        calculate_rrv: bool = True,
        calculate_rav: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a raw respiratory signal and extract respiratory metrics.

        Parameters
        ----------
        signal : np.ndarray
            Raw respiratory signal as a one-dimensional array.

            The input may contain raw ADC values or another device-specific
            representation accepted by ``algorithm.convertRESP()``.

        calculate_rrv : bool, default=True
            Whether to calculate respiratory-rate-variability metrics.

        calculate_rav : bool, default=True
            Whether to calculate respiratory-amplitude-variability metrics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the processing results:

            - ``raw_signal`` : np.ndarray
                Original input respiratory signal.
            - ``converted_signal`` : np.ndarray
                Converted respiratory signal.
            - ``signals`` : pandas.DataFrame
                Sample-aligned signals returned by NeuroKit2.
            - ``info`` : dict
                Respiratory-event metadata returned by NeuroKit2.
            - ``resp_rate`` : dict
                Summary respiratory-rate metrics.
            - ``resp_amplitude`` : dict
                Summary respiratory-amplitude metrics.
            - ``resp_rvt`` : dict
                Summary RVT metrics.
            - ``rrv`` : pandas.DataFrame or None
                RRV metrics, when ``calculate_rrv`` is ``True``.
            - ``rav`` : pandas.DataFrame or None
                RAV metrics, when ``calculate_rav`` is ``True``.

        Raises
        ------
        ValueError
            If ``signal`` is ``None``, empty, or not one-dimensional.

        Notes
        -----
        The processed ``signals`` DataFrame contains sample-by-sample outputs,
        while the metric dictionaries and variability DataFrames contain
        segment-level summaries.
        """

        if signal is None:
            raise ValueError("Signal must not be None")

        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)

        if signal.size == 0:
            raise ValueError("Signal must be non-empty")

        if signal.ndim != 1:
            raise ValueError(
                "Respiratory signal must be a one-dimensional array"
            )

        self._raw_signal = signal

        self._converted_signal = self.algorithm.convertRESP(signal)

        self._signals, self._info = self.algorithm.getRESPsignals(
            self._converted_signal
        )

        self._resp_rate = self.algorithm.getRespRate(
            self._signals
        )

        self._resp_amplitude = self.algorithm.getRespAmplitude(
            self._signals
        )

        self._resp_rvt = self.algorithm.getRespRVT(
            self._signals
        )

        self._rrv = None
        self._rav = None

        if calculate_rrv:
            self._rrv = self.algorithm.getRRV(
                self._signals,
                self._info,
            )

        if calculate_rav:
            self._rav = self.algorithm.getRAV(
                self._signals,
                self._info,
            )

        return {
            "raw_signal": self._raw_signal,
            "converted_signal": self._converted_signal,
            "signals": self._signals,
            "info": self._info,
            "resp_rate": self._resp_rate,
            "resp_amplitude": self._resp_amplitude,
            "resp_rvt": self._resp_rvt,
            "rrv": self._rrv,
            "rav": self._rav,
        }

    def convert(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a raw respiratory signal without processing it.

        Parameters
        ----------
        signal : np.ndarray
            Raw respiratory signal.

        Returns
        -------
        np.ndarray
            Converted respiratory signal.
        """
        return self.algorithm.convertRESP(signal)

    def process_signals(
        self,
        signal: np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a respiratory signal without calculating summary metrics.

        Parameters
        ----------
        signal : np.ndarray
            Converted or raw respiratory signal accepted by the algorithm.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Processed respiratory signals and NeuroKit2 metadata.
        """
        return self.algorithm.getRESPsignals(signal)

    def get_resp_rate(
        self,
        signals: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get respiratory-rate metrics.

        Parameters
        ----------
        signals : Optional[pd.DataFrame]
            Processed respiratory signals. If ``None``, the signals from the
            most recent ``process()`` call are used.

        Returns
        -------
        Optional[Dict[str, Any]]
            Respiratory-rate metrics, or ``None`` if no processing result is
            available.
        """
        if signals is not None:
            return self.algorithm.getRespRate(signals)

        return self._resp_rate

    def get_resp_amplitude(
        self,
        signals: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get respiratory-amplitude metrics.

        Parameters
        ----------
        signals : Optional[pd.DataFrame]
            Processed respiratory signals. If ``None``, the signals from the
            most recent ``process()`` call are used.

        Returns
        -------
        Optional[Dict[str, Any]]
            Respiratory-amplitude metrics, or ``None`` if no processing result
            is available.
        """
        if signals is not None:
            return self.algorithm.getRespAmplitude(signals)

        return self._resp_amplitude

    def get_resp_rvt(
        self,
        signals: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get respiratory-volume-per-time metrics.

        Parameters
        ----------
        signals : Optional[pd.DataFrame]
            Processed respiratory signals. If ``None``, the signals from the
            most recent ``process()`` call are used.

        Returns
        -------
        Optional[Dict[str, Any]]
            RVT metrics, or ``None`` if no processing result is available.
        """
        if signals is not None:
            return self.algorithm.getRespRVT(signals)

        return self._resp_rvt

    def get_rrv(
        self,
        signals: Optional[pd.DataFrame] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get respiratory-rate-variability metrics.

        Parameters
        ----------
        signals : Optional[pd.DataFrame]
            Processed respiratory signals. If ``None``, the most recent
            processed signals are used.

        info : Optional[Dict[str, Any]]
            NeuroKit2 respiratory metadata. If ``None``, the metadata from the
            most recent ``process()`` call are used.

        Returns
        -------
        Optional[pandas.DataFrame]
            RRV metrics, or ``None`` if no processing result is available.
        """
        if signals is not None or info is not None:
            if signals is None or info is None:
                raise ValueError(
                    "Both signals and info must be provided together"
                )

            return self.algorithm.getRRV(signals, info)

        return self._rrv

    def get_rav(
        self,
        signals: Optional[pd.DataFrame] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get respiratory-amplitude-variability metrics.

        Parameters
        ----------
        signals : Optional[pd.DataFrame]
            Processed respiratory signals. If ``None``, the most recent
            processed signals are used.

        info : Optional[Dict[str, Any]]
            NeuroKit2 respiratory metadata. If ``None``, the metadata from the
            most recent ``process()`` call are used.

        Returns
        -------
        Optional[pandas.DataFrame]
            RAV metrics, or ``None`` if no processing result is available.
        """
        if signals is not None or info is not None:
            if signals is None or info is None:
                raise ValueError(
                    "Both signals and info must be provided together"
                )

            return self.algorithm.getRAV(signals, info)

        return self._rav

    def get_signals(self) -> Optional[pd.DataFrame]:
        """
        Get the processed respiratory signals from the latest process call.

        Returns
        -------
        Optional[pandas.DataFrame]
            Sample-aligned NeuroKit2 respiratory signals, or ``None`` if
            ``process()`` has not been called.
        """
        return self._signals

    def get_info(self) -> Optional[Dict[str, Any]]:
        """
        Get respiratory-event metadata from the latest process call.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing detected respiratory peak and trough
            locations, or ``None`` if ``process()`` has not been called.
        """
        return self._info

    def get_raw_signal(self) -> Optional[np.ndarray]:
        """
        Get the raw respiratory signal from the latest process call.

        Returns
        -------
        Optional[np.ndarray]
            Raw input signal, or ``None`` if ``process()`` has not been called.
        """
        return self._raw_signal

    def get_converted_signal(self) -> Optional[np.ndarray]:
        """
        Get the converted respiratory signal from the latest process call.

        Returns
        -------
        Optional[np.ndarray]
            Converted respiratory signal, or ``None`` if ``process()`` has not
            been called.
        """
        return self._converted_signal

    def get_config(self) -> Dict[str, Any]:
        """
        Get the respiratory configuration.

        Returns
        -------
        Dict[str, Any]
            Configuration values returned by the algorithm implementation.
        """
        return self.algorithm.get_config()

    def set_algorithm(
        self,
        algorithm: RESP_base,
    ) -> None:
        """
        Replace the respiratory-processing algorithm.

        Parameters
        ----------
        algorithm : RESP_base
            New algorithm implementing the ``RESP_base`` interface.

        Raises
        ------
        TypeError
            If ``algorithm`` is not an instance of ``RESP_base``.

        Notes
        -----
        Cached results are cleared after replacing the algorithm. The signal
        must be processed again to generate results using the new algorithm.
        """
        self._raw_signal = None
        self._converted_signal = None
        self._signals = None
        self._info = None
        self._resp_rate = None
        self._resp_amplitude = None
        self._resp_rvt = None
        self._rrv = None
        self._rav = None

