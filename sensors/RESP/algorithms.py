from typing import Tuple, Dict, Any

import numpy as np
import scipy
from sensors.RESP.config import RESP_Config
from sensors.RESP.base import RESP_base
import neurokit2 as nk
import pandas as pd


class RESPAlgorithm(RESP_base):
    """
    Concrete implementation of respiratory signal-processing algorithms.

    This class converts raw respiratory sensor values, processes respiratory
    signals using NeuroKit2, and extracts descriptive statistics for
    respiratory rate, respiratory amplitude, and respiratory volume per time.

    Attributes
    ----------
    config : RESP_Config
        Configuration object containing sensor-conversion parameters and
        respiratory-processing settings.

    Methods
    -------
    convertRESP(signal)
        Convert raw RESP ADC values into voltage.

    getRESPsignals(signal)
        Process the raw respiratory signal with NeuroKit2.

    getRespRate(signals)
        Extract summary statistics from the respiratory-rate signal.

    getRespAmplitude(signals)
        Extract summary statistics from the respiratory-amplitude signal.

    getRespRVT(signals)
        Extract summary statistics from the respiratory-volume-per-time signal.
    """

    def __init__(self):
        """
        Initialize the RESP algorithm.

        Creates a default RESP_Config instance containing the parameters
        required for respiratory signal conversion and NeuroKit2 processing.

        Returns
        -------
        None
        """
        self.config = RESP_Config()

    def convertRESP(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert the raw RESP signal to voltage.

        The conversion normalizes the raw ADC values, centers them around
        half of the ADC range, scales them by the reference voltage, and
        compensates for the configured sensor gain.

        Parameters
        ----------
        signal : np.ndarray
            Raw RESP signal samples represented as a one-dimensional NumPy
            array.

            Expected shape:
                ``(n_samples,)``

            Values are expected to correspond to the ADC range defined by
            ``config.resolution``.

        Returns
        -------
        np.ndarray
            Converted RESP signal in volts.

            The returned array has the same shape and length as ``signal``.

        Notes
        -----
        Configuration attributes used:

        - ``VCC``
        - ``resolution``
        - ``gain``
        """
        VCC = self.config.VCC
        resolution = self.config.resolution
        signal_V = ((signal / (2**resolution - 1)) - 1 / 2) * VCC / self.config.gain
        signal_V = np.asarray(signal_V)

        return signal_V

    def getRESPsignals(
        self,
        signal: np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process the raw respiratory signal with NeuroKit2.

        Parameters
        ----------
        signal : np.ndarray
            Raw respiratory signal, such as a signal acquired from a
            respiration belt.

            Expected shape:
                ``(n_samples,)``

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            A tuple containing two elements.

            signals : pandas.DataFrame
                Sample-aligned processed respiratory signals. Depending on the
                NeuroKit2 version and processing method, this DataFrame may
                contain:

                - ``RSP_Raw``: Raw respiratory signal.
                - ``RSP_Clean``: Cleaned respiratory signal.
                - ``RSP_Peaks``: Binary markers for detected respiratory peaks.
                - ``RSP_Troughs``: Binary markers for detected respiratory
                  troughs.
                - ``RSP_Rate``: Respiratory rate interpolated over time.
                - ``RSP_Amplitude``: Respiratory amplitude interpolated over
                  time.
                - ``RSP_Phase``: Respiratory phase.
                - ``RSP_Phase_Completion``: Completion of the current
                  respiratory phase, represented from 0 to 1.
                - ``RSP_RVT``: Respiratory volume per time.

            info : dict
                Metadata returned by NeuroKit2. This dictionary contains
                respiratory-event locations, including:

                - ``info["RSP_Peaks"]``: Sample indices of detected peaks.
                - ``info["RSP_Troughs"]``: Sample indices of detected troughs.
                - ``info["sampling_rate"]``: Sampling rate used for processing.

        Notes
        -----
        The ``signals`` DataFrame is used for sample-by-sample respiratory
        analysis and visualization. The ``info`` dictionary is useful when
        accessing the exact sample positions of respiratory peaks and troughs.

        The processing parameters are obtained from ``self.config``:

        - ``sampling_rate``
        - ``method``
        """
        signals, info = nk.rsp_process(
            signal,
            sampling_rate=self.config.sampling_rate,
            method=self.config.method,
        )

        return signals, info

    def getRespRate(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for respiratory rate.

        Parameters
        ----------
        signals : pandas.DataFrame
            Processed respiratory signals returned by ``getRESPsignals()``.

            The DataFrame must contain the ``RSP_Rate`` column.

        Returns
        -------
        dict
            Dictionary containing the following respiratory-rate metrics:

            - ``Avg_Resp_Rate``: Average respiratory rate.
            - ``Min_Resp_Rate``: Minimum respiratory rate.
            - ``Max_Resp_Rate``: Maximum respiratory rate.
            - ``STD_Resp_Rate``: Standard deviation of respiratory rate.

            Respiratory-rate units are determined by NeuroKit2 and are
            typically breaths per minute.

        Raises
        ------
        KeyError
            Raised when the ``RSP_Rate`` column is not available in
            ``signals``.

        Notes
        -----
        NaN-safe NumPy functions are used so that missing values are ignored
        when calculating the summary statistics. If all values are NaN, the
        corresponding metrics are returned as ``np.nan``.
        """
        if "RSP_Rate" not in signals.columns:
            raise KeyError("Missing respiratory column: 'RSP_Rate'")

        rsp_rate = signals["RSP_Rate"]

        return {
            "Avg_Resp_Rate": (
                float(np.nanmean(rsp_rate)) if not np.isnan(rsp_rate).all() else np.nan
            ),
            "Min_Resp_Rate": (
                float(np.nanmin(rsp_rate)) if not np.isnan(rsp_rate).all() else np.nan
            ),
            "Max_Resp_Rate": (
                float(np.nanmax(rsp_rate)) if not np.isnan(rsp_rate).all() else np.nan
            ),
            "STD_Resp_Rate": (
                float(np.nanstd(rsp_rate)) if not np.isnan(rsp_rate).all() else np.nan
            ),
        }

    def getRespAmplitude(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for respiratory amplitude.

        Parameters
        ----------
        signals : pandas.DataFrame
            Processed respiratory signals returned by ``getRESPsignals()``.

            The DataFrame must contain the ``RSP_Amplitude`` column.

        Returns
        -------
        dict
            Dictionary containing the following respiratory-amplitude metrics:

            - ``Avg_Resp_Amplitude``: Average respiratory amplitude.
            - ``Min_Resp_Amplitude``: Minimum respiratory amplitude.
            - ``Max_Resp_Amplitude``: Maximum respiratory amplitude.
            - ``STD_Resp_Amplitude``: Standard deviation of respiratory
              amplitude.

            The amplitude units depend on the units of the input respiratory
            signal.

        Raises
        ------
        KeyError
            Raised when the ``RSP_Amplitude`` column is not available in
            ``signals``.

        Notes
        -----
        These metrics summarize the continuous ``RSP_Amplitude`` signal over
        the analyzed recording. They do not represent separate
        respiratory-amplitude-variability metrics calculated breath by breath.
        """
        if "RSP_Amplitude" not in signals.columns:
            raise KeyError("Missing respiratory column: 'RSP_Amplitude'")

        rsp_amp = signals["RSP_Amplitude"]

        return {
            "Avg_Resp_Amplitude": (
                float(np.nanmean(rsp_amp)) if not np.isnan(rsp_amp).all() else np.nan
            ),
            "Min_Resp_Amplitude": (
                float(np.nanmin(rsp_amp)) if not np.isnan(rsp_amp).all() else np.nan
            ),
            "Max_Resp_Amplitude": (
                float(np.nanmax(rsp_amp)) if not np.isnan(rsp_amp).all() else np.nan
            ),
            "STD_Resp_Amplitude": (
                float(np.nanstd(rsp_amp)) if not np.isnan(rsp_amp).all() else np.nan
            ),
        }

    def getRespRVT(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for respiratory volume per time.

        Parameters
        ----------
        signals : pandas.DataFrame
            Processed respiratory signals returned by ``getRESPsignals()``.

            The DataFrame must contain the ``RSP_RVT`` column.

        Returns
        -------
        dict
            Dictionary containing the following RVT metrics:

            - ``Avg_Resp_RVT``: Average respiratory volume per time.
            - ``Min_Resp_RVT``: Minimum respiratory volume per time.
            - ``Max_Resp_RVT``: Maximum respiratory volume per time.
            - ``STD_Resp_RVT``: Standard deviation of respiratory volume per
              time.

            RVT units depend on the calibration and units of the input
            respiratory signal.

        Raises
        ------
        KeyError
            Raised when the ``RSP_RVT`` column is not available in
            ``signals``.

        Notes
        -----
        The returned values summarize the sample-level ``RSP_RVT`` signal
        across the analyzed recording. RVT is provided by NeuroKit2 during
        respiratory signal processing.
        """
        if "RSP_RVT" not in signals.columns:
            raise KeyError("Missing respiratory column: 'RSP_RVT'")

        rsp_rvt = signals["RSP_RVT"]

        return {
            "Avg_Resp_RVT": (
                float(np.nanmean(rsp_rvt)) if not np.isnan(rsp_rvt).all() else np.nan
            ),
            "Min_Resp_RVT": (
                float(np.nanmin(rsp_rvt)) if not np.isnan(rsp_rvt).all() else np.nan
            ),
            "Max_Resp_RVT": (
                float(np.nanmax(rsp_rvt)) if not np.isnan(rsp_rvt).all() else np.nan
            ),
            "STD_Resp_RVT": (
                float(np.nanstd(rsp_rvt)) if not np.isnan(rsp_rvt).all() else np.nan
            ),
        }

    def getRAV(self,signals: pd.DataFrame, info: Dict[str, Any]) -> pd.DataFrame:
        """
            Calculate respiratory-amplitude-variability (RAV) metrics.

            Parameters
            ----------
            signals : pandas.DataFrame
                Processed respiratory signals returned by ``getRESPsignals()``.

                The DataFrame must contain the following column:

                - ``RSP_Amplitude``: Respiratory-amplitude signal interpolated over
                  the recording.

            info : Dict[str, Any]
                Metadata returned by NeuroKit2 during respiratory signal processing.

                The dictionary must contain the following keys:

                - ``RSP_Peaks``: Sample indices of detected respiratory peaks.
                - ``RSP_Troughs``: Sample indices of detected respiratory troughs.

            Returns
            -------
            pandas.DataFrame
                DataFrame containing respiratory-amplitude-variability metrics
                calculated by ``neurokit2.rsp_rav()``.

                Depending on the NeuroKit2 version, the returned DataFrame may
                contain amplitude mean, standard deviation, RMSSD, coefficient of
                variation, and other RAV-related measures.

            Raises
            ------
            KeyError
                Raised when ``RSP_Amplitude`` is not available in ``signals``.

            KeyError
                Raised when one or more required respiratory-event keys are not
                available in ``info``.

            Notes
            -----
            RAV quantifies variation in respiratory amplitude from breath to breath.
            The respiratory-amplitude signal is supplied through
            ``signals["RSP_Amplitude"]``.

            The detected respiratory peaks and troughs are supplied through
            ``info["RSP_Peaks"]`` and ``info["RSP_Troughs"]``. These event locations
            allow NeuroKit2 to calculate amplitude variability using respiratory
            cycles rather than treating every sample as an independent breath.

            The returned DataFrame generally contains segment-level RAV metrics rather
            than a sample-by-sample time series.
        """

        if "RSP_Amplitude" not in signals.columns:
            raise KeyError("Missing respiratory column: 'RSP_Amplitude'")

        required_keys = [
            "RSP_Peaks",
            "RSP_Troughs",
        ]

        missing_columns = [
            key
            for key in required_keys
            if key not in info
        ]

        if missing_columns:
            raise KeyError(
                f"Missing respiratory columns: {missing_columns}"
            )

        rsp_rav = nk.rsp_rav(signals["RSP_Amplitude"], peaks=info["RSP_Peaks"], troughs=info["RSP_Troughs"])

        return rsp_rav

    def getRRV(self, signals: pd.DataFrame, info: Dict[str,Any]) -> pd.DataFrame:
        """
            Calculate respiratory-rate-variability (RRV) metrics.

            Parameters
            ----------
            signals : pandas.DataFrame
                Processed respiratory signals returned by ``getRESPsignals()``.

                The DataFrame must contain the following column:

                - ``RSP_Rate``: Respiratory-rate signal interpolated over time.

            info : Dict[str, Any]
                Metadata returned by NeuroKit2 during respiratory signal processing.

                The dictionary must contain:

                - ``RSP_Troughs``: Sample indices corresponding to the detected
                  respiratory troughs.

            Returns
            -------
            pandas.DataFrame
                DataFrame containing respiratory-rate-variability metrics calculated
                by ``neurokit2.rsp_rrv()``.

                Depending on the NeuroKit2 version and the available respiratory
                events, the returned DataFrame may include time-domain, frequency-
                domain, and nonlinear RRV measures.

            Raises
            ------
            KeyError
                Raised when the ``RSP_Rate`` column is not available in ``signals``.

            KeyError
                Raised when ``RSP_Troughs`` is not available in ``info``.

            Notes
            -----
            RRV describes variation in the time intervals between consecutive
            respiratory events. The respiratory-rate signal is supplied through
            ``signals["RSP_Rate"]``, while the detected trough locations are supplied
            through ``info["RSP_Troughs"]``.

            The sampling rate is obtained from ``self.config.sampling_rate`` and is
            passed to NeuroKit2 for correct temporal interpretation of the signal
            and trough locations.

            The returned DataFrame generally contains one row of segment-level RRV
            metrics rather than a sample-by-sample time series.
        """

        if "RSP_Rate" not in signals.columns:
            raise KeyError("Missing respiratory column: 'RSP_Rate'")
        if "RSP_Troughs" not in info:
            raise KeyError("Missing respiratory troughs in info dictionary")

        rrv_dataframe = nk.rsp_rrv(signals["RSP_Rate"], troughs=info["RSP_Troughs"], sampling_rate=self.config.sampling_rate)

        return rrv_dataframe
