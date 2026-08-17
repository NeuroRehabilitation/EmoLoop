from typing import Tuple, Dict, Any

import numpy as np
import scipy
from sensors.RESP.config import RESP_Config
from sensors.RESP.base import RESP_base
import neurokit2 as nk
import pandas as pd


class RESPAlgorithm(RESP_base):
    def __init__(self):
        self.config = RESP_Config()

    def convertRESP(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert the raw RESP signal to a standardized format.

        Parameters
        ----------
        signal : np.ndarray
            Raw RESP signal samples as a 1-D numpy array of floats.
            Expected shape: (n_samples,)

        Returns
        -------
        np.ndarray
            Converted RESP signal (1-D numpy array). Should have the same length
            as `signal` unless the implementation documents a different contract.
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
            Raw respiratory signal, for example from a respiration belt.
            Expected shape: (n_samples,).

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            A tuple containing:

            signals : pandas.DataFrame
                Processed respiratory signals, including RSP_Raw,
                RSP_Clean, RSP_Peaks, RSP_Troughs, RSP_Rate,
                RSP_Amplitude, RSP_Phase, RSP_Phase_Completion,
                and RSP_RVT.

            info : dict
                Information returned by NeuroKit2, including detected
                respiratory peaks and troughs.

        Notes
        -----
        The exact optional arguments supported by NeuroKit2 can vary by
        installed version. The method below uses the standard arguments
        shared by current releases.
        """

        signals, info = nk.rsp_process(
            signal,
            sampling_rate=self.config.sampling_rate,
            method=self.config.method,
        )

        return signals, info

    def getRespRate(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract respiratory rate.

        Parameters
        ----------
        signals : pandas.DataFrame
            Output from getRESPsignals().

        Returns
        -------
        dict
            Dictionary containing Respiration Rate Metrics.

        Raises
        ------
        KeyError
            If one or more required columns are unavailable.
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
