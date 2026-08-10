from typing import Dict, Any

from sensors.PPG.base import PPG_base
from sensors.PPG.config import PPG_Config

import numpy as np
import scipy


class PPGAlgorithm(PPG_base):
    def __init__(self, config: PPG_Config):
        self.config = config

    @staticmethod
    def _butter_sos(
        filter_type: str,
        fs: float,
        order: int,
        lowcut: float | None = None,
        highcut: float | None = None,
    ):
        sos = scipy.signal.butter(
            order, [lowcut, highcut], btype=filter_type, fs=fs, output="sos"
        )
        return sos

    def filter(self, signal: np.ndarray) -> np.ndarray:
        sos = self._butter_sos(
            self.config.filter_type,
            self.config.sampling_rate,
            self.config.butter_order,
            lowcut=self.config.lowpass_freq,
            highcut=self.config.highpass_freq,
        )

        filtered_ppg = scipy.signal.sosfiltfilt(
            sos,
            signal,
        )

        return filtered_ppg

    @staticmethod
    def findPeaksPPG(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect all local maxima.

        Returns
        -------
        peaksAmp : np.ndarray
            Amplitudes of detected peaks.

        peaksIndex : np.ndarray
            Sample indices of detected peaks.
        """
        peaksIndex, _ = scipy.signal.find_peaks(signal)
        peaksAmp = signal[peaksIndex]

        return peaksAmp, peaksIndex

    @staticmethod
    def findValleysPPG(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect all local minima.

        Returns
        -------
        valleysAmp : np.ndarray
            Amplitudes of detected valleys.

        valleysIndex : np.ndarray
            Sample indices of detected valleys.
        """

        valleysIndex, _ = scipy.signal.find_peaks(-signal)
        valleysAmp = signal[valleysIndex]

        return valleysAmp, valleysIndex

    @staticmethod
    def pairPeakValleys(
        signal: np.ndarray, peaksIndex: np.ndarray, valleysIndex: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Pair each detected peak with the most recent preceding valley.

        No physiological constraints are applied.
        """

        pairedPeaks = []
        pairedValleys = []

        valleyPointer = 0
        latestValley = None

        for peakIndex in peaksIndex:
            while (
                valleyPointer < len(valleysIndex)
                and valleysIndex[valleyPointer] < peakIndex
            ):
                latestValley = valleysIndex[valleyPointer]
                valleyPointer += 1

            if latestValley is None:
                continue

            if pairedValleys and latestValley == pairedValleys[-1]:
                continue

            pairedPeaks.append(peakIndex)
            pairedValleys.append(latestValley)

        pairedPeaks = np.asarray(pairedPeaks, dtype=int)
        pairedValleys = np.asarray(pairedValleys, dtype=int)

        peaksAmp = signal[pairedPeaks]
        valleysAmp = signal[pairedValleys]

        return {
            "PeaksAmp": peaksAmp,
            "PairedPeaks": pairedPeaks,
            "ValleysAmp": valleysAmp,
            "PairedValleys": pairedValleys,
        }

    @staticmethod
    def peaks_valleyDiff(peaksAmp: np.ndarray, valleysAmp: np.ndarray) -> np.ndarray:
        """
        Calculate peak-to-valley amplitude differences.
        """

        return peaksAmp - valleysAmp

    @staticmethod
    def validatePeaksPPG(
        paired_data: Dict[str, np.ndarray],
        threshold: float = 0.7,
        window: int = 5,
    ) -> Dict[str, np.ndarray]:
        peaksAmp = paired_data["PeaksAmp"]
        peaksIndex = paired_data["PairedPeaks"]
        valleysAmp = paired_data["ValleysAmp"]
        valleysIndex = paired_data["PairedValleys"]

        valleyPeaksDiff = peaksAmp - valleysAmp

        if valleyPeaksDiff.size == 0:
            return {
                "PeaksAmp": peaksAmp,
                "PairedPeaks": peaksIndex,
                "ValleysAmp": valleysAmp,
                "PairedValleys": valleysIndex,
                "ValleyPeaksDiff": valleyPeaksDiff,
            }

        keep = np.zeros(valleyPeaksDiff.size, dtype=bool)
        half_window = window // 2

        for i in range(valleyPeaksDiff.size):
            start = max(0, i - half_window)
            stop = min(valleyPeaksDiff.size, i + half_window + 1)
            local_mean = np.mean(valleyPeaksDiff[start:stop])
            keep[i] = valleyPeaksDiff[i] > threshold * local_mean

        return {
            "PeaksAmp": peaksAmp[keep],
            "PeaksIndex": peaksIndex[keep],
            "ValleysAmp": valleysAmp[keep],
            "ValleysIndex": valleysIndex[keep],
            "ValleyPeaksDiff": valleyPeaksDiff[keep],
        }

    def detect_peaks(self, signal: np.ndarray) -> np.ndarray:
        filtered_signal = self.filter(signal)
        peaksAmp, peaksIndex = self.findPeaksPPG(filtered_signal)
        valleysAmp, valleysIndex = self.findValleysPPG(filtered_signal)
        paired_data = self.pairPeakValleys(filtered_signal, peaksIndex, valleysIndex)

        validated_data = self.validatePeaksPPG(
            paired_data=paired_data,
            threshold=0.7,
            window=5,
        )

        return validated_data["PeaksIndex"]

    def rr_intervals(self, peaks: np.ndarray) -> np.ndarray:
        # Implement RR interval calculation logic here
        pass

    def get_config(self) -> Dict[str, Any]:
        return self.config.__dict__
