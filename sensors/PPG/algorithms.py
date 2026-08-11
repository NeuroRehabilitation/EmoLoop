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
        """
        _butter_sos(filter_type, fs, order, lowcut=None, highcut=None)

        Design a Butterworth filter and return second-order sections (SOS).

        Parameters
        ----------
        filter_type : str
            Filter type: "bandpass", "lowpass", "highpass", etc.

        fs : float
            Sampling frequency (Hz)

        order : int
            Filter order (higher = steeper rolloff, more computation)

        lowcut : float, optional
            Low cutoff frequency (Hz). Required for highpass and bandpass.

        highcut : float, optional
            High cutoff frequency (Hz). Required for lowpass and bandpass.

        Returns
        -------
        ndarray
            Second-order sections (SOS) format for scipy.signal.sosfiltfilt.
            More numerically stable than transfer function format.
        """
        sos = scipy.signal.butter(
            order, [highcut, lowcut], btype=filter_type, fs=fs, output="sos"
        )
        return sos

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        filter(signal)

        Apply bandpass Butterworth filter to PPG signal.

        Applies zero-phase filtering using scipy.signal.sosfiltfilt, which
        filters in both forward and reverse directions to eliminate phase distortion.

        Parameters
        ----------
        signal : np.ndarray
            Raw PPG signal (1D array)

        Returns
        -------
        np.ndarray
            Filtered signal (same length as input)

        Notes
        -----
        - Filter parameters (type, order, cutoff frequencies) come from self.config
        - Zero-phase filtering doubles the effective filter order
        """
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
        findPeaksPPG(signal)

        Detect all local maxima in a signal.

        Uses scipy.signal.find_peaks to identify local peaks without amplitude thresholding.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (typically filtered PPG)

        Returns
        -------
        peaksAmp : np.ndarray
            Amplitudes of detected peaks

        peaksIndex : np.ndarray
        Sample indices of detected peaks
        """
        peaksIndex, _ = scipy.signal.find_peaks(signal)
        peaksAmp = signal[peaksIndex]

        return peaksAmp, peaksIndex

    @staticmethod
    def findValleysPPG(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        findValleysPPG(signal)

        Detect all local minima in a signal.

        Finds valleys by inverting the signal and detecting peaks on -signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (typically filtered PPG)

        Returns
        -------
        valleysAmp : np.ndarray
            Amplitudes of detected valleys

        valleysIndex : np.ndarray
            Sample indices of detected valleys
        """

        valleysIndex, _ = scipy.signal.find_peaks(-signal)
        valleysAmp = signal[valleysIndex]

        return valleysAmp, valleysIndex

    @staticmethod
    def pairPeakValleys(
        signal: np.ndarray, peaksIndex: np.ndarray, valleysIndex: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """pairPeakValleys(signal, peaksIndex, valleysIndex)

        Pair each detected peak with the most recent preceding valley.

        Creates peak-valley pulse pairs by matching each peak to the last valley that
        occurs before it. Ensures each valley is used at most once and skips peaks
        without a preceding valley.

        Parameters
        ----------
        signal : np.ndarray
            The filtered PPG signal

        peaksIndex : np.ndarray
            Sample indices of detected peaks

        valleysIndex : np.ndarray
            Sample indices of detected valleys

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys:
            - "PeaksAmp": Peak amplitudes (paired subset)
            - "PairedPeaks": Peak indices (paired subset)
            - "ValleysAmp": Valley amplitudes (paired subset)
            - "PairedValleys": Valley indices (paired subset)

        Notes
        -----
        - No physiological constraints applied; purely temporal pairing
        - Requires alternating valley-peak pattern for correct pairing
        - Linear time complexity: O(n_peaks + n_valleys)
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
    def validatePeaksPPG(
        paired_data: Dict[str, np.ndarray],
        threshold,
        window,
    ) -> Dict[str, np.ndarray]:

        """
        Validate detected PPG peaks using adaptive local mean threshold filtering.

        This function filters peak-valley pairs by comparing each peak-valley difference
        to a threshold-scaled local mean of the differences. Peaks with low amplitude
        relative to their local neighborhood are rejected, helping eliminate false detections.

        Parameters
        ----------
        paired_data : dict[str, np.ndarray]
            Dictionary containing:
            - "PeaksAmp": Peak amplitudes
            - "PairedPeaks": Peak sample indices
            - "ValleysAmp": Valley amplitudes
            - "PairedValleys": Valley sample indices

        threshold : float
            Multiplier for the local mean. A peak is kept if:
            peak_valley_diff > threshold * local_mean

        window : int
            Window size (in samples) for computing local mean.
            The local mean is centered at each peak: [i - window//2, i + window//2]

        Returns
        -------
        dict[str, np.ndarray]
            Filtered peak-valley pairs:
            - "PeaksAmp": Validated peak amplitudes
            - "PeaksIndex": Validated peak indices
            - "ValleysAmp": Corresponding valley amplitudes
            - "ValleysIndex": Corresponding valley indices
            - "ValleyPeaksDiff": Peak-valley amplitude differences for validated pairs

        Notes
        -----
        - Uses a sliding window mean to adapt thresholding to local signal variability
        - Window boundaries are clipped (no zero-padding)
        - Returns all data unchanged if no peaks are detected
        """

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
        """detect_peaks(signal)

        Complete peak detection pipeline: filter → detect → validate.

        Performs end-to-end PPG peak detection: applies filtering, finds peaks and valleys,
        pairs them, and validates pairs using adaptive thresholding.

        Parameters
        ----------
        signal : np.ndarray
            Raw PPG signal

        Returns
        -------
        np.ndarray
            Sample indices of validated peaks (ready for RR interval calculation)

        Notes
        -----
        - Uses configuration from self.config for all parameters
        - Returns only peaks that pass validation threshold
        """
        filtered_signal = self.filter(signal)
        peaksAmp, peaksIndex = self.findPeaksPPG(filtered_signal)
        valleysAmp, valleysIndex = self.findValleysPPG(filtered_signal)
        paired_data = self.pairPeakValleys(filtered_signal, peaksIndex, valleysIndex)

        validated_data = self.validatePeaksPPG(
            paired_data=paired_data,
            threshold=self.config.threshold,
            window=self.config.window,
        )

        return validated_data["PeaksIndex"]

    def rr_intervals(self, peaksIndex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """rr_intervals(peaksIndex)

        Calculate R-R intervals (beat-to-beat intervals) from peak indices.

        Computes the time intervals between consecutive detected peaks.
        Useful for heart rate variability (HRV) analysis.

        Parameters
        ----------
        peaksIndex : np.ndarray
            Sample indices of detected peaks (from detect_peaks)

        Returns
        -------
        rr_intervals : np.ndarray
            Inter-beat intervals in seconds (length = len(peaksIndex) - 1)

        rr_time : np.ndarray
            Time of each R-R interval in seconds (length = len(peaksIndex) - 1)
            Timestamps are relative to start of signal

        Notes
        -----
        - First peak is used to anchor time; no interval for first peak
        - RR interval = sample_diff / sampling_rate
        - Example: if peaks at indices [100, 150, 220] and fs=100 Hz,
          rr_intervals = [0.5, 0.7] seconds"""

        rr_intervals = np.diff(peaksIndex) / self.config.sampling_rate
        rr_time = peaksIndex[1:] / self.config.sampling_rate

        return rr_intervals, rr_time

    def get_config(self) -> Dict[str, Any]:
        """get_config()

        Retrieve all configuration parameters as a dictionary.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary (self.config.__dict__)
            Contains filter settings, thresholds, window sizes, sampling rate, etc."""
        return self.config.__dict__
