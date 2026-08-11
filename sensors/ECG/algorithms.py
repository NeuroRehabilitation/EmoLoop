"""
sensors.ECG.algorithms

PanTompkinsAlgorithm - a reference implementation of the ECG_base interface.

This module implements the Pan-Tompkins algorithm for R-peak detection in ECG signals.
The Pan-Tompkins method is a widely-used, robust algorithm that combines:
  - Bandpass filtering to isolate the QRS complex frequency range
  - Signal derivative to enhance the rising/falling edges
  - Signal squaring to emphasize higher frequencies
  - Peak detection with adaptive thresholding
  - R-peak validation and synchronization to ensure accuracy

The algorithm is particularly effective for real-time ECG processing and can handle
noisy signals. It includes gap detection to identify missing beats and adaptive
thresholds based on recent RR intervals.

References:
  - Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
    IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236.
"""

from abc import abstractmethod

import numpy as np
import scipy
from sensors.ECG.base import ECG_base
from sensors.ECG.config import ECG_Config


class PanTompkinsAlgorithm(ECG_base):
    """
    Pan-Tompkins R-peak detection algorithm for ECG signals.

    This class implements the classic Pan-Tompkins algorithm which detects
    R-peaks (the main positive deflections of the QRS complex) in ECG signals.
    The algorithm works by applying a series of signal processing steps followed
    by peak detection with adaptive thresholding.

    Processing pipeline:
      1. Convert raw ADC signal to millivolts (convertECG)
      2. Bandpass filter to isolate QRS complex (filter)
      3. Calculate signal derivative to highlight edges (derivativeECG)
      4. Square the signal to emphasize peaks (squareECG)
      5. Detect peaks in the squared signal (_find_peaks)
      6. Apply adaptive thresholding based on signal/noise peaks
      7. Track RR intervals to detect dropped beats and fill gaps (rr_1_update, rr_2_update)
      8. Synchronize detected peaks to exact maximum in original signal (sync)

    Attributes
    ----------
    config : ECG_Config
        Configuration object containing:
          - sampling_rate: ECG sampling frequency (Hz)
          - VCC, gain, resolution: parameters for ADC conversion
          - butter_order, lowpass_freq, highpass_freq: filter parameters
          - filter_type: "bandpass" or other filter types
          - valley: boolean to detect valleys instead of peaks
          - edge: "rising", "falling", or "both" for peak type
          - mph: minimum peak height threshold
          - threshold: minimum peak prominence
          - mpd: minimum peak distance (samples)
          - kpsh: keep peaks same height boolean
          - discard_window: time to discard at signal start (seconds)

    Parameters
    ----------
    config : ECG_Config
        Configuration object with all algorithm parameters
    """

    def __init__(self, config: ECG_Config) -> None:
        self.config = config

    def convertECG(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert raw ADC signal to millivolts.

        Transforms a digitized ECG signal (typically in arbitrary ADC units)
        to standard ECG units (millivolts) using the analog-to-digital converter
        parameters from the configuration.

        Conversion formula:
            signal_volts = (signal * 2^resolution - 1/2) * VCC / gain
            signal_mv = signal_volts * 1000

        Parameters
        ----------
        signal : np.ndarray
            Raw ADC signal samples (in arbitrary units). Typically values
            in the range [0, 2^resolution).

        Returns
        -------
        np.ndarray
            Signal converted to millivolts (mV).

        Notes
        -----
        - Requires config.VCC, config.gain, config.resolution to be properly set.
        - VCC: reference voltage (volts)
        - gain: amplifier gain (unitless)
        - resolution: ADC bit resolution (typically 8, 10, 12, or 16 bits)
        """
        VCC = self.config.VCC
        gain = self.config.gain
        resolution = self.config.resolution
        signal_volts = (signal * pow(2, resolution) - 1 / 2) * VCC / gain
        signal_mv = signal_volts * 1000

        return signal_mv

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Filter the ECG signal using a Butterworth bandpass filter.

        Applies a digital Butterworth bandpass filter to isolate the QRS complex
        frequency range (typically 5–15 Hz for ECG). The filter removes low-
        frequency baseline wandering and high-frequency noise.

        Uses scipy.signal.butter() and scipy.signal.sosfilt() for numerically
        stable filtering (second-order sections format, applied forward-backward
        via sosfiltfilt for zero phase distortion).

        Parameters
        ----------
        signal : np.ndarray
            Input ECG signal (in mV or any units).

        Returns
        -------
        np.ndarray
            Filtered ECG signal with the same shape as input.

        Notes
        -----
        - Filter parameters are read from config:
          - butter_order: filter order (typically 5–10)
          - lowpass_freq: high-pass frequency cutoff (Hz, e.g., 5)
          - highpass_freq: low-pass frequency cutoff (Hz, e.g., 15)
          - filter_type: "bandpass" (default) or other types
          - sampling_rate: signal sampling frequency (Hz)
        - sosfiltfilt applies the filter twice (forward and backward) for
          zero phase distortion; this doubles the effective filter order.
        - Requires: scipy.signal
        """
        sos = scipy.signal.butter(
            self.config.butter_order,
            [self.config.lowpass_freq, self.config.highpass_freq],
            btype=self.config.filter_type,
            fs=self.config.sampling_rate,
            output="sos",
        )

        return scipy.signal.sosfiltfilt(sos, signal)

    @staticmethod
    def derivativeECG(signal: np.ndarray) -> np.ndarray:
        """
        Calculate the first-order derivative of the ECG signal.

        Computes the discrete time derivative using the difference between
        consecutive samples. This step enhances the rising and falling edges
        of the QRS complex, making peaks more prominent.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (filtered ECG).

        Returns
        -------
        np.ndarray
            Derivative of the signal with the same shape as input. The first
            element is set to signal[0] (prepended value).

        Notes
        -----
        - Uses np.diff() with prepend=signal[0] to maintain array length.
        - The derivative amplifies high-frequency content (edges and sharp
          transitions) while attenuating low-frequency drift.
        - This step is part of the Pan-Tompkins QRS enhancement cascade.
        """
        return np.diff(signal, prepend=signal[0])

    @staticmethod
    def squareECG(signal: np.ndarray) -> np.ndarray:
        """
        Square the ECG signal element-wise.

        Squares each sample of the signal and scales by a constant factor (50).
        Squaring emphasizes large amplitudes (peaks) and further suppresses
        noise, while the scaling ensures reasonable magnitude for subsequent
        peak detection.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (typically the derivative of filtered ECG).

        Returns
        -------
        np.ndarray
            Squared and scaled signal: 50 * signal^2

        Notes
        -----
        - Squaring is a nonlinear operation that makes peaks more prominent
          relative to noise.
        - The scaling factor (50) is somewhat arbitrary but chosen to ensure
          reasonable signal magnitudes for the peak detection threshold.
        - This step is the last of the QRS enhancement cascade in Pan-Tompkins.
        """
        return 50 * np.square(signal)

    def _find_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect peaks in data based on their amplitude and other features.

        A flexible peak detection routine that finds local maxima (or minima if
        configured) subject to optional constraints:
          - Minimum peak height (mph)
          - Minimum peak prominence (threshold)
          - Minimum peak distance (mpd)
          - Peak type (rising, falling, or both edges)

        This is an internal method that integrates multiple peak detection
        criteria to handle real-world signals with noise and varying amplitudes.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (typically the squared and filtered ECG).

        Returns
        -------
        np.ndarray
            Array of indices (int) where peaks are detected. Empty array if
            no peaks meet all criteria.

        Notes
        -----
        Peak detection steps:
          1. Convert input to 1D float64 array; return empty if size < 3.
          2. Negate signal if valley=True (to find minima instead of maxima).
          3. Find raw peaks by checking where first difference changes sign
             (interior peaks, rising edges, or falling edges based on config).
          4. Remove peaks adjacent to NaN values.
          5. Exclude first and last sample (cannot be peaks by definition).
          6. Apply minimum height (mph) filter if configured.
          7. Apply minimum prominence (threshold) filter if configured.
          8. Cluster peaks within minimum distance (mpd) and keep the highest
             in each cluster (or all if kpsh=True).

        Config parameters used:
          - valley: bool, detect valleys instead of peaks (negates signal)
          - edge: None, "rising", "falling", or "both"
          - mph: minimum peak height (None to disable)
          - threshold: minimum peak prominence
          - mpd: minimum peak distance (samples)
          - kpsh: keep peaks same height (bool)
        """
        x = np.atleast_1d(signal).astype("float64")
        if x.size < 3:
            return np.array([], dtype=int)
        if self.config.valley:
            x = -x
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not self.config.edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if self.config.edge.lower() in ["rising", "both"]:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if self.config.edge.lower() in ["falling", "both"]:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[
                np.in1d(
                    ind,
                    np.unique(np.hstack((indnan, indnan - 1, indnan + 1))),
                    invert=True,
                )
            ]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size - 1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and self.config.mph is not None:
            ind = ind[x[ind] >= self.config.mph]
        # remove peaks - neighbors < threshold
        if ind.size and self.config.threshold > 0:
            dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
            ind = np.delete(ind, np.where(dx < self.config.threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and self.config.mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - self.config.mpd) & (
                        ind <= ind[i] + self.config.mpd
                    ) & (x[ind[i]] > x[ind] if self.config.kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])

        return ind

    @staticmethod
    def rr_1_update(rr_1, NFound, Found):
        """
        Update the running RR interval buffer from recently found R-peaks.

        Maintains a rolling buffer of the 8 most recent RR intervals and
        computes their mean. This is used for aggressive beat detection when
        few peaks have been found.

        Parameters
        ----------
        rr_1 : np.ndarray
            Buffer of 8 RR intervals (in samples). Shape: (8,).
        NFound : int
            Number of R-peaks found so far (index-like, 1-based).
        Found : np.ndarray
            Array of found peaks with shape (NPeaks, 3), where each row is
            [peak_index, peak_amplitude, cycle_step].

        Returns
        -------
        tuple (rr_1_updated, rr_average_1)
            - rr_1_updated: Updated 8-element buffer with new RR intervals
            - rr_average_1: Mean of the updated buffer (in samples)

        Notes
        -----
        - If NFound <= 7, fills the first NFound-1 elements with consecutive
          differences of peak indices.
        - If NFound > 7, replaces the entire buffer with the last 7 RR intervals.
        - Used in Pan-Tompkins for beat rate tracking when not enough beats
          have been detected yet.
        - This is a looser constraint compared to rr_2_update.
        """
        if np.logical_and(NFound <= 7, NFound > 0):
            rr_1[0 : NFound - 1] = np.ediff1d(Found[0:NFound, 0])
        elif NFound > 7:
            rr_1 = np.ediff1d((Found[NFound - 7 : NFound - 1, 0]))

        rr_average_1 = np.mean(rr_1)

        return rr_1, rr_average_1

    @staticmethod
    def rr_2_update(rr_2, NFound, Found, rr_low_limit, rr_high_limit):
        """
        Update the running RR interval buffer and adaptive thresholds.

        Maintains a stricter RR interval buffer (keeps only intervals within
        expected physiological range) and adapts detection thresholds based
        on recent heart rate. Also detects gaps (missing beats) where the
        actual RR interval exceeds the expected range.

        Parameters
        ----------
        rr_2 : np.ndarray
            Buffer of 8 RR intervals within the expected range (in samples).
            Shape: (8,).
        NFound : int
            Number of R-peaks found so far (index-like, 1-based).
        Found : np.ndarray
            Array of found peaks with shape (NPeaks, 3), where each row is
            [peak_index, peak_amplitude, cycle_step].
        rr_low_limit : float
            Lower bound for acceptable RR intervals (in samples).
        rr_high_limit : float
            Upper bound for acceptable RR intervals (in samples).

        Returns
        -------
        tuple (rr_2_updated, rr_average_2_updated, flag, rr_low_limit_updated, rr_high_limit_updated)
            - rr_2_updated: Updated buffer with intervals within acceptable range
            - rr_average_2_updated: Mean of the updated buffer (in samples)
            - flag: 1 if a gap (missed beat) was detected, 0 otherwise
            - rr_low_limit_updated: Updated lower threshold (0.92 * rr_average_2)
            - rr_high_limit_updated: Updated upper threshold (1.16 * rr_average_2)

        Notes
        -----
        - Only adds RR intervals to rr_2 if they fall within [rr_low_limit, rr_high_limit].
        - Updates thresholds based on the mean of accepted intervals to adapt
          to changing heart rate.
        - Detects missed beats by checking if delta > 1.66 * rr_average_2
          (expecting one beat missed, gap of ~66% longer than average).
        - These conservative thresholds are used when NFound > 1 to avoid
          accepting outlier intervals.
        """
        rr_average_2 = np.mean(rr_2)
        rr_missed_limit = 1.66 * rr_average_2
        flag = 0

        if NFound > 0:
            delta_arr = np.ediff1d(Found[NFound - 1 : NFound, 0])

            if delta_arr.size > 0:
                delta = delta_arr.item() if delta_arr.size == 1 else delta_arr[-1]

                if rr_low_limit <= delta <= rr_high_limit:
                    pos = NFound % 7
                    rr_2[7 if pos == 0 else pos] = delta

                rr_average_2 = np.mean(rr_2)
                rr_low_limit = 0.92 * rr_average_2
                rr_high_limit = 1.16 * rr_average_2
                rr_missed_limit = 1.66 * rr_average_2

                if delta > rr_missed_limit:
                    flag = 1

        return rr_2, rr_average_2, flag, rr_low_limit, rr_high_limit

    @staticmethod
    def sync(Found, NFound, ecg, N):
        """
        Synchronize detected peaks to exact R-peak location in original signal.

        Refines the peak indices to point exactly to the maximum value within
        a ±60 sample window around each detected peak. This improves accuracy
        by correcting for slight shifts in the squared/filtered signal.

        Parameters
        ----------
        Found : np.ndarray
            Array of detected peaks with shape (NPeaks, 3), where each row is
            [peak_index, peak_amplitude, cycle_step].
        NFound : int
            Number of peaks found (1-based index).
        ecg : np.ndarray
            Original filtered ECG signal used for synchronization.
        N : int
            Total number of samples in the signal.

        Returns
        -------
        np.ndarray
            Refined R-peak indices (int) pointing to the maximum value in each
            ±60 sample window. Duplicates are removed via np.unique().

        Notes
        -----
        - Searches within a ±60 sample window around each candidate peak.
        - If the window exceeds signal boundaries, it is clipped to [0, N).
        - Finds the maximum value in the window and returns its index.
        - Removes duplicate peaks (which may occur after refinement) using
          np.unique().
        - This synchronization step is critical for accurate HRV measurement,
          as small shifts in peak position can affect RR intervals.
        """
        R = np.ones(NFound, dtype=int)

        for ii in range(0, NFound):
            xtemp = Found[ii, 0]

            if xtemp - 60 > 0:
                indInf = xtemp - 60
            else:
                indInf = 0

            if xtemp + 60 < N:
                indSup = xtemp + 60
            else:
                indSup = N

            ind = range(int(indInf), int(indSup))

            xlook = ecg[ind]

            # Find the maximum and its index (handle case where max might not be found)
            max_val = max(ecg[ind])
            match_indices = np.where(xlook == max_val)[0]

            if len(match_indices) > 0:
                R[ii] = indInf + match_indices[0]
            else:
                R[ii] = int(xtemp)  # Use original peak index if no match

        # Remove duplicate peaks using np.unique()
        R = np.unique(R)

        return R

    def detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks in an ECG signal using the Pan-Tompkins algorithm.

        Full pipeline:
          1. Square the filtered ECG signal
          2. Detect all peaks in the squared signal
          3. Separate signal peaks (SPKI) from noise peaks (NPKI) using amplitude
          4. Set dynamic thresholds (threshold1 for signal detection, threshold2
             for searching back in gaps)
          5. Initialize RR interval buffers and track beat-to-beat intervals
          6. Iterate through all detected peaks, applying thresholds and
             tracking intervals
          7. If a gap is detected (interval too long), back-search for missed beats
          8. Synchronize final peaks to signal maximum within ±60 sample window
          9. Discard peaks in the initial discardable window (e.g., first 0.2 seconds)

        Parameters
        ----------
        signal : np.ndarray
            Filtered ECG signal (output from filter() method, in mV).

        Returns
        -------
        np.ndarray
            Array of R-peak sample indices (int), sorted in ascending order.
            Indices within config.discard_window are excluded.

        Raises
        ------
        ValueError
            If no peaks are detected in the squared signal.

        Notes
        -----
        - Requires a pre-filtered ECG signal (e.g., from the filter() method).
        - The algorithm assumes baseline heart rate ~78 BPM (1.3 seconds per beat).
        - Dynamic thresholds adapt based on recent signal and noise peaks.
        - Gap detection and back-search improve robustness to missed beats.
        - config.discard_window (seconds) is converted to samples and removed
          from the final result to exclude unreliable initial transients.

        References:
          - Pan, J., & Tompkins, W. J. (1985)
        """
        N = len(signal)
        # Squaring
        ecg_filter = 50.0 * signal**2.0

        # Find Peaks
        pksInd = self._find_peaks(ecg_filter)

        if len(pksInd) == 0:
            raise ValueError("No peak was detected in the signal.")

        pks = ecg_filter[pksInd]

        SPKI = np.mean(pks) * 0.5
        NPKI = np.mean(pks) * 0.1

        threshold1 = NPKI + 0.25 * (SPKI - NPKI)
        threshold2 = 0.5 * threshold1

        # %Assuming an average of 78 BPM, then the time between points is 1.3 ->
        # %1.3*fs = number of points between ind;

        rr_1 = np.ones(8) * 1.3 * self.config.sampling_rate
        rr_average_1 = np.mean(rr_1)

        rr_2 = np.ones(8) * 1.3 * self.config.sampling_rate

        rr_average_2 = np.mean(rr_2)

        rr_low_limit = 0.92 * rr_average_2
        rr_high_limit = 1.16 * rr_average_2

        NPeaks = len(pksInd)

        Found = np.ones((NPeaks, 3))
        Found[:, 1] = 1.3 * self.config.sampling_rate

        NFound = 0
        NFound_Old = NFound - 1

        flag = 0
        back = 0
        ii = 0

        while NPeaks - ii > 0:
            ii += 1
            # if peak found and didn't came back to check the peak again, use
            # threshold 1
            if ii - back > 0:
                TH = threshold1
            # use threshold 2
            else:
                TH = threshold2

            # if threshold inferior to the peak amplitude
            if pks[ii - 1] >= TH:
                # found 1 peak
                NFound += 1
                # fill the found array with [peak index, peak amplitude, cycle
                # step]
                Found[NFound - 1, :] = np.r_[pksInd[ii - 1], pks[ii - 1], ii - 1]

                # Update threshold
                # if not needed to go back:
                if ii - back > 0:
                    SPKI = 0.125 * pks[ii - 1] + 0.875 * SPKI
                else:
                    SPKI = 0.25 * pks[ii - 1] + 0.75 * SPKI

            else:
                NPKI = 0.125 * pks[ii - 1] + 0.875 * NPKI

            threshold1 = NPKI + 0.25 * (SPKI - NPKI)
            threshold2 = 0.5 * threshold1

            if NFound_Old != NFound - 1:
                rr_1, rr_average_1 = self.rr_1_update(rr_1, NFound - 1, Found)
                rr_2, rr_average_2, flag, rr_low_limit, rr_high_limit = (
                    self.rr_2_update(
                        rr_2, NFound - 1, Found, rr_low_limit, rr_high_limit
                    )
                )

                NFound_Old = NFound - 1

            if flag:
                print("Gap Found")

                flag = 0
                back = ii
                ii = Found[-1, 2]

        R = self.sync(Found, NFound, signal, N)

        min_start = int(self.config.discard_window * self.config.sampling_rate)

        return R[R >= min_start]

    def rr_intervals(self, r_peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate RR intervals and timestamps from R-peak indices.

        Computes the intervals (in seconds) between consecutive R-peaks, along
        with the timestamps of each interval's end point. These RR intervals
        are the fundamental data for HRV (heart rate variability) analysis.

        Parameters
        ----------
        r_peaks : np.ndarray
            Array of R-peak sample indices (int). Must have at least 2 elements.

        Returns
        -------
        tuple (rr_intervals, rr_time)
            - rr_intervals: np.ndarray of RR intervals in seconds. Shape: (N-1,)
            - rr_time: np.ndarray of timestamps for each RR interval in seconds.
                       These are the sample indices of peaks[1:] divided by
                       sampling_rate. Shape: (N-1,)

        Notes
        -----
        - RR intervals are computed as the difference between consecutive peak
          indices, then scaled by (1 / sampling_rate) to convert from samples
          to seconds.
        - rr_time[i] is the timestamp of the (i+1)-th R-peak (the end of the
          i-th RR interval).
        - If input has fewer than 2 peaks, the output arrays will be empty.
        - These outputs are typically passed to HRV processing algorithms
          (e.g., HRVAlgorithm) for further analysis.
        """
        rr_intervals = np.diff(r_peaks) / self.config.sampling_rate
        rr_time = r_peaks[1:] / self.config.sampling_rate

        return rr_intervals, rr_time

    def get_config(self):
        """
        Get the configuration of the ECG sensor/algorithm.

        Returns
        -------
        ECG_Config
            The configuration object passed during initialization, containing
            all algorithm parameters (sampling rate, filter settings, peak
            detection thresholds, etc.).
        """
        return self.config
