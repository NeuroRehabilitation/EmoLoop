from abc import abstractmethod

import numpy as np
import scipy
from base import ECG_base
from config import ECG_Config


class PanTompkinsAlgorithm(ECG_base):
    def __init__(self, config: ECG_Config) -> None:
        self.config = config

    def convertECG(self, signal: np.ndarray) -> np.ndarray:
        """Convert ECG signal to mV."""
        VCC = self.config.VCC
        gain = self.config.gain
        resolution = self.config.resolution
        signal_volts = (signal * pow(2, resolution) - 1 / 2) * VCC / gain
        signal_mv = signal_volts * 1000

        return signal_mv

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """Filter the ECG signal using Butterworth bandpass algorithm."""

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
        """Calculate the derivative of the ECG signal."""

        return np.diff(signal, prepend=signal[0])

    @staticmethod
    def squareECG(signal: np.ndarray) -> np.ndarray:
        """Square the ECG signal."""
        return 50 * np.square(signal)

    def _find_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Detect peaks in data based on their amplitude and other features."""

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
        if np.logical_and(NFound <= 7, NFound > 0):
            rr_1[0 : NFound - 1] = np.ediff1d(Found[0:NFound, 0])
        elif NFound > 7:
            rr_1 = np.ediff1d((Found[NFound - 7 : NFound - 1, 0]))

        rr_average_1 = np.mean(rr_1)

        return rr_1, rr_average_1

    @staticmethod
    def rr_2_update(rr_2, NFound, Found, rr_low_limit, rr_high_limit):
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
        """Sync to find exact R-peaks in original filtered signal."""
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

                # if np.mod(NFound, 8) == 0:
                # print(['Average of the 8 most recent HR is ',
                # str(rr_average_1 / fs * 60.0), ' (BPM)'])
                # print('')

            if flag:
                print("Gap Found")

                flag = 0
                back = ii
                ii = Found[-1, 2]

        R = self.sync(Found, NFound, signal, N)

        min_start = int(self.config.discard_window * self.config.sampling_rate)

        return R[R >= min_start]

    def calculate_heart_rate(self, r_peaks: np.ndarray) -> Dict[str, Any]:
        """Calculate the heart rate from R-peaks."""
        rr_intervals = (
            np.diff(r_peaks) / self.config.sampling_rate
        )  # Convert to seconds
        heart_rate = 60 / rr_intervals  # Convert to beats per minute

        return {
            "RR Intervals": rr_intervals,
            "Heart Rate": heart_rate,
        }

    def get_config(self):
        """Get the configuration of the ECG sensor."""
        return self.config
