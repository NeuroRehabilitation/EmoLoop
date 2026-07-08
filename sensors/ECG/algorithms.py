import numpy as np
import scipy
from novainstrumentation import butter_bandpass_filter, detect_panthomkins_peaks, rr_1_update, rr_2_update, sync

from base import ECG_base
from config import ECG_Config
import novainstrumentation as ni


class PanTompkinsAlgorithm(ECG_base):
    def __init__(self, config: ECG_Config) -> None:
        self.config = config

    def convertECG(self, signal: np.ndarray) -> np.ndarray:
        """Convert ECG signal to mV."""
        VCC = self.config.VCC
        gain = self.config.gain
        resolution = self.config.resolution
        signal_volts = (signal*pow(2,resolution) - 1/2)*VCC/gain
        signal_mv = signal_volts*1000

        return signal_mv

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """Filter the ECG signal using Butterworth bandpass algorithm."""

        filtered_signal = butter_bandpass_filter(signal, self.config.lowpass_freq, self.config.highpass_freq, fs=self.config.sampling_rate)

        return filtered_signal

    def derivativeECG(self, signal: np.ndarray) -> np.ndarray:
        """Calculate the derivative of the ECG signal."""

        return np.diff(signal,prepend=signal[0])

    def integrateECG(self, signal: np.ndarray) -> np.ndarray:
        nbr_sampls_int_wind = int(self.config.integration_window * self.config.sampling_rate)
        integrated_signal = np.zeros_like(signal)
        cumulative_sum = signal.cumsum()
        integrated_signal[nbr_sampls_int_wind:] = (
            cumulative_sum[nbr_sampls_int_wind:] - cumulative_sum[:-nbr_sampls_int_wind]
        ) / nbr_sampls_int_wind
        integrated_signal[:nbr_sampls_int_wind] = cumulative_sum[
            :nbr_sampls_int_wind
        ] / np.arange(1, nbr_sampls_int_wind + 1)

        return integrated_signal


    def detect_r_peaks(self, filtered_data: np.ndarray, fs: int) -> np.ndarray:
        """Detect R-peaks in filtered ECG signal using Pan-Tompkins algorithm."""
        # Squaring
        ecg_squared = 50.0 * filtered_data ** 2.0

        # Find Peaks
        pksInd = detect_panthomkins_peaks(ecg_squared, mpd=35)

        pks = ecg_squared[pksInd]

        SPKI = np.mean(pks) * 0.5
        NPKI = np.mean(pks) * 0.1

        threshold1 = NPKI + 0.25 * (SPKI - NPKI)
        threshold2 = 0.5 * threshold1

        # %Assuming an average of 78 BPM, then the time between points is 1.3 ->
        # %1.3*fs = number of points between ind;

        rr_1 = np.ones(8) * 1.3 * fs
        rr_average_1 = np.mean(rr_1)

        rr_2 = np.ones(8) * 1.3 * fs

        rr_average_2 = np.mean(rr_2)

        rr_low_limit = 0.92 * rr_average_2
        rr_high_limit = 1.16 * rr_average_2

        NPeaks = len(pksInd)

        Found = np.ones((NPeaks, 3))
        Found[:, 1] = 1.3 * fs

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
                rr_1, rr_average_1 = rr_1_update(rr_1, NFound - 1, Found)
                rr_2, rr_average_2, flag, rr_low_limit, rr_high_limit = rr_2_update(
                    rr_2, NFound - 1, Found, rr_low_limit, rr_high_limit)

                NFound_Old = NFound - 1

                # if np.mod(NFound, 8) == 0:
                # print(['Average of the 8 most recent HR is ',
                # str(rr_average_1 / fs * 60.0), ' (BPM)'])
                # print('')

            if flag:
                print('Gap Found')

                flag = 0
                back = ii
                ii = Found[-1, 2]

        R = sync(Found, NFound, filtered_data, len(filtered_data))

        return R

    def calculate_heart_rate(self, r_peaks: np.ndarray, fs: int) -> np.ndarray:
        """Calculate the heart rate from R-peaks."""
        rr_intervals = np.diff(r_peaks) / fs  # Convert to seconds
        heart_rate = 60 / rr_intervals  # Convert to beats per minute

        return heart_rate

    def get_config(self):
        """Get the configuration of the ECG sensor."""
        return self.config
