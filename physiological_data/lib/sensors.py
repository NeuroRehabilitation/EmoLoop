import math

try:
    from physiological_data.lib.biosignals import *
except (ImportError, ModuleNotFoundError):
    from physiological_data.lib.biosignals import *
import biosignalsnotebooks as bsnb
import scipy as sc
import neurokit2 as nk
from sklearn.decomposition import FastICA
from scipy.signal import welch
from scipy import integrate
from biosppy.signals.eda import *
import novainstrumentation as ni
import pandas as pd

try:
    from physiological_data.lib.tools import *
    from physiological_data.lib.respRT import peak_detector_Resp
    from physiological_data.lib.signal_processing import integration_DO
except (ImportError, ModuleNotFoundError):
    from physiological_data.lib.tools import *
    from physiological_data.lib.respRT import peak_detector_Resp
    from physiological_data.lib.signal_processing import integration_DO
"""EDA Class"""


class EDA(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

    def filterEDA(self):

        lowfrequency = 1
        order = 2

        data_filtered = bsnb.lowpass(
            self.data, lowfrequency, order, self.fs, use_filtfilt=True
        )

        return data_filtered

    @staticmethod
    def componentsEDA(data_filtered):

        eda_components = nk.eda_phasic(data_filtered)
        eda_phasic = eda_components["EDA_Phasic"].values
        eda_tonic = eda_components["EDA_Tonic"].values

        return eda_phasic, eda_tonic

    def featuresSCR(self, eda_phasic):
        info, signals = nk.eda_peaks(eda_phasic, self.fs, method="neurokit")

        try:
            SCR_Amplitude = signals["SCR_Amplitude"]
        except Exception as e:
            print("Error SCR Amp")
            print(e)
            SCR_Amplitude = np.nan
        try:
            SCR_RiseTime = signals["SCR_RiseTime"]
        except Exception as e:
            print(e)
            print("Error SCR Rise Time")
            SCR_RiseTime = np.nan
        try:
            SCR_RecoveryTime = signals["SCR_RecoveryTime"]
        except Exception as e:
            print(e)
            print("Error SCR Recovery Time")
            SCR_RecoveryTime = np.nan

        return SCR_Amplitude, SCR_RiseTime, SCR_RecoveryTime

    @staticmethod
    def frequencyAnalysis(data_filtered):
        try:
            downsampled1 = sc.signal.decimate(data_filtered, q=10, n=8)
            downsampled2 = sc.signal.decimate(downsampled1, q=10, n=8)
            downsampled3 = sc.signal.decimate(downsampled2, q=10, n=8)

            signal_filtered = bsnb.highpass(downsampled3, 0.01, order=8)

            freqs, power = sc.signal.welch(
                signal_filtered, nperseg=128, window="blackman"
            )

            return freqs, power

        except Exception as e:
            print("Error on EDA frequencyAnalysis")
            freqs = np.nan
            power = np.nan
            print(e)
            return freqs, power

    @staticmethod
    def frequencyFeatures(freq, power):
        """
        Indexes of Frequencies of each component
        """
        vlf_indexes = np.where((freq[:] >= 0.0033) & (freq[:] < 0.04))[0]
        lf_indexes = np.where((freq[:] >= 0.04) & (freq[:] < 0.15))[0]
        hf_indexes = np.where((freq[:] >= 0.15) & (freq[:] < 0.4))[0]
        total_power_indexes = np.where((freq[:] >= 0.0033) & (freq[:] <= 0.4))[0]
        """
            Power of each frequency component in the desired range of frequencies
            """

        try:
            vlf = round(
                sc.integrate.trapz(power[vlf_indexes], freq[vlf_indexes]) * 1000000,
                4,
            )
        except Exception as e:
            print(e)
            print("Error EDA VLF")
            vlf = np.nan
        try:
            lf = round(
                sc.integrate.trapz(power[lf_indexes], freq[lf_indexes]) * 1000000, 4
            )
        except Exception as e:
            print(e)
            print("Error EDA LF")
            lf = np.nan
        try:
            hf = round(
                sc.integrate.trapz(power[hf_indexes], freq[hf_indexes]) * 1000000, 4
            )
        except Exception as e:
            print(e)
            print("Error EDA HF")
            hf = np.nan
        try:
            total_power = round(
                sc.integrate.trapz(
                    power[total_power_indexes], freq[total_power_indexes]
                )
                * 1000000,
                4,
            )
        except Exception as e:
            print(e)
            print("Error EDA Total Power")
            total_power = np.nan

        """
            Frequency components in normalized units (n.u)
            Balance - LF(n.u)/HF(n.u)
            """

        try:
            lf_norm = round((lf / (total_power - vlf)) * 100, 2)
        except Exception as e:
            print(e)
            print("Error EDA LF(nu)")
            lf_norm = np.nan
        try:
            hf_norm = round((hf / (total_power - vlf)) * 100, 2)
        except Exception as e:
            print(e)
            print("Error EDA HF(nu)")
            hf_norm = np.nan
        try:
            ratio = round(lf_norm / hf_norm, 2)

            if math.isinf(ratio):
                ratio = np.nan
        except Exception as e:
            print(e)
            print("Error EDA ratio")
            ratio = np.nan

        frequency_features = {
            "VLF Power": [vlf],
            "LF Power": [lf],
            "HF Power": [hf],
            "Total Power": [total_power],
            "LF (nu)": [lf_norm],
            "HF (nu)": [hf_norm],
            "LF/HF": [ratio],
        }

        return frequency_features

    def getFeatures(self):
        eda_filtered = self.filterEDA()
        eda_phasic, eda_tonic = self.componentsEDA(eda_filtered)
        SCR_Amplitude, SCR_RiseTime, SCR_RecoveryTime = self.featuresSCR(eda_phasic)
        freqs, power = self.frequencyAnalysis(eda_filtered)
        frequency_features = self.frequencyFeatures(freqs, power)

        eda_phasic_dict = self.statistical_Features(eda_phasic)
        eda_tonic_dict = self.statistical_Features(eda_tonic)
        SCR_Amplitude_dict = self.statistical_Features(SCR_Amplitude)
        SCR_RiseTime_dict = self.statistical_Features(SCR_RiseTime)
        SCR_RecoveryTime_dict = self.statistical_Features(SCR_RecoveryTime)

        return (
            eda_phasic_dict,
            eda_tonic_dict,
            SCR_Amplitude_dict,
            SCR_RiseTime_dict,
            SCR_RecoveryTime_dict,
            frequency_features,
        )


"""ECG Class"""


class ECG(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

    def convertECG(self):
        """
        :param signal: Raw ECG signal
        :param resolution: channel resolution from device
        :return: ECG signal in milivolts and volts
        """

        VCC = 3000
        gain = 1000
        signal_volts = ((np.array(self.data) / 2**self.resolution) - 1 / 2) * VCC / gain
        signal_mv = signal_volts * 1000

        return signal_mv

    def filterECG(self):
        """
        :param signal: ECG signal
        :param fs: sampling frequency of the ECG signal
        :return: filtered_signal - Band Pass filtered ECG signal
        """

        """Band pass filter between 5 and 15 Hz"""
        filtered_signal = bsnb.detect._ecg_band_pass_filter(self.data, self.fs)

        return filtered_signal

    @staticmethod
    def _differentiateECG(signal):
        """
        :param signal: Filtered ECG signal
        :return: Derivative of the filtered ECG signal
        """

        differentiated_signal = np.diff(signal)

        return differentiated_signal

    @staticmethod
    def _squaredECG(signal):
        """
        :param signal: Derivative ECG signal
        :return: Squared signal of the ECG derivative
        """

        squared_signal = signal**2

        return squared_signal

    @staticmethod
    def _integrateECG(signal, fs, window=0.080):
        """
        :param signal: Squared ECG signal
        :param fs: sampling frequency of the ECG signal
        :param window: window of integration (80 ms)
        :return: Integrated ECG signal
        """

        nbr_sampls_int_wind = int(window * fs)
        integrated_signal = np.zeros_like(signal)
        cumulative_sum = signal.cumsum()
        integrated_signal[nbr_sampls_int_wind:] = (
            cumulative_sum[nbr_sampls_int_wind:] - cumulative_sum[:-nbr_sampls_int_wind]
        ) / nbr_sampls_int_wind
        integrated_signal[:nbr_sampls_int_wind] = cumulative_sum[
            :nbr_sampls_int_wind
        ] / np.arange(1, nbr_sampls_int_wind + 1)

        return integrated_signal

    def _detectRPeaks(self):
        """
        :param signal: Non-filtered ECG signal
        :param fs: sampling frequency of the ECG signal
        :return: time_peaks - time instant for each R-peak detected
                 amp_peaks - amplitude of each R-peak detected
        """
        # peaks, valleys = peak_detector(self.data, self.fs)
        peaks = ni.panthomkins.panthomkins(self.data, self.fs)

        # return peaks[1]
        return peaks

    def calculateHR(self, peaks):
        try:
            hr = []
            time = []
            for i in range(1, len(peaks)):
                value = (peaks[i] - peaks[i - 1]) * 60 / self.fs
                hr.append(value)
                time.append(peaks[i] - peaks[i - 1] / self.fs)
            return hr, time
        except Exception as e:
            print(e)
            pass

    def processECG(self):
        try:
            peaks = self._detectRPeaks()

            return peaks
            # hr, time = self.calculateHR(peaks)
            # return hr, time
        except Exception as e:
            print(e)
            pass


"""HRV Class"""


class HRV(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

    def RR_interval(self):
        """
        :param signal: ECG signal
        :param fs: sampling frequency of ECG signal
        :return: rr_interval - RR interval series (diff between R-peaks)
                 rr_interval_time - RR interval series time axis.
        """

        rr_interval, rr_interval_time = bsnb.tachogram(
            self.data, self.fs, signal=True, out_seconds=True
        )

        return rr_interval, rr_interval_time

    @staticmethod
    def remove_EctopyBeats(rr_interval, rr_interval_time):
        """
        :param rr_interval: RR interval series
        :param rr_interval_time: RR interval series time axis
        :return: rr_interval_NN - RR interval series with no ectopic beats
                 rr_interval_time_NN - RR interval series time axis with no ectopic beats
        """
        try:
            rr_interval_NN, rr_interval_time_NN = bsnb.remove_ectopy(
                rr_interval, rr_interval_time
            )
            if len(rr_interval_NN) > 0:
                rr_interval_NN.pop(0)
            if len(rr_interval_time_NN):
                rr_interval_time_NN.pop(0)
            rr_interval_NN = np.array(rr_interval_NN)
            # print(rr_interval_NN)
            rr_interval_time_NN = np.array(rr_interval_time_NN)

        except Exception as e:
            rr_interval_NN = rr_interval
            rr_interval_time_NN = rr_interval_time
            print("Error on Remove EctopyBeats")
            print(e)

        return rr_interval_NN, rr_interval_time_NN

    def heart_rate(self, rr_interval_NN):
        """
        :param rr_interval_NN: RR interval series with no ectopic beats
        :return: array of Heart Rate in beats per minute (Bpm) along time.
        """
        try:
            heart_rate = 60.0 / rr_interval_NN
        except Exception as e:
            print("Error on HR")
            print(e)
            heart_rate = np.nan

        return heart_rate

    def heartRate_features(self, heart_rate):

        statistical_features = self.statistical_Features(heart_rate)

        hr = {
            "Avg HR": statistical_features["AVG"],
            "Min HR": statistical_features["Minimum"],
            "Max HR": statistical_features["Maximum"],
            "SD": statistical_features["STD"],
        }

        # print(hr)

        return hr

    def timeDomainFeatures(self, rr_interval_NN):
        statistical_features = self.statistical_Features(rr_interval_NN)

        """
            :param rr_interval_NN: RR interval series with no ectopic beats
            :return: dict with time-domain features of HRV
            """

        rr_interval_diff = np.diff(rr_interval_NN)
        rr_interval_abs = np.abs(rr_interval_diff)

        """Standard deviation of RR interval series with no ectopic beats"""
        try:
            SDNN = round(np.std(rr_interval_NN) * 1000, 4)
        except Exception as e:
            print(e)
            print("Error HRV SDNN")
            SDNN = np.nan

        """Root Mean Square of the Standard deviation"""
        try:
            RMSSD = round(
                np.sqrt(np.sum((rr_interval_diff) ** 2) / (len(rr_interval_NN) - 1))
                * 1000,
                4,
            )
        except Exception as e:
            print(e)
            print("Error HRV RMSSD")
            RMSSD = np.nan

        """Number and percentage of RR interval longer than 50 ms"""
        try:
            NN50 = sum(1 for i in rr_interval_abs if i > 0.05)
        except Exception as e:
            print(e)
            print("Error HRV NN50")
            NN50 = np.nan
        try:
            pNN50 = round((float(NN50) / len(rr_interval_NN) * 100), 4)
        except Exception as e:
            print(e)
            print("Error HRV pNN50")
            pNN50 = np.nan

        """Number and percentage of RR interval longer than 20 ms"""
        try:
            NN20 = sum(1 for i in rr_interval_abs if i > 0.02)
        except Exception as e:
            print(e)
            print("Error HRV NN20")
            NN20 = np.nan
        try:
            pNN20 = round((float(NN20) / len(rr_interval_NN) * 100), 4)
        except Exception as e:
            print(e)
            print("Error HRV pNN20")
            pNN20 = np.nan

        time_domain_features = {
            "AVG RR": statistical_features["AVG"],
            "Minimum RR": statistical_features["Minimum"],
            "Maximum RR": statistical_features["Maximum"],
            "SDNN": SDNN,
            "RMSSD": RMSSD,
            "NN50": NN50,
            "pNN50": pNN50,
            "NN20": NN20,
            "pNN20": pNN20,
        }
        # print(time_domain_features)

        return time_domain_features

    @staticmethod
    def poincareFeatures(rr_interval_NN):
        """
        :param rr_interval_NN: RR interval series with no ectopic beats
        :return: dict with Poincare features - non-linear features
        """

        """Standard Deviation of RR interval series"""
        try:
            STD = round(float(np.std(rr_interval_NN)), 4)
        except Exception as e:
            print(e)
            print("Error HRV STD")
            STD = np.nan

        """Standard Deviation of the successive differences of RR interval series"""
        try:
            SDSD = round(float(np.std(np.diff(rr_interval_NN))), 4)
        except Exception as e:
            print(e)
            print("Error HRV SDSD")
            SDSD = np.nan

        """Length of the longitudinal line in Poincaré plot"""
        try:
            SD2 = round(np.sqrt(2 * STD**2 - 0.5 * SDSD**2), 4) * 1000
        except Exception as e:
            print(e)
            print("Error HRV SD2")
            SD2 = np.nan

        """Length of the transverse line in Poincaré plot"""
        try:
            SD1 = round(np.sqrt(0.5 * SDSD**2), 4) * 1000
        except Exception as e:
            print(e)
            print("Error HRV SD1")
            SD1 = np.nan

        "SD2/SD1"
        try:
            SD_ratio = round(SD2 / SD1, 4)

            if math.isinf(SD_ratio):
                SD_ratio = np.nan

        except Exception as e:
            print(e)
            print("Error HRV SD_ratio")
            SD_ratio = np.nan

        poincaré_features = {
            "SD1": [SD1],
            "SD2": [SD2],
            "SD2/SD1": [SD_ratio],
        }

        return poincaré_features

    @staticmethod
    def evenlySpaced_RR(rr_interval_NN, time, new_freq):
        """
        :param rr_interval_NN: RR interval series with no ectopic beats
        :param time: time axis
        :param new_freq: Frequency to which the RR interval series will be downsampled
        :return: Interpolated and Downsampled RR interval series
        """
        try:
            """Time axis downsampled to new frequency"""
            downsampled_time = np.linspace(0, int(time[-1]), int(time[-1] * new_freq))

            """Time array with RR interval series evenly time-spaced"""
            evenlySpaced_time = np.linspace(0, int(time[-1]), int(len(rr_interval_NN)))

            """
            (t,c,k) a tuple containing the vector of knots, the B-spline coefficients, and degree of spline
            Model of the RR interval series
            """
            tck = sc.interpolate.splrep(evenlySpaced_time, rr_interval_NN)

            """Interpolated RR interval series using the model obtained"""
            interpolatedRR = sc.interpolate.splev(downsampled_time, tck)

            return interpolatedRR
        except Exception as e:
            print(e)
            pass

    @staticmethod
    def frequencyAnalysis(
        rr_interval_time_NN, rr_interval_NN, window="hann", interpolation_rate=4
    ):
        try:
            try:
                init_time = int(rr_interval_time_NN[0])
                fin_time = int(rr_interval_time_NN[-1])
            except Exception as e:
                print(e)
                pass

            try:
                tck = sc.interpolate.splrep(rr_interval_time_NN, rr_interval_NN)

                nn_time_even = np.linspace(
                    init_time, fin_time, (fin_time - init_time) * interpolation_rate
                )
                nn_tachogram_even = sc.interpolate.splev(nn_time_even, tck)
            except Exception as e:
                nn_tachogram_even = rr_interval_NN
                print(e)
                pass

            try:
                freq_axis, power_axis = sc.signal.welch(
                    nn_tachogram_even,
                    interpolation_rate,
                    window=sc.signal.get_window(
                        window, min(len(nn_tachogram_even), 1000)
                    ),
                    nperseg=min(len(nn_tachogram_even), 1000),
                )
            except Exception as e:
                print(e)
                pass

            try:
                freqs = np.array([round(val, 3) for val in freq_axis if val < 0.5])
                power = np.array(
                    [
                        round(val, 4)
                        for val, freq in zip(power_axis, freq_axis)
                        if freq < 0.5
                    ]
                )
            except Exception as e:
                print(e)
                pass

            return freqs, power

        except Exception as e:
            print(e)
            print("Error on Frequency Analysis")
            pass

    @staticmethod
    def frequencyFeatures(freq, power):
        """
        Indexes of Frequencies of each component
        """
        vlf_indexes = np.where((freq[:] >= 0.0033) & (freq[:] < 0.04))[0]
        lf_indexes = np.where((freq[:] >= 0.04) & (freq[:] < 0.15))[0]
        hf_indexes = np.where((freq[:] >= 0.15) & (freq[:] < 0.4))[0]
        total_power_indexes = np.where((freq[:] >= 0.0033) & (freq[:] <= 0.4))[0]
        """
            Power of each frequency component in the desired range of frequencies
            """
        try:
            vlf = round(
                sc.integrate.trapz(power[vlf_indexes], freq[vlf_indexes]) * 1000000,
                4,
            )
        except Exception as e:
            print("Error HRV VLF")
            print(e)
            vlf = np.nan
        try:
            lf = round(
                sc.integrate.trapz(power[lf_indexes], freq[lf_indexes]) * 1000000, 4
            )
        except Exception as e:
            print(e)
            print("Error HRV LF")
            lf = np.nan
        try:
            hf = round(
                sc.integrate.trapz(power[hf_indexes], freq[hf_indexes]) * 1000000, 4
            )
        except Exception as e:
            print(e)
            print("Error HRV HF")
            hf = np.nan
        try:
            total_power = round(
                sc.integrate.trapz(
                    power[total_power_indexes], freq[total_power_indexes]
                )
                * 1000000,
                4,
            )
        except Exception as e:
            print(e)
            print("Error HRV Total Power")
            total_power = np.nan

        """
            Frequency components in normalized units (n.u)
            Balance - LF(n.u)/HF(n.u)
            """
        try:
            lf_norm = round(lf / (total_power - vlf) * 100, 2)
        except Exception as e:
            print(e)
            print("Error HRV LF(nu)")
            lf_norm = np.nan
        try:
            hf_norm = round(hf / (total_power - vlf) * 100, 2)
        except Exception as e:
            print(e)
            print("Error HRV HF(nu)")
            hf_norm = np.nan
        try:
            ratio = round(lf_norm / hf_norm, 2)

            if math.isinf(ratio):
                ratio = np.nan
        except Exception as e:
            print(e)
            print("Error HRV ratio")
            ratio = np.nan

        frequency_features = {
            "HRV VLF Power": [vlf],
            "HRV LF Power": [lf],
            "HRV HF Power": [hf],
            "HRV Total Power": [total_power],
            "HRV LF (nu)": [lf_norm],
            "HRV HF (nu)": [hf_norm],
            "HRV LF/HF": [ratio],
        }

        return frequency_features

    def getFeatures(self):
        rr_interval, rr_interval_time = self.RR_interval()
        rr_interval_NN, rr_interval_time_NN = self.remove_EctopyBeats(
            rr_interval, rr_interval_time
        )
        heart_rate = self.heart_rate(rr_interval_NN)
        heart_rate_features = self.heartRate_features(heart_rate)
        freq, power = self.frequencyAnalysis(rr_interval_time_NN, rr_interval_NN)
        time_features = self.timeDomainFeatures(rr_interval_NN)
        poincare_features = self.poincareFeatures(rr_interval_NN)
        frequency_features = self.frequencyFeatures(freq, power)

        return (
            heart_rate_features,
            time_features,
            poincare_features,
            frequency_features,
        )


"""PPG Class"""


class PPG(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

    def filterPPG(
        self, lowpassFreq=5, highpassFreq=0.1, lowpassOrder=2, highpassOrder=2
    ):

        filteredPPG = bsnb.highpass(
            bsnb.lowpass(self.data, lowpassFreq, lowpassOrder, self.fs),
            highpassFreq,
            highpassOrder,
            self.fs,
        )

        return filteredPPG

    @staticmethod
    def findPeaksPPG(signal):
        peaks = []
        peaks_location = []

        for i in range(len(signal) - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                peaks.append(signal[i])
                peaks_location.append(i)

        peaksAmp = np.array(peaks)
        peaksIndex = np.array(peaks_location)

        return peaksAmp, peaksIndex

    @staticmethod
    def findValleysPPG(signal):

        valleys = []
        valleys_location = []

        for i in range(len(signal) - 1):
            if signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
                valleys.append(signal[i])
                valleys_location.append(i)

        valleysAmp = np.array(valleys)
        valleysIndex = np.array(valleys_location)

        return valleysAmp, valleysIndex

    @staticmethod
    def remove1stPeak(peaksAmp, peaksIndex, valleysIndex):

        if peaksIndex[0] < valleysIndex[0]:
            peaksAmp.remove(peaksAmp[0])
            peaksIndex.remove(peaksIndex[0])

        return peaksAmp, peaksIndex

    @staticmethod
    def peaks_valleyDiff(peaksAmp, valleysAmp):

        vpd = []

        for i in range(len(peaksAmp)):
            vpd.append(peaksAmp[i] - valleysAmp[i])

        valley_peak_diff = np.array(vpd)

        return valley_peak_diff

    def detectPeaksPPG(self, lowpassFreq=5, highpassFreq=0.1):

        filteredPPG = PPG.filterPPG(self, lowpassFreq, highpassFreq)
        peaksAmp, peaksIndex = PPG.findPeaksPPG(filteredPPG)
        valleysAmp, valleysIndex = PPG.findValleysPPG(filteredPPG)
        peaksAmp, peaksIndex = PPG.remove1stPeak(peaksAmp, peaksIndex, valleysIndex)
        valley_peak_diff = PPG.peaks_valleyDiff(peaksAmp, valleysAmp)

        numberPeaks = len(valley_peak_diff)
        new_numberPeaks = -1
        flag = True

        while flag:
            peak_location = []
            peak_amplitude = []
            for i in range(2, len(peaksAmp) - 2):
                if valley_peak_diff[i] > (
                    0.7
                    * (
                        valley_peak_diff[i - 2]
                        + valley_peak_diff[i - 1]
                        + valley_peak_diff[i]
                        + valley_peak_diff[i + 1]
                        + valley_peak_diff[i + 2]
                    )
                    / 5
                ):
                    peak_location.append(peaksIndex[i])
                    peak_amplitude.append(peaksAmp[i])

            new_peaks_number = len(peak_location)

            if numberPeaks == new_peaks_number:
                flag = False
            else:
                numberPeaks = new_peaks_number

            peaksIndexes = np.array(peak_location)
            peaksAmplitude = np.array(peak_amplitude)

            return peaksAmplitude, peaksIndexes


"""fNIRS Class"""

# class fNIRS(Sensor):
#     def __init__(self, data, fs, resolution):
#         super().__init__(data, fs, resolution)
#         # self.red = self.convertPhys(data[:, 0], resolution)
#         # self.infrared = self.convertPhys(data[:, 1], resolution)
#
#         self.red = self.filterData(data[:,0], fs)
#         self.infrared = self.filterData(data[:,1], fs)
#
#         # print(self.red, self.infrared)
#
#     @staticmethod
#     def convertPhys(data, resolution):
#         return (0.15 * data) / (2**resolution)
#
#     def convertConcentration(self):
#         converter = Mes2Hb()
#         self.hbo, self.hb, self.hbt = converter.convert([self.red.copy(), self.infrared.copy()], wavelength=[660, 860])
#
#     def detectPeaks(self):
#         pass
#
#     @staticmethod
#     def filterData(data, fs):
#         return bsnb.bandpass(data, 0.05, 0.4, fs=fs, use_filtfilt=True)
#
#     @staticmethod
#     def root_mean_square(signal):
#         """Signal should be a segment"""
#         return np.sqrt(np.mean(signal**2))
#
#     @staticmethod
#     def slope_regression(signal):
#         """Signal should be a segment"""
#         time = np.arange(0, len(signal)).reshape(-1, 1)
#         model = LinearRegression()
#         model = model.fit(time, signal.reshape(-1,1))
#         return model.coef_[0]
#
#     @staticmethod
#     def slope_naive(signal):
#         """Signal should be a segment"""
#         return signal[-1] - signal[0]
#
#     def processfNIRS(self):
#         self.convertConcentration()
#
#     def getFeatures(self):
#
#         hb_dict = self.statistical_Features(self.hb)
#         hbo_dict = self.statistical_Features(self.hbo)
#         hbt_dict = self.statistical_Features(self.hbt)
#
#         fNIRS_dict={"AVG Hb":[hb_dict["AVG"]],"Minimum Hb":[hb_dict["Minimum"]],"Maximum Hb":[hb_dict["Maximum"]],"STD Hb":[hb_dict["STD"]],
#                     "AVG Hbo":[hbo_dict["AVG"]],"Minimum Hbo":[hbo_dict["Minimum"]],"Maximum Hbo":[hbo_dict["Maximum"]],"STD Hbo":[hbo_dict["STD"]],
#                     "AVG Hbt":[hbt_dict["AVG"]],"Minimum Hbt":[hbt_dict["Minimum"]],"Maximum Hbt":[hbt_dict["Maximum"]],"STD Hbt":[hbt_dict["STD"]]}
#
#         return fNIRS_dict
#

"""EEG Class"""


class EEG(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

        self.data = data  # converted and filtered EEG data
        self.fs = fs
        self.resolution = resolution
        self.bands = {
            "alpha": [8, 14],
            "betha": [14, 30],
            "gamma": [30, 49],
            "theta": [4, 8],
            "delta": [0.5, 4],
        }

    @staticmethod
    def ICA(data):
        try:
            ica = FastICA(whiten="unit-variance")

            EEG_ICA = ica.fit_transform(np.array(data).reshape(-1, 1))

            return EEG_ICA
        except Exception as e:
            print(e)
            pass

    @staticmethod
    def filterData(data, fs):
        EEG_shift = data[:, 0] - np.mean(data)
        EEG_filtered = bsnb.bandpass(
            EEG_shift, 1, 50, order=8, fs=fs, use_filtfilt=True
        )

        return EEG_filtered

    @staticmethod
    def frequencyAnalysis(data, fs):
        try:
            freqs, power = welch(data, fs, nperseg=fs / 2)

            alpha_indexes = np.where((freqs[:] >= 8) & (freqs[:] < 14))[0]
            betha_indexes = np.where((freqs[:] >= 14) & (freqs[:] < 30))[0]
            gamma_indexes = np.where((freqs[:] >= 30) & (freqs[:] < 49))[0]
            theta_indexes = np.where((freqs[:] >= 4) & (freqs[:] < 8))[0]
            delta_indexes = np.where((freqs[:] >= 0.5) & (freqs[:] < 4))[0]

            try:
                alpha = sc.integrate.trapz(power[alpha_indexes], x=freqs[alpha_indexes])
            except Exception as e:
                print(e)
                alpha = np.nan
            try:
                betha = sc.integrate.trapz(power[betha_indexes], x=freqs[betha_indexes])
            except Exception as e:
                print(e)
                betha = np.nan
            try:
                gamma = sc.integrate.trapz(power[gamma_indexes], x=freqs[gamma_indexes])
            except Exception as e:
                print(e)
                gamma = np.nan
            try:
                theta = sc.integrate.trapz(power[theta_indexes], x=freqs[theta_indexes])
            except Exception as e:
                print(e)
                theta = np.nan
            try:
                delta = sc.integrate.trapz(power[delta_indexes], x=freqs[delta_indexes])
            except Exception as e:
                print(e)
                delta = np.nan

            bands_power = {
                "alpha": alpha,
                "betha": betha,
                "gamma": gamma,
                "theta": theta,
            }  # ,"delta":delta}

            return bands_power
        except Exception as e:
            print(e)
            pass

    def extractBand(self, band: list):
        try:
            f1, f2 = band
            win = self.fs / 2
            freq, power = welch(self.data, self.fs, nperseg=win)
            idx_band = np.logical_and(
                freq >= f1, freq <= f2
            )  # Get the band of frequencies
            power_freq = np.trapz(
                power[idx_band], x=freq[idx_band]
            )  # Calculate the power of the band of frequencies

            return power_freq
        except Exception as e:
            print(e)
            pass

    def extractAllBands(self):
        try:
            band_powers = {}

            for key, item in self.bands.items():
                band_powers[key] = self.extractBand(self.data, item)

            return band_powers
        except Exception as e:
            print(e)
            pass

    # def getFeatures(self):
    #     # self.ICA()
    #     self.filterData()
    #
    #     band_powers = self.frequencyAnalysis()
    #
    #     return band_powers

    def getDominantFreq(self, data, fs):
        win = 4 * fs
        freq, power = welch(data, fs, nperseg=win)
        dominant_freq = freq[np.argmax(power)]
        return dominant_freq

    def getCombinationFreq(self, power_freqs: dict):
        combinations = {}
        for key, item in power_freqs.items():
            for other_key, other_item in power_freqs.items():
                if key != other_key:
                    combinations[f"{key}/{other_key}"] = item / other_item
        return combinations


"""TEMP Class"""


class TEMP(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

    def filterData(self, lowpassfreq: int):
        self.data = bsnb.lowpass(
            self.data, lowpassfreq, order=2, fs=self.fs, use_filtfilt=True
        )

        return self.data

    def getFeatures(self):
        self.data = self.filterData(lowpassfreq=1)
        temp = self.statistical_Features(self.data)

        temp_dict = {
            "AVG Temp": [temp["AVG"]],
            "Max Temp": [temp["Maximum"]],
            "Min Temp": [temp["Minimum"]],
            "STD Temp": [temp["STD"]],
        }

        temp_Dataframe = pd.DataFrame.from_dict(temp_dict, orient="columns")

        return temp_Dataframe


"""ACC Class"""


class ACC(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

        self.data = data * 9.8  # Convert to m/s

    def getIntegration(self, data, fs):
        # t = bsnb.generate_time(data, fs)
        return integration_DO(data, fs)

    def getVel(self, acc, fs):
        return self.vel

    def getDesl(self, vel, fs):
        return self.desl

    def getFeatures(self):
        self.acc = self.data
        self.vel = self.getIntegration(self.acc, self.fs)
        self.desl = self.getIntegration(self.vel, self.fs)

        features = []
        for i in [self.acc, self.vel, self.desl]:
            for func in [np.mean, np.min, np.max, np.std]:
                features.append(func(i))

        return np.array(features)


"""RESP Class"""


class RESP(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

        self.data = np.array(data).astype(float)

    def process_RESP(self):

        signals, info = nk.rsp_process(self.data, self.fs, method="biosppy")

        return signals, info

    @staticmethod
    def RESP_RRV(signals):
        try:
            info, peak_signals = nk.rsp_peaks(signals["RSP_Clean"])

            rrv_dataframe = nk.rsp_rrv(
                signals["RSP_Rate"], troughs=peak_signals["RSP_Troughs"]
            )

            return rrv_dataframe
        except Exception as e:
            print("RESP_RRV")
            print(e)
            rrv_dataframe = pd.DataFrame(
                columns=[
                    "RRV_RMSSD",
                    "RRV_MeanBB",
                    "RRV_SDBB",
                    "RRV_SDSD",
                    "RRV_CVBB",
                    "RRV_CVSD",
                    "RRV_MedianBB",
                    "RRV_MadBB",
                    "RRV_MCVBB",
                    "RRV_nn20",
                    "RRV_nn50",
                    "RRV_pNN50",
                    "RRV_pNN20",
                    "RRV_HF",
                    "RRV_SD1",
                ]
            )
            return rrv_dataframe

    def getFeatures(self, signals, rrv_dataframe):

        rsp_rate_dict = self.statistical_Features(signals["RSP_Rate"])
        rsp_amp_dict = self.statistical_Features(signals["RSP_Amplitude"])

        # rrv_dataframe = pd.DataFrame.from_dict(rsp_rate_dict)  # comment for full dataframe

        """uncomment following lines to get full dataframe"""
        try:
            rrv_dataframe.insert(0, "STD_RSP_Amplitude", rsp_amp_dict["STD"])
        except Exception as e:
            print(e)
            rrv_dataframe.insert(0, "STD_RSP_Amplitude", np.nan)

        try:
            rrv_dataframe.insert(0, "Maximum_RSP_Amplitude", rsp_amp_dict["Maximum"])
        except Exception as e:
            print(e)
            rrv_dataframe.insert(0, "Maximum_RSP_Amplitude", np.nan)

        try:
            rrv_dataframe.insert(0, "Minimum_RSP_Amplitude", rsp_amp_dict["Minimum"])
        except Exception as e:
            print(e)
            rrv_dataframe.insert(0, "Minimum_RSP_Amplitude", np.nan)

        try:
            rrv_dataframe.insert(0, "Mean_RSP_Amplitude", rsp_amp_dict["AVG"])
        except Exception as e:
            print(e)
            rrv_dataframe.insert(0, "Mean_RSP_Amplitude", np.nan)

        try:
            rrv_dataframe.insert(0, "STD_RSP_Rate", rsp_rate_dict["STD"])
        except Exception as e:
            print(e)
            rrv_dataframe.insert(0, "STD_RSP_Rate", np.nan)

        try:
            rrv_dataframe.insert(0, "Maximum_RSP_Rate", rsp_rate_dict["Maximum"])
        except Exception as e:
            print(e)
            rrv_dataframe.insert(0, "Maximum_RSP_Rate", np.nan)

        try:
            rrv_dataframe.insert(0, "Minimum_RSP_Rate", rsp_rate_dict["Minimum"])
        except Exception as e:
            print(e)
            rrv_dataframe.insert(0, "Minimum_RSP_Rate", np.nan)

        try:
            rrv_dataframe.insert(0, "Mean_RSP_Rate", rsp_rate_dict["AVG"])
        except Exception as e:
            print(e)
            rrv_dataframe.insert(0, "Mean_RSP_Rate", np.nan)

        return rrv_dataframe
