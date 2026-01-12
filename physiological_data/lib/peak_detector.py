import numpy as np
from scipy.signal import decimate

try:
    from physiological_data.lib.signal_processing import peakDetection, pan_tompkins_filter
except (ImportError, ModuleNotFoundError):
    from physiological_data.lib.signal_processing import peakDetection


def peak_detector_RT(signal, fs, step=1, win_size=1.5, x_thr=0.3):
    if fs > 50:
        signal = decimate(signal, int(fs/50))
        fs /= int(fs/50)
    win_size *= fs
    x_thr *= fs
    win = []
    pro_th = np.inf # Update the value after the first 3 seconds

    has_valley = False # A valley can only be found after a peak

    valleys, peaks = [[], []], [[], []]
    peak_index, valley_index = 0, 0
    peak_value = -np.inf
    valley_value = np.inf
    index = 0

    data = 0

    for i in range(0, len(signal), step):
        old_has_valley = has_valley

        if len(win) >= win_size:
            win = win[step:]

            data = signal[i]

            win.append(data)
        else:
            win.append(signal[i])

        if len(win) >= win_size:
            win_aux = pan_tompkins_filter(win, fs)

            pro = 0.3 * np.ptp(win_aux)
            sup_thr = min(win_aux) + .5 * np.ptp(win_aux)
            inf_thr = min(win_aux) + .1  * np.ptp(win_aux)

            value = win_aux[len(win)//2]

            ######################################################################################
            ################################## VALLEY DETECTION ##################################
            ######################################################################################
            valley_condition = True
            if len(valleys[1]) > 0:
                valley_condition = index > valleys[1][-1] + x_thr
            
            # print(valley_value, value, inf_thr, valley_condition, np.abs(peak_value - value))
            if value < valley_value and value < inf_thr and valley_condition and np.abs(peak_value - value) > pro:
                valley_value = value
                valley_index = int(index + len(win)//2)
                has_valley = True
            
            if value > sup_thr and old_has_valley:
                has_valley = False
            
            # Peak Detection between two valleys
            if not has_valley and value > peak_value and np.abs(value - valley_value) > pro:
                peak_value = value
                peak_index = int(index + len(win)//2)
            
            valley_condition = True
            if len(peaks[1]) > 0 and len(valleys[1]) > 0:
                valley_condition = peaks[1][-1] < valley_index and peaks[1][-1] > valleys[1][-1]
            if old_has_valley != has_valley and not has_valley and valley_condition:
                valleys[0].append(valley_value)
                valleys[1].append(valley_index)

                valley_value = np.inf

            if old_has_valley != has_valley and has_valley:
                peaks[0].append(peak_value)
                peaks[1].append(peak_index)

                peak_value = -np.inf
                has_valley = True

            index += step

    return peaks, valleys


def peak_detector(signal, fs, window=3, overlap=.3, x_thr=.3, pro=0.5):
    peaks, peaksPos, valleys, valleysPos = [], [], [], []
    window *= fs
    overlap = int(fs*overlap)
    
    for i in range(overlap, len(signal), window):
        segment = signal[i - overlap:i+window]
        
        peaks, peakPos_det, _, valleyPos_det = peakDetection(segment, x_threshold=int(overlap * fs), proeminence=pro*np.ptp(segment))

        if len(peakPos_det) > 1:
            peakPos_det, valleyPos_det = peakPos_det[:-1], valleyPos_det[:-1]
        
        for p in range(len(peakPos_det)):
            peaksPos += [peakPos_det[p] + i - overlap]
            valleysPos += [valleyPos_det[p] + i - overlap]

        overlap = len(segment) - peakPos_det[-1] + 1
    
    return (peaks, peaksPos), (valleys, valleysPos)



if __name__ == '__main__':
    from biosignals import Devices
    import matplotlib.pyplot as plt

    device = Devices(r'..\..\acquisitions\Acquisitions\03_11_2020')
    data_ecg = device.getSensorsData(['ECG'])
    ecg_signal = device.convertAndfilterSignal(data_ecg['data'][:, 1], 'ECG', device.fs, device.resolution)

    peaks, valleys = peak_detector_RT(ecg_signal, device.fs, x_thr=.1)

    plt.figure()
    plt.plot(decimate(ecg_signal, int(device.fs/50)))
    # plt.plot(ecg_signal)
    # plt.scatter(peaks[1], peaks[0], c='r')
    plt.vlines(peaks[1], -.5, .75, 'r')
    plt.show()

    data_resp = device.getSensorsData(['RESPIRATION'])
    resp_signal = device.convertAndfilterSignal(data_resp['data'][:, 1], 'RESPIRATION', device.fs, device.resolution)

    peaks, valleys = peak_detector_RT(resp_signal, device.fs, win_size=20)

    plt.figure()
    plt.plot(decimate(resp_signal, int(device.fs/50)))
    # plt.plot(ecg_signal)
    # plt.scatter(peaks[1], peaks[0], c='r')
    plt.vlines(peaks[1], -.5, .75, 'r')
    plt.show()
