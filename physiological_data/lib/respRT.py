
import numpy as np
from scipy.signal import decimate


def peak_detector_Resp(signal, fs, window=20, x_threshold=.3, prominence_threshold=.1):
    if fs > 50:
        signal = decimate(signal, int(fs / 50))
        fs /= int(fs / 50)

    win_size = int(window * fs)
    plot = False

    x_thr = x_threshold * fs
    win = []

    peak_value = -np.inf
    valley_value = np.inf
    has_peak = False

    peaks = [], []  # Index, Value

    peak_index = 0
    index = 0

    for i in range(0, len(signal)):
        print(f"{i * 100 / len(signal):.2f}%", end='\r')
        old_has_peak = has_peak

        if len(win) >= win_size:
            win = win[1:]

        data = signal[i]

        win += [data]

        if len(win) >= win_size:
            ptp = np.ptp(win)
            pro = np.min(win) + 0.5 * ptp
            prominence = prominence_threshold * ptp

            value = win[-1]

            if value > pro and value > peak_value:
                peak_index = index
                peak_value = value
                has_peak = True

            if value < pro:
                if has_peak:
                    has_peak = False

            if value < valley_value:
                valley_value = value

            if len(peaks[1]) > 0:
                if old_has_peak != has_peak and not has_peak and peak_value - peaks[1][-1] > prominence:
                    peaks[0].append(peak_index)
                    peak_value = -np.inf

            if old_has_peak != has_peak and has_peak:
                peaks[1].append(valley_value)
                valley_value = np.inf

        index += 1

    return peaks, signal[peaks[0]]


if __name__ == '__main__':
    from biosignals import Devices
    import matplotlib.pyplot as plt

    device = Devices(r'..\..\acquisitions\Acquisitions\03_11_2020')
    data = device.getSensorsData(['RESPIRATION'])
    signal = device.convertAndfilterSignal(data['data'][:, 1], 'RESPIRATION', device.fs, device.resolution)

    peaks, _ = peak_detector_Resp(signal, device.fs)

    signal = decimate(signal, int(device.fs / 50))

    plt.figure()
    plt.plot(signal, zorder=-1)
    plt.scatter(peaks[0], signal[peaks[0]], c='r', marker='x', zorder=1)
    plt.show()
