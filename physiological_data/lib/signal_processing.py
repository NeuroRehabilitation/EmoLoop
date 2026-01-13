import numpy as np
from biosignalsnotebooks import bsnb

def almostEqual(v1, v2, delta):
    if isinstance(v1, int):
        return abs(v1 - v2) < delta
    else:
        return all(abs(v1 - v2) < delta)

def pan_tompkins_filter(data, fs):
    # Pan & Tompkins signal conditioning: https://www.robots.ox.ac.uk/~gari/teaching/cdt/A3/readings/ECG/Pan+Tompkins.pdf
    # (used the normal differentiation, as the 4-point differentiation gave equal results)
    red_diff = differentation(data) # differentiate data
    red_sqr = square(red_diff) # square it
    red_sqr_int_DO = integration_DO(red_sqr, fs) # and integrate it
    return red_sqr_int_DO

def getValuesArray(array, index_s, index_e):
    index_s, index_e = int(index_s), int(index_e)
    new_array = np.zeros((index_e - index_s, ))
    for i in range(index_s, index_e):
        new_array[i - index_s] = array[i]
    return new_array

def differentation(signal):
    diff_signal = np.empty(len(signal) - 1)
    for i in range(1, len(signal)):
        diff_signal[i-1] = signal[i] - signal[i-1]
    return diff_signal

def mean(data):
    mean_value = 0
    for i in range(len(data)):
        mean_value += data[i]
    mean_value = mean_value / len(data)
    return mean_value

def std(data):
    mean_value = mean(data)
    std_value = 0
    for i in range(len(data)):
        aux_value = data[i] - mean_value
        std_value += aux_value * aux_value
    std_value /= len(data)
    std_value = np.sqrt(std_value)

    return std_value


def square(data):
    square_data = np.zeros(len(data))
    for i in range(len(data)):
        square_data[i] = data[i]*data[i]
    return square_data

def smooth_(signal, n):
    smoothen = np.zeros(len(signal))
    aux_signal = np.zeros(2*n + len(signal))
    for i in range(n):
        # mirror the start of signal
        aux_signal[i] = signal[n-i]
        # mirror the end of signal
        aux_signal[len(signal)+n+i] = signal[len(signal)-i-2]
    for i in range(len(signal)):
        aux_signal[i+n] = signal[i]
    for i in range(n, len(aux_signal)-n):
        smoothen[i-n] = mean(getValuesArray(aux_signal, i-n//2, i+n//2))
    return smoothen

def smooth(signal, n):
    smoothen = np.zeros(len(signal))
    aux_signal = np.zeros(2*n + len(signal))
    for i in range(len(signal)):
        if i < n:
            # mirror the start of signal
            aux_signal[i] = signal[n-i]
            # mirror the end of signal
            aux_signal[len(signal)+n+i] = signal[len(signal)-i-2]
        aux_signal[i+n] = signal[i]
    for i in range(n, len(aux_signal)-n):
        smoothen[i-n] = mean(getValuesArray(aux_signal, i-n//2, i+n//2))
    return smoothen

def bsnbSmooth(input_signal, window_len=10, window='hanning'):
    sig = np.r_[input_signal[window_len:0:-1], input_signal, input_signal[-2:-window_len-2:-1]]
    win = np.ones(window_len, 'd')
    sig_conv = np.convolve(win / win.sum(), sig, mode='same')

    return sig_conv[window_len: -window_len]
    # return sig

def cumsum(data):
    value = data
    for i in range(1, len(data)):
        value[i] = value[i-1] + data[i]
    return value

def integrate(data, rate):
    wind_size = int(0.15*rate)
    int_ecg = np.zeros(len(data)-wind_size)
    cum_sum = cumsum(data)
    for i in range(len(int_ecg)):
        if i >= wind_size:
            int_ecg[i] = (cum_sum[i] - cum_sum[i-wind_size]) / wind_size
        else:
            int_ecg[i] = cum_sum[i] / (i+1)
    return int_ecg

def integration_DO(data, fs):
    # Moving window integration. N is the number of samples in the width of the integration window
    out = [ ]
    points = int(fs*0.15)
    for i in range(0, len(data)-points):
        temp = 0;
        for j in range(0, points):
            temp += data[i+j];
        
        out.append(temp/points);
    return out;


def peakDetection_(data, x_threshold=0, y_threshold=-np.inf, delta=0, proeminence=0):
    peaks = []
    peaks_pos = []
    prev_value = -1
    diff_data = differentation(data)
    i = 0
    while i < len(diff_data):
        if prev_value > 0 and diff_data[i] < 0 and data[i] > y_threshold or i == 0:
            if x_threshold > 0:
                if i + x_threshold < len(diff_data):
                    pos, peak = higherPeak(getValuesArray(data, i, i+x_threshold), getValuesArray(diff_data, i, i+x_threshold), y_threshold)

                else:
                    pos, peak = higherPeak(getValuesArray(data, i, len(diff_data)-1), getValuesArray(diff_data, i, len(diff_data)-1), y_threshold)
                peaks.append(peak)
                peaks_pos.append(i+pos)
                i += pos + x_threshold
            else:
                peaks.append(data[i])
                peaks_pos.append(i)
                i += 1
        if i < len(diff_data):
            prev_value = diff_data[i]
        i += 1
    return peaks, peaks_pos


def peakDetection(data, x_threshold=0, y_threshold=-np.inf, delta=0, proeminence=0):
    if proeminence > 0:
        peaksPos, peaks = calculateProeminencePeaks(data, proeminence)
    else:
        peaksPos, peaks = detectAllPeaks(data)

    peaksPos, peaks, valleysPos, valleys = correctPeaks(data, peaksPos)

    if x_threshold > 0:
        peaksToDelete = limitDistancePeaks(peaksPos, x_threshold)
        peaksPos = deletePos(peaksPos, peaksToDelete)
        peaks = deletePos(peaks, peaksToDelete)

        peaksPos, peaks, valleysPos, valleys = correctPeaks(data, peaksPos)

    return peaks, peaksPos, valleys, valleysPos


def median(array):
    array = np.sort(array)
    length = int(len(array))
    if(length%2 == 0):
        sumOfMiddleElements = array[int(length / 2)] +  array[int(length / 2 - 1)]
        medians = ( sumOfMiddleElements) / 2;
    else:
        medians = array[int(length/2)]
    return medians


def deletePos(data, deleteIndexes):
    deleteIndexes = list(set(deleteIndexes))
    deleteIndexes.sort()
    array = np.zeros(len(data) - len(deleteIndexes))
    aux = 0
    for i in range(len(data)):
        if aux < len(deleteIndexes):
            if i != deleteIndexes[aux]:
                array[i - aux] = data[i]
            else:
                aux += 1
        else:
            array[i - aux] = data[i]
    return array


def detectAllPeaks(data):
    peaks, peaksPos = [], []
    prev_value = data[0]
    for i in range(1, len(data)-1):
        if data[i-1] < data[i] and data[i+1] < data[i]:
            peaks.append(data[i])
            peaksPos.append(i)
    return peaksPos, peaks


def higherPeak(data, diff_data, y_threshold):
    peak, pos = data[0], 0
    prev_value = -1
    for i in range(len(diff_data)-1):
        if prev_value > 0 and diff_data[i] < 0 and data[i] > y_threshold:
            if peak != -1000 and data[i] > peak:
                peak = data[i]
                pos = i
        prev_value = diff_data[i]
    return pos, peak

def limitProeminences(data, peaksPos, threshold):
    peaksToDelete = []
    aux = 0
    for i in range(1, len(peaksPos)-1):
        proeminence = calculateProeminencePeaks(getValuesArray(data, peaksPos[aux], peaksPos[i+1]), peaksPos[i] - peaksPos[aux])
        print(proeminence, threshold)
        if threshold > 0 and proeminence < threshold:
            # print("  i: ", i, peaksPos[i] - peaksPos[aux], x_threshold)
            peaksToDelete.append(i)
        else:
            aux = i
    return peaksToDelete

def limitDistancePeaks(peaksPos, threshold):
    peaksToDelete = []
    aux = 0
    for i in range(1, len(peaksPos)):
        if threshold > 0 and peaksPos[i] - peaksPos[aux] < threshold:
            # print("  i: ", i, peaksPos[i] - peaksPos[aux], x_threshold)
            peaksToDelete.append(i)
        else:
            aux = i
    return peaksToDelete

def correctPeaks(data, peaksPos):
    if len(peaksPos) == 0:
        return [], [], [], []
    peaksPos = np.sort(peaksPos)
    if peaksPos[0] == 0:
        peaksPos = getValuesArray(peaksPos, 1, len(peaksPos))
    if len(peaksPos) == 0:
        return [], [], [], []
    if peaksPos[len(peaksPos) - 1] == len(peaksPos) - 1:
        peaksPos = getValuesArray(peaksPos, 0, len(peaksPos)-1)
    if len(peaksPos) == 0:
        return [], [], [], []
    elif len(peaksPos) == 1:
        peakPos, peak, valleyPos, valley = getCorrectPeak(data, peaksPos[0])
        return [peakPos], [peak], [valleyPos], [valley]

    new_peaksPos = np.empty(len(peaksPos))
    new_peaks = np.empty(len(peaksPos))
    valleysPos = np.empty(len(peaksPos))
    valleys = np.empty(len(peaksPos))

    for i in range(len(peaksPos)):
        if i == 0:
            if len(peaksPos) > 1:
                new_data = getValuesArray(data, 0, peaksPos[i+1])
            else:
                new_data = data
            peakPos, peak, valleyPos, valley = getCorrectPeak(new_data, peaksPos[i])
        elif i == len(peaksPos) - 1:
            new_data = getValuesArray(data, peaksPos[i-1], len(data))
            peakPos, peak, valleyPos, valley = getCorrectPeak(new_data, peaksPos[i] - peaksPos[i-1])
            peakPos += peaksPos[i-1]
            valleyPos += peaksPos[i-1]
        else:
            new_data = getValuesArray(data, peaksPos[i-1], peaksPos[i+1])
            peakPos, peak, valleyPos, valley = getCorrectPeak(new_data, peaksPos[i] - peaksPos[i-1])
            peakPos += peaksPos[i-1]
            valleyPos += peaksPos[i-1]
        new_peaksPos[i] = peakPos
        new_peaks[i] = peak
        valleysPos[i] = valleyPos
        valleys[i] = valley
    return [int(v) for v in new_peaksPos], new_peaks, [int(v) for v in valleysPos], valleys


def getCorrectPeak(data, peakPos):
    first_valley = minimumValue(getValuesArray(data, 0, peakPos))[0]
    second_valley = peakPos + minimumValue(getValuesArray(data, peakPos, len(data)))[0]
    peakPos, peak = maximumValue(getValuesArray(data, first_valley, second_valley))
    peakPos += first_valley

    # plt.figure()
    # plt.plot(data)
    # plt.scatter(peakPos, peak)
    # plt.show()

    return peakPos, peak, first_valley, data[first_valley]


def calculateProeminencePeaks(data, peakPos):
    first_proeminence = data[int(peakPos)] - minimumValue(getValuesArray(data, 0, peakPos))[1]
    second_proeminence = data[int(peakPos)] - minimumValue(getValuesArray(data, peakPos, len(data)))[1]
    return minimumValue([first_proeminence, second_proeminence])[1]

def calculateProeminencePeaks(data, threshold):
    peaksPos, peaks = [], []
    min_abs_value = minimumValue(data)[1]
    max_abs_value = maximumValue(data)[1]
    min_value = max_abs_value
    max_value = min_abs_value
    for i in range(1, len(data) - 1):
        if data[i] < min_value:
            min_value = data[i]
        if data[i] > max_value:
            max_value = data[i]
        if data[i-1] < data[i] and data[i] > data[i+1] and data[i] - min_value > threshold:
            peaksPos.append(i)
            peaks.append(data[i])
            max_value = min_abs_value
            min_value = max_abs_value
    # plt.figure()
    # plt.plot(data)
    # plt.scatter(peaksPos, peaks, marker='x', c='r')
    # plt.show()

    return peaksPos, peaks
        

def invertSignal(data):
    mean = np.mean(data)
    data = (-(data - mean)) + mean
    return data

def minimumValue(data):
    minimum = data[0]
    minPos = 0
    for i in range(len(data)):
        if data[i] < minimum:
            minimum = data[i]
            minPos = i
    return minPos, minimum

def maximumValue(data):
    maximum = data[0]
    maxPos = 0
    for i in range(len(data)):
        if data[i] > maximum:
            maximum = data[i]
            maxPos = i
    
    return maxPos, maximum

def detect_SpO2Peaks(red, infrared, fs):
    # filt_red = bsnb.lowpass(red, 5, 3, fs, True)
    # filt_infrared = bsnb.lowpass(infrared, 5, 3, fs, True)
    filt_red = bsnb.bandpass(red, .5, 5, 2, fs, True)
    filt_infrared = bsnb.bandpass(infrared, .5, 5, 2, fs, True)
    aux = filt_red + filt_infrared

    # Filter Signal
    if len(aux) > 1 * fs:
        baseline = bsnbSmooth(aux, fs*1)
    else:
        baseline = mean(aux)
    signal = aux - baseline
    aux = bsnbSmooth(signal, int(.3*fs))
    # aux = bsnb.lowpass(signal, 3, 2,  fs, True)

    peak, peakPos, valley, valleyPos = peakDetection(aux, proeminence=.9*np.std(aux), x_threshold=.3*fs)

    # plt.figure()
    # plt.plot(aux)
    # plt.scatter(peakPos, peak, marker='x', c='r')
    # plt.scatter(valleyPos, valley, marker='x', c='r')
    # plt.show()

    return filt_red, filt_infrared, peakPos, valleyPos


def calculate_SpO2(Rpeak, Rvalley, IRpeak, IRvalley):
    # r_ratio = (Rpeak - Rvalley) / (Rvalley + (Rpeak - Rvalley)/2)
    # r_ratio = (Rpeak - Rvalley) / (Rpeak + Rvalley)
    # r_ratio = (Rpeak - Rvalley) / (Rvalley)
    r_ratio = np.log10(Rpeak - Rvalley)
    # ir_ratio = (IRpeak - IRvalley) / (IRvalley + (IRpeak - IRvalley)/2)
    # ir_ratio = (IRpeak - IRvalley) / (IRpeak + IRvalley)
    # ir_ratio = (IRpeak - IRvalley) / (IRvalley)
    ir_ratio = np.log10(IRpeak - IRvalley)   
    
    R = r_ratio / ir_ratio
    # R = r_ratio - ir_ratio
    print(R)
    percentage = 110 - 25*R
    # percentage = 103.32 - 6.2033 * R
    # percentage = 109.06 + 6.3985 * (R**2) - 19.018 * R
    # r_o = 319.6
    # r_d = 3226.56
    # ir_o = 1204
    # ir_d = 602.24
    # percentage = np.abs((r_d - R * ir_d) * 100 / (R * (ir_o - ir_d) + (r_o - r_d)))
    return percentage

def calculate_SpO2_(Rpeak, Rvalley, IRpeak, IRvalley):
    red_AC = Rpeak - Rvalley
    red_DC = ((Rpeak - Rvalley) / 2) + Rvalley
    infrared_AC = IRpeak - IRvalley
    infrared_DC = ((IRpeak - IRvalley) / 2) + IRvalley
    R = (red_AC / red_DC) / (infrared_AC / infrared_DC)
    return max(0, min(100, 110 - 25*R))

def calculate_SpO2_____(Rpeak, Rvalley, IRpeak, IRvalley):
    r_ratio = np.log(np.abs(Rpeak/Rvalley))
    ir_ratio = np.log(np.abs(IRpeak / IRvalley))
    R = r_ratio / ir_ratio
    # percentage = 110 - 25*R
    percentage = 113.8 - 24.87*R
    # print("Percentage: ", percentage)
    return percentage

if __name__ == '__main__':
    from scipy.misc import electrocardiogram
    t = np.linspace(-50, 50, 1000)
    signal = np.sin(t*2*np.pi/1) * .3

    plt.figure()
    plt.plot(t, signal)
    plt.plot(t, invertSignal(signal))
    plt.show()

    plt.figure()
    plt.plot(electrocardiogram()[: 3600])
    plt.plot(invertSignal(electrocardiogram()[: 3600]))
    plt.show()