try:
    from acquisition import *
except (ImportError, ModuleNotFoundError):
    from lib.acquisition import *
from os import listdir
from os.path import join

import matplotlib.pylab as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

AUDIO = 'audio.wav'
KEYBOARD = 'keyboard.csv'
MOUSE = 'mouse.csv'
SCREENSHOT = 'screenshot.csv'
SNAPSHOT = 'snapshot.csv'
files = [AUDIO, KEYBOARD, MOUSE, SCREENSHOT, SNAPSHOT]


class Latent:
    def __init__(self, pathFolder, fs):
        self.path = self.getPath(pathFolder)
        self.data = {}
        self.fs = fs

    def getPath(self, pathFolder):
        for folder in listdir(pathFolder):
            if 'ip' in folder:
                return join(pathFolder, folder)

    def readLatentFile(self, file):
        end = file[-4:]
        if '.csv' == end:
            data = read_csv(join(self.path, file))
            return data
        elif '.wav' == end:
            sampling_ratem, data = wavfile.read(join(self.path, AUDIO))
            return data, sampling_ratem
        return []

    def readData(self):
        for file in files:
            if AUDIO != file:
                self.data[file.split('.')[0]] = self.readLatentFile(file)
            else:
                self.data[file.split('.')[0]], self.audioFS = self.readLatentFile(file)

    @staticmethod
    def makeSpectrogram(signal, fs):
        f, t, Sxx = spectrogram(signal, fs)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        return f, t, Sxx

    def syncInterfaces(self):
        pass

    def interpolateInterfaces(self, interface):
        columns, keys = self.identifyTypeInterp(interface)
        # time = np.arange(self.data[interface].time[0], self.data[interface].time[1]+1, T)
        time_DF = np.array(self.data[interface].time)
        time = np.linspace(time_DF[0], time_DF[-1], int(np.round((time_DF[-1] - time_DF[0]) * self.fs)))

        time[0] = self.data[interface].time[0]
        data = np.empty((len(time), len(self.data[interface].keys())), dtype=np.object)
        data[:] = np.nan

        data[:, 0] = time
        data = self.duplicateValues(self.data[interface], interface, data, time_DF, keys['duplicate'], columns['duplicate'])
        data = self.fillValues(self.data[interface], interface, data, time_DF, keys['fill']+keys['interpolate'], columns['fill'] + columns['interpolate'])
        data = self.interpolateValues(self.data[interface], interface, data, time_DF, keys['interpolate'], columns['interpolate'])
        return data

    def duplicateValues(self, duplicate_data, interface, new_data, data, keys, columns):
        time = new_data[:, 0]
        duplicate_index = 0
        for i in range(len(time)):
            print(f"Duplicating {interface} Data: {i*100/len(time):.2f}%", end='\r')
            try:
                while np.abs(data[duplicate_index + 1] - time[i]) < np.abs(data[duplicate_index] - time[i]):
                    duplicate_index += 1
            except IndexError:
                duplicate_index = len(data) - 1
            # duplicate_index += closestIndex(data[duplicate_index:], time[i])
            for j, key in enumerate(keys):
                new_data[i, columns[j]] = duplicate_data[key][duplicate_index]
        print(' ')
        return new_data

    def fillValues(self, fill_data, interface, new_data, data, keys, columns):
        prev_fill_index = 0
        time = new_data[:, 0]
        fill_index = 0
        for i, value in enumerate(data):
            print(f"Filling {interface} Data: {i*100/len(data):.2f}%", end='\r')
            try:
                while np.abs(time[fill_index + 1] - value) < np.abs(time[fill_index] - value):
                    fill_index += 1
            except IndexError:
                fill_index = len(time) - 1
            if prev_fill_index != fill_index:
                for j, key in enumerate(keys):
                    new_data[fill_index, columns[j]] = fill_data[key][i]
            prev_fill_index = fill_index
        print(' ')
        return new_data
    
    def interpolateValues(self, interpolate_data, interface, new_data, data, keys, columns):
        prev_index, index = -1, 0
        time = new_data[:, 0]
        not_nan_indexes = self.not_nan(new_data[:, columns[1]])
        T = 1/self.fs
        m, b, x0, x1, y0, y1 = 0, 0, 0, 0, 0, 0
        for j, key in enumerate(columns[1:]):
            for i in range(1, len(new_data)):
                print(f"Interpolating {interface} Data: {i*100/len(new_data):.2f}%", end='\r')
                
                if np.isnan(new_data[i, key]):
                    new_value = m * time[i] + b
                    new_data[i, key] = new_value
                else:
                    if index + 1 < len(not_nan_indexes):
                        x0 = time[not_nan_indexes[index]]
                        x1 = time[not_nan_indexes[index+1]]
                        y0 = new_data[not_nan_indexes[index], key]
                        y1 = new_data[not_nan_indexes[index+1], key]
                        m = (y1 - y0) / (x1 - x0)
                        b = y0 - m * x0
            
                try:
                    prev_index = index
                    if i >= not_nan_indexes[index]:
                        index += 1
                except IndexError:
                    index = len(not_nan_indexes) - 1
        print(' ')
        return new_data
    
    @staticmethod
    def not_nan(data):
        index = []
        for i, value in enumerate(data):
            if not np.isnan(value):
                index.append(i)
        return index            

    def identifyTypeInterp(self, interface):
        if interface.lower() == 'mouse':
            return {'interpolate': [0, 7, 8, 9, 10, 14], 'duplicate': [1, 2, 13], 'fill': [3, 4, 5, 6, 11, 12]}, {'interpolate': ['time', 'screen_x', 'screen_y', 'page_x', 'page_y', 'event_timestamp'], 'duplicate': ['type', 'tab_id', 'xpath'], 'fill': ['button', 'shift_key', 'alt_key', 'ctrl_key', 'delta_y', 'scrolltop']}
        elif interface.lower() == 'keyboard': 
            return {'interpolate': [0, 8], 'duplicate': [1, 2], 'fill': [3, 4, 5, 6, 7]}, {'interpolate': ['time', 'event_timestamp'], 'duplicate': ['type', 'tab_id'], 'fill': ['key_code', 'shift_key', 'alt_key', 'ctrl_key', 'capslock_key']}

if __name__ == '__main__':
    latent = Latent(r"D:\Google Drive\Faculdade\Doutoramento\Acquisitions\new_biosignalsnotebooks\acquisitions\Acquisitions\03_11_2020", 100)
    latent.readData()
    data = latent.interpolateInterfaces('keyboard')
    print(data[:, 7])
    plt.figure()
    plt.scatter(latent.data['keyboard'].time, latent.data['keyboard'].key_code)
    plt.scatter(data[:, 0], data[:, 3], color='r')
    plt.show()
    # latent.interpolateInterfaces(1000, 'keyboard')
    # data, fs = latent.readLatentFile(AUDIO)
    # latent.makeSpectrogram(data[::1000, 0], fs/1000)

