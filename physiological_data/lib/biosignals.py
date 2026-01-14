import os

try:
    from physiological_data.lib.tools import *
except (ImportError, ModuleNotFoundError):
    from physiological_data.lib.tools import *
from json import loads
import biosignalsnotebooks as bsnb
import abc


END_BASELINE = "End of Baseline"
BEGIN_BASELINE = "Begin of Baseline"
N_BACK = "N-back"
SUBTRACTION = "o"
REST_SUBTRACTION = "sub_baseline"

FNIRS = "hSpO2"
EEG = "EEG"
ACC = "XYZ"
RESP = "RESPIRATION"
ECG = "ECG"
EDA = "EDA"
PUSHBUTTON = "CUSTOM/0.5/1.0/V"

ALL = [FNIRS, EEG, ACC, RESP, ECG, EDA]
TASK = [N_BACK, SUBTRACTION, REST_SUBTRACTION]


class Devices:
    def __init__(self, path):
        self.path = path
        self.data, self.header = self.read_bio_data(self.path)
        self.macs = list(self.header.keys())
        self.fs = self.header[self.macs[0]]["sampling rate"]
        self.pushButton = []
        self.resolution = int(self.header[self.macs[0]]["resolution"][0])
        self.time = self.data[:, 0]
        self.sensorData = None
        self.sensors = None

    def read_bio_data(self, path):
        signals = []
        header = ""
        for file in os.listdir(path):
            if "modified.txt" in file and "converted_modified.txt" not in file:
                print(os.path.join(path, file))
                file_path = os.path.join(path, file)
                header = self.readHeader(file_path)
                data = np.loadtxt(file_path)
                signals.append(data)
        if np.shape(signals)[0] > 1:
            return np.vstack(signals), header
            # return signals, header
        else:
            return signals[0], header

    @staticmethod
    def readHeader(path):
        with open(path, "r") as f:
            lines = f.readlines()[1]
        return loads(lines[2:])

    @staticmethod
    def convertAndfilterSignal(data, sensor, fs, resolution):
        if sensor == "ECG":
            low_f, high_f = 5, 15
            data = ((data / (2**resolution)) - 1 / 2) * 3000 / 1019  # millivolts
        elif sensor == "hSpO2":
            low_f, high_f = 0.05, 1
            data = (0.15 * data) / (2**resolution)  # microAmpere
        elif sensor == "EDA":
            low_f, high_f = 0.016, 35
            data = 1e6 * (data / (2**resolution)) * 3 / 0.12  # microSiemens
        elif sensor == "EEG":
            # CONFIRM
            low_f, high_f = 1, 30
            data = 1e6 * (data / ((2**resolution) - 1) - 0.5) * 3 / 40000  # microVolts
        elif sensor == "RESPIRATION":
            low_f, high_f = 0.01, 1
            data = ((data / (2**resolution)) - 1 / 2) * 100  # % of displacement
        elif sensor == "XYZ":
            # CONFIRM - Movements happening at more than 1 per second is very unlikely...
            low_f, high_f = 0.01, 10
            # The conversion requires a calibration step
        return bsnb.bandpass(data, low_f, high_f, fs=fs, use_filtfilt=True)

    def convertSensors(self):
        if type(self.sensorData) == type(None) or type(self.sensors) == type(None):
            print(
                "Use getSensorsData before applying the filters and conversion to physical units."
            )
            return
        if len(self.sensors) == self.sensorData.shape[1]:
            aux = 0
        if len(self.sensors) == self.sensorData.shape[1] - 1:
            aux = 1
        for i, sensor in enumerate(self.sensors):
            self.sensorData[:, i + aux] = self.convertAndfilterSignal(
                self.sensorData[:, i + aux], sensor, self.fs, self.resolution
            )
        return self.sensorData.copy()

    def segmentSignalsIndex(self, signal, timestamps, labels, binary=False):
        """
        Segment arrays according to PsychoPy indexes
        """
        signalArray = np.array(signal)
        indexes = self.getSegmentationIndexes(signalArray[:, 0], timestamps)
        segmentedSignal = []
        segmentLabels = []
        for i in range(1, len(indexes[1:])):
            # This interval corresponds to reading instructions
            if not (labels[i - 1] == END_BASELINE and labels[i] == BEGIN_BASELINE):
                segmentLabels.append(labels[i - 1])
                segmentedSignal.append(signalArray[indexes[i - 1] : indexes[i]])

        if binary:
            return segmentedSignal, self.convertBinaryLabels(segmentLabels)
        else:
            return segmentedSignal, segmentLabels

    def segmentSignalsWindowing(
        self,
        signal,
        timestamps,
        labels,
        timeWindow=1,
        overlap=0,
        binary=False,
        normalize=False,
        returnTime=False,
    ):
        signalArray = np.array(signal)
        signalArray = self.removeNonPsycho(signal, timestamps)
        if normalize:
            signalArray = self.normalizeSensors(signalArray, True)
        segments = windowing(
            signalArray, sampling_rate=self.fs, time_window=timeWindow, overlap=overlap
        )
        print(f"Segments Shape: {segments.shape}")
        segments, segmentLabels = self.makeLabels(segments, labels, timestamps)

        if binary:
            segmentLabels = self.convertBinaryLabels(segmentLabels)
        print(
            f"Shape of segments before clean is {segments.shape}, of labels {np.shape(segmentLabels)}"
        )
        segments, segmentLabels = self.cleanSegments(segments, segmentLabels)
        print(
            f"Shape of segments after clean is {segments.shape}, of labels {segmentLabels.shape}"
        )
        if returnTime:
            return segments[:, :, 1:], segmentLabels, segments[:, :, 0]
        return segments[:, :, 1:], segmentLabels

    def makeLabels(self, segments, labels, timestamps):
        """
        Make labels based on the timestamps and delete the segments that have more than one label or which label is
        'End of Baseline'.
        :return:
        """
        segmentLabels, deleteIndexes = [], []
        aux = 0
        total = len(segments)
        for i in range(len(segments)):
            print(f"{(i/total)*100:.1f}%", end="\r")
            # Check the label for each timestamp
            currentLabel = labels[
                closestIndex(timestamps, segments[i, 0, 0])
            ]  # first label
            lastLabel = labels[
                closestIndex(timestamps, segments[i, -1, 0])
            ]  # last label
            if (currentLabel != lastLabel) or (currentLabel == END_BASELINE):
                deleteIndexes.append(i)
            else:
                # print(f"{currentLabel}", end='\r')
                segmentLabels.append(currentLabel)

        return np.delete(segments, deleteIndexes, axis=0), segmentLabels

    def convertBinaryLabels(self, labels):
        new_labels = []
        for label in labels:
            if label != END_BASELINE:
                if "baseline" in label.lower():
                    new_labels.append("baseline")
                elif label in TASK:
                    new_labels.append("task")
                else:
                    new_labels.append("other")
        return new_labels

    def getSensorData(self, sensor=FNIRS, rightMAC=None):
        indexes = {"right": [], "left": []}
        sensorsOut = {"right": [], "left": []}
        if not rightMAC:
            indexes, sensorsOut = [], []

        for i, mac in enumerate(self.header.keys()):
            if i == 0:
                aux = 2  # Timestamps and Digital
                nSensors = len(
                    self.header[mac]["sensor"]
                )  # get number of sensors to sum in the else statement
            else:
                aux = 4 + nSensors  # Timestamps and Digital from both devices
            index = strIndex(self.header[mac]["sensor"], sensor) + aux
            if len(index) != 0:
                if not rightMAC:
                    indexes.append(index)
                    sensorsOut.append([sensor] * self.nColumns(sensor))
                elif mac == rightMAC:
                    indexes["right"].append(index)
                    sensorsOut["right"].append([sensor] * self.nColumns(sensor))
                else:
                    indexes["left"].append(index)
                    sensorsOut["left"].append([sensor] * self.nColumns(sensor))
        if not rightMAC:
            indexes = np.concatenate(indexes)
            return self.data[:, indexes], sensorsOut
        else:
            if len(indexes["right"]) != 0:
                right_data = self.data[:, np.concatenate(indexes["right"])]
            else:
                right_data = []
            if len(indexes["left"]) != 0:
                left_data = self.data[:, np.concatenate(indexes["left"])]
            else:
                left_data = []
            return {"right": right_data, "left": left_data}, sensorsOut

    def getSensorsData(self, sensors=("EEG", "ECG", "RESPIRATION"), rightMAC=None):
        if not rightMAC:
            data = self.time.reshape(-1, 1)
            sensorsOut = []
            for sensor in sensors:
                sensor_data, sensors_result = self.getSensorData(sensor, rightMAC)
                data = np.hstack([data, sensor_data])
                sensorsOut += sensors_result
            self.sensorData = data
            self.sensors = sensorsOut
            return {"data": data, "sensors": sensorsOut}
        else:
            final, sensors_final = self.time.reshape(-1, 1), []
            data = {"right": [], "left": []}
            sensorsOut = {"right": [], "left": []}
            for sensor in sensors:
                sensor_data, sensors_result = self.getSensorData(sensor, rightMAC)
                if len(sensor_data["right"]) > 0:
                    data["right"].append(sensor_data["right"])
                    sensorsOut["right"].append(sensors_result["right"])
                if len(sensor_data["left"]) > 0:
                    data["left"].append(sensor_data["left"])
                    sensorsOut["left"].append(sensors_result["left"])

            try:
                data["right"] = np.hstack(data["right"])
            except Exception as e:
                print(e)
            try:
                data["left"] = np.hstack(data["left"])
            except Exception as e:
                print(e)

            if len(data["left"]) == 0 and len(data["right"]) != 0:
                final = np.hstack([final, data["right"]])
            elif len(data["right"]) == 0 and len(data["left"]) != 0:
                final = np.hstack([final, data["left"]])
            else:
                final = np.hstack([final, data["left"], data["right"]])
            sensors_final = sensorsOut["left"] + sensorsOut["right"]
            # TODO Order sensors - First the ones that are not left or right, then the one on the left and finally the ones on the right
            final, sensors_final = self.orderSensors(final, sensors_final)
            print("Sensors: ", sensors_final)

            self.sensorData = final
            self.sensors = sensors_final
            return {"data": final, "sensors": sensors_final}

    def orderSensors(self, data, sensors):
        sensors_names = []
        bipolar = []
        other = []
        aux = 0
        final_data = self.time.reshape(-1, 1)
        final_sensors = []
        for i in range(len(sensors)):
            for j in range(len(sensors[i])):
                for sensor in sensors[i][j]:
                    sensors_names.append(sensor)
                    if FNIRS == sensor or EEG == sensor:
                        bipolar.append(aux)
                    else:
                        other.append(aux)
                    aux += 1
        ordering = np.array(bipolar + other)
        for index in ordering:
            final_data = np.hstack([final_data, data[:, index + 1].reshape(-1, 1)])
            final_sensors.append(sensors_names[index])
        return final_data, final_sensors

    def getPushButtonData(self, read=False):
        if read or self.pushButton == []:
            for i, mac in enumerate(self.header.keys()):
                if i == 0:
                    aux = 2  # Timestamps and Digital
                    nSensors = len(
                        self.header[mac]["sensor"]
                    )  # get number of sensors to sum in the else statement
                else:
                    aux = 4 + nSensors  # Timestamps and Digital from both devices
                index = strIndex(self.header[mac]["sensor"], PUSHBUTTON) + aux
                if len(index) != 0:
                    self.pushButton = np.hstack(
                        [np.reshape(self.data[:, 0], (-1, 1)), self.data[:, index]]
                    )
        return self.pushButton

    def getFinalSensorsData(
        self, sensors=("EEG", "ECG", "RESPIRATION"), ordered=True, rightPos=None
    ):
        indexes = [np.array([0])]
        sensorsOut = [["nSeq"]]
        for i, mac in enumerate(self.header.keys()):
            for sensor in sensors:
                if i == 0:
                    aux = 2  # Timestamps and Digital
                    nSensors = len(
                        self.header[mac]["sensor"]
                    )  # get number of sensors to sum in the else statement
                else:
                    aux = 4 + nSensors  # Timestamps and Digital from both devices
                index = strIndex(self.header[mac]["sensor"], sensor) + aux
                if len(index) != 0:
                    indexes.append(index)
                    sensorsOut.append([sensor] * self.nColumns(sensor))

        indexes, sensorsOut = np.concatenate(indexes), np.concatenate(sensorsOut)
        if ordered:
            indexes[1:], sensorsOut[1:] = self.orderSensors(indexes[1:], sensorsOut[1:])
        return {"data": self.data[:, indexes], "sensors": sensorsOut}

    def getSegmentationIndexes(self, signal, timestamps):
        signalArray = self.removeNonPsycho(signal, timestamps)
        return np.array([closestIndex(signalArray, i) for i in timestamps])

    def removeNonPsycho(self, signal, timestamps):
        signalArray = np.array(signal)
        return signalArray[
            closestIndex(signalArray[:, 0], timestamps[0]) : closestIndex(
                signalArray[:, 0], timestamps[-1]
            )
        ]

    def normalizeSensors(self, data, firstTime=True):
        if firstTime:
            data[:, 1:] = data[:, 1:] - np.mean(data[:, 1:], axis=0)
            data[:, 1:] = data[:, 1:] / np.std(data[:, 1:], axis=0)
            return data
        else:
            data = data - np.mean(data, axis=0)
            return data / np.std(data, axis=0)

    def nColumns(self, sensor):
        if sensor == ACC:
            return 3
        elif sensor == FNIRS:
            return 2
        else:
            return 1

    def getPushButtonEvents(self):
        data = self.getPushButtonData()
        diff_data = np.diff(data[:, -1])
        indexes = np.where(diff_data > 20000)[0]
        return data[indexes, 0]

    def deleteArtifacts(self, singles, doubles):
        """
        Single push button presses have limited duration defined in the description file, while "doubles" corresponds
        to two presses that are related: the first marks the beginning of an event and the second marks the end.

        :param singles:
        :param doubles:
        :return:
        """
        if len(singles) != 0 or len(doubles) != 0:

            events = self.getPushButtonEvents()
            indexes = []
            for i, value in singles.items():
                index = np.where(self.data[:, 0] == events[int(i)])[0]
                if index + (value * self.fs) < self.data.shape[0]:
                    indexes.append(np.arange(index, index + (value * self.fs)))
                else:
                    indexes.append(np.arange(index, self.data.shape[0]))

            for first, second in doubles:
                first_index = np.where(self.data[:, 0] == events[first])[0]
                second_index = np.where(self.data[:, 0] == events[second])[0]
                indexes.append(np.arange(first_index, second_index))

            indexes = np.concatenate(indexes)
            # self.data[indexes, 1:] = 0
            self.data = np.delete(self.data, indexes, axis=0)
            self.time = self.data[:, 0].copy()
        return self.data.copy()

    def cleanSegments(self, segments, labels):
        new_segments = segments
        new_labels = labels
        aux = 0
        for i, segment in enumerate(segments):
            if any(np.diff(segment[:, 0]) > 2 / self.fs):
                new_segments = np.delete(new_segments, i - aux, axis=0)
                new_labels = np.delete(new_labels, i - aux, axis=0)
                aux += 1
        return new_segments, new_labels


class Sensor:
    def __init__(self, data, fs, resolution):
        self.data = data
        self.fs = fs
        self.resolution = resolution

    def convertPhys(self):
        pass

    @staticmethod
    def statistical_Features(signal):

        try:
            maximum = np.nanmax(signal)
        except Exception as e:
            print(e)
            print("Error on Maximum")
            maximum = np.nan
        try:
            minimum = np.nanmin(signal)
        except Exception as e:
            print(e)
            print("Error on Minimum")
            minimum = np.nan
        try:
            average = np.nanmean(signal)
        except Exception as e:
            print(e)
            print("Error on Average")
            average = np.nan
        try:
            std = np.nanstd(signal)
        except Exception as e:
            print(e)
            print("Error on STD")
            std = np.nan

        statistical_features = {
            "AVG": [average],
            "Maximum": [maximum],
            "Minimum": [minimum],
            "STD": [std],
        }

        return statistical_features

    @abc.abstractmethod
    def getFeatures(self):
        pass


# if __name__ == '__main__':
#     device = Devices(r'..\..\acquisitions\Acquisitions\03_11_2020')
#     print(device.getSensorsData())
