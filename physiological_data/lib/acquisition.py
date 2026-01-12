import numpy as np
import os
from json import load

try:
    from lib.biosignals import *
    from lib.psychopy import *
except (ImportError, ModuleNotFoundError):
    from lib.biosignals import *
    from lib.psychopy import *


class Acquisition:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.participantID = self.dir_path.split(os.sep)[-1]
        self.psychoFile = self.getPsychoFile(self.dir_path)
        self.psycho = Psycho(self.psychoFile)
        self.biosignals = Devices(self.dir_path)
        self.segmentedBiosignals = []
        self.features = []
        self.labels = []
        self.sensors = []
        self.signal = []
        self.description = None
        try:
            with open(os.path.join(dir_path, "description.json"), "r") as file:
                self.description = load(file)
        except FileNotFoundError as e:
            print(e, "The description file does not exist and will not be loaded.")

    @staticmethod
    def getPsychoFile(path):
        for file in os.listdir(path):
            if "results_" in file and ".csv" in file:
                return os.path.join(path, file)

    def getPsychoTimestamps(self):
        return self.psycho.getTimestamps()

    def getPsychoActivity(self):
        return self.psycho.getActivity()

    def getBiosignalsSensors(self, sensors=(FNIRS, ACC, EDA), rightPos=None):
        rightMAC = None
        if rightPos:
            for key in self.description["Position"].keys():
                if key != "left":
                    rightMAC = self.description["Position"][key]

        data = self.biosignals.getSensorsData(sensors, rightMAC)
        self.signal = data["data"]
        self.sensors = data["sensors"]

    def convertSensors(self):
        self.signal = self.biosignals.convertSensors()

    def segmentWindowingBiosignals(
        self, signal, timestamps, labels, timeWindow=0.1, overlap=0, binary=True
    ):
        self.segmentedBiosignals, self.labels = self.biosignals.segmentSignalsWindowing(
            signal, timestamps, labels, timeWindow, overlap, binary
        )

    def extractFeatures(self, segments, featuresFunc=[np.mean, np.std, np.max]):
        self.features = []
        for segment in segments:
            features = []
            for column in range(np.shape(segment)[1]):
                for func in featuresFunc:
                    features.append(func(segment[:, column]))
            self.features.append(features)
        self.features = np.array(self.features)

    def getParticipantIDClassification(self):
        return [self.dir_path.split(os.sep)[-1]] * len(self.labels)

    def getDataset(self):
        return (
            self.features,
            np.reshape(self.labels, (-1, 1)),
            np.reshape(self.getParticipantIDClassification(), (-1, 1)),
        )

    def deleteArtifacts(self):
        if self.description != None:
            self.signal = self.biosignals.deleteArtifacts(
                self.description["Push Button"]["singles"],
                self.description["Push Button"]["doubles"],
            )
        else:
            print(f"description.json file is not available for the acquisition.")


# if __name__ == '__main__':
#     data = Acquisition(r'..\..\acquisitions\Acquisitions\03_11_2020')
