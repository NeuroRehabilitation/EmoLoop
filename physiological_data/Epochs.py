import neurokit2 as nk
import numpy as np
import pandas as pd


def getMarkers(marker, timestamps):
    category, onset, offset = list(), list(), list()

    for timestamp, markers in zip(timestamps, marker):
        if markers[1] == "1":
            offset.append(timestamp)
        elif markers[0] == "end":
            offset.append(timestamp)
        else:
            onset.append(timestamp)
            category.append(markers[0])

    return onset, offset, category


@staticmethod
def getRatingsIndex(onset: list, time_opensignals: list) -> list:
    onset_index = list()

    if len(onset) > 0:
        for i in range(0, len(onset)):
            onset_index.append(np.where(time_opensignals >= onset[i])[0][0])

    return onset_index


@staticmethod
def getMarkersIndex(onset, offset, time_Opensignals):
    onset_index = list()
    offset_index = list()

    for i in range(0, len(onset)):
        onset_index.append(np.where(time_Opensignals >= onset[i])[0][0])
        offset_index.append(np.where(time_Opensignals <= offset[i])[0][-1])

    return onset_index, offset_index


def CalculateEventsDiff(onset, offset):
    events_diff = list()

    for i in range(0, len(onset)):
        events_diff.append(offset[i] - onset[i])

    return events_diff


def getOnset4sec(epochs):
    onset_4s = {}
    for keys in epochs.keys():
        onset_list = list()
        for i in range(0, len(epochs[keys]["Time"]), 4000):
            onset_list.append(i)
        onset_4s[keys] = onset_list

    return onset_4s


def getEpochs4sec(epochs, onset_4s, fs, epoch_duration=4):
    data = {}
    for keys in epochs.keys():
        d = {
            "Time": epochs[keys]["Time"],
            "ECG": epochs[keys]["ECG"],
            "EDA": epochs[keys]["EDA"],
            "EDA_Clean": epochs[keys]["EDA_Clean"],
            "EDA_Tonic": epochs[keys]["EDA_Tonic"],
            "EDA_Phasic": epochs[keys]["EDA_Phasic"],
        }
        df = pd.DataFrame(data=d)
        data[keys] = nk.epochs_create(
            df,
            events=onset_4s[keys],
            sampling_rate=fs,
            epochs_start=0,
            epochs_end=epoch_duration,
        )

    return data
