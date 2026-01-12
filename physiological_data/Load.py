def Load_PsychopyMarkers(data, stream_name: str):
    markers, timestamps = list(), list()
    for stream in data:
        if stream["info"]["name"][0] == stream_name:
            markers = list(stream["time_series"])
            timestamps = list(stream["time_stamps"])

    return markers, timestamps


def Load_Ratings(data, stream_name: str):
    arousal_timestamps, valence_timestamps, valence, arousal = (
        list(),
        list(),
        list(),
        list(),
    )

    ratings = {}

    for stream in data:
        if stream["info"]["name"][0] == stream_name:
            timeseries = list(stream["time_series"])

            for i, item in enumerate(timeseries):
                if item[0] == "Valence":
                    valence_timestamps.append(stream["time_stamps"][i])
                if item[0] == "Arousal":
                    arousal_timestamps.append(stream["time_stamps"][i])
                if float(item[1]) < 4:
                    if item[0] == "Valence":
                        valence.append("Low")
                    else:
                        arousal.append("Low")
                elif float(item[1]) >= 4 and float(item[1]) < 7:
                    if item[0] == "Valence":
                        valence.append("Medium")
                    else:
                        arousal.append("Medium")
                else:
                    if item[0] == "Valence":
                        valence.append("High")
                    else:
                        arousal.append("High")

        ratings = {"Valence": valence, "Arousal": arousal, "Valence Timestamps": valence_timestamps,
                   "Arousal Timestamps": arousal_timestamps}

    return ratings


def Load_Opensignals(data, stream_name: str):
    Opensignals_Data = {}
    fs = int()

    for stream in data:
        if stream["info"]["name"][0] == stream_name:
            Opensignals_Data["time_Opensignals"] = stream["time_stamps"]
            fs = int(stream["info"]["nominal_srate"][0])
            for i in range(0, int(stream["info"]["channel_count"][0]) - 1):
                Opensignals_Data["CH" + str(i + 1)] = stream["time_series"][:, i + 1]

    return Opensignals_Data, fs


def Load_EEG(data, stream_name: str):
    EEG_data = {}
    time_EEG = []
    EEG_fs = int()

    for stream in data:
        if stream["info"]["name"][0] == stream_name:
            for i in range(0, 32):
                # print(i)
                EEG_data["EEG_" + str(i + 1)] = stream["time_series"][:, i]
            time_EEG = stream["time_stamps"]
            EEG_fs = int(stream["info"]["nominal_srate"][0])

    return EEG_data, time_EEG, EEG_fs
