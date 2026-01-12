from physiological_data.lib.sensors import *


def Process_ECG(data, fs, resolution):
    sensor = ECG(data, fs, resolution)
    ecg = np.array(sensor.data)
    fs = sensor.fs
    resolution = sensor.resolution
    time = bsnb.generate_time(ecg, fs)

    peaks = sensor.processECG()

    return peaks, time


def Process_HRV(data, fs, resolution):
    sensor = HRV(data, fs, resolution)

    (
        heart_rate,
        time_features,
        poincare_features,
        frequency_features,
    ) = sensor.getFeatures()

    heart_rate_df = pd.DataFrame.from_dict(heart_rate, orient="columns")
    time_features_df = pd.DataFrame.from_dict(time_features, orient="columns")
    poincare_features_df = pd.DataFrame.from_dict(poincare_features, orient="columns")
    frequency_features_df = pd.DataFrame.from_dict(frequency_features, orient="columns")

    HRV_Dataframe = (
        (heart_rate_df.join(time_features_df)).join(poincare_features_df)
    ).join(frequency_features_df)

    return HRV_Dataframe


def Process_RESP(data, fs, resolution):
    sensor = RESP(data, fs, resolution)

    signals, info = sensor.process_RESP()

    # uncomment line below
    df = sensor.RESP_RRV(signals)

    resp_Dataframe = sensor.getFeatures(signals, df)

    # correct form is (signals,df)

    columns_to_remove = [
        "RRV_VLF",
        "RRV_LF",
        "RRV_HF",
        "RRV_ApEn",
        "RRV_SampEn",
        "RRV_LFHF",
        "RRV_LFn",
        "RRV_HFn",
        # "RRV_SD2",
        # "RRV_SD2SD1",
        "RRV_DFA_alpha2",
        "RRV_MFDFA_alpha2_Width",
        "RRV_MFDFA_alpha2_Peak",
        "RRV_MFDFA_alpha2_Mean",
        "RRV_MFDFA_alpha2_Max",
        "RRV_MFDFA_alpha2_Delta",
        "RRV_MFDFA_alpha2_Asymmetry",
        "RRV_MFDFA_alpha2_Fluctuation",
        "RRV_MFDFA_alpha2_Increment",
    ]

    for column in columns_to_remove:
        if column in resp_Dataframe.columns:
            resp_Dataframe = resp_Dataframe.drop(column, axis=1)

    return resp_Dataframe


def Process_EDA(data, fs, resolution):
    sensor = EDA(data, fs, resolution)

    (
        eda_phasic_dict,
        eda_tonic_dict,
        SCR_Amplitude_dict,
        SCR_RiseTime_dict,
        SCR_RecoveryTime_dict,
        frequency_features,
    ) = sensor.getFeatures()

    EDA_dict = {
        "Phasic_AVG": eda_phasic_dict["AVG"],
        "Phasic_MAX": eda_phasic_dict["Maximum"],
        "Phasic_MIN": eda_phasic_dict["Minimum"],
        "Phasic_STD": eda_phasic_dict["STD"],
        "Tonic_AVG": eda_tonic_dict["AVG"],
        "Tonic_MAX": eda_tonic_dict["Maximum"],
        "Tonic_MIN": eda_tonic_dict["Minimum"],
        "Tonic_STD": eda_tonic_dict["STD"],
        "SCR_Amp_AVG": SCR_Amplitude_dict["AVG"],
        "SCR_Amp_MAX": SCR_Amplitude_dict["Maximum"],
        "SCR_Amp_MIN": SCR_Amplitude_dict["Minimum"],
        "SCR_Amp_STD": SCR_Amplitude_dict["STD"],
        "SCR_Rt_AVG": SCR_RiseTime_dict["AVG"],
        "SCR_Rt_MAX": SCR_RiseTime_dict["Maximum"],
        "SCR_Rt_MIN": SCR_RiseTime_dict["Minimum"],
        "SCR_Rt_STD": SCR_RiseTime_dict["STD"],
        "SCR_Rect_AVG": SCR_RecoveryTime_dict["AVG"],
        "SCR_Rect_MAX": SCR_RecoveryTime_dict["Maximum"],
        "SCR_Rect_MIN": SCR_RecoveryTime_dict["Minimum"],
        "SCR_Rect_STD": SCR_RecoveryTime_dict["STD"],
    }

    EDA_Dataframe = (pd.DataFrame.from_dict(EDA_dict)).join(
        pd.DataFrame.from_dict(frequency_features)
    )
    if "LF/HF" in EDA_Dataframe.columns:
        EDA_Dataframe = EDA_Dataframe.drop(["LF/HF"], axis=1)

    return EDA_Dataframe


# def Process_EEG(data, fs, resolution):
#     EEG_dict = {}
#     EEG_filtered = {}
#     band_powers = {}
#     freqs = {}
#     power = {}
#
#     for keys in data.keys():
#         EEG_dict[keys] = EEG(data[keys], fs, resolution)
#         EEG_filtered[keys], freqs[keys], power[keys], band_powers[keys] = EEG_dict[
#             keys
#         ].getFeatures()
#
#     bands_df = pd.DataFrame.from_dict(band_powers, orient="index")
#
#     return bands_df

# def Process_fNIRS(data,fs,resolution):
#
#     sensor = fNIRS(data,fs,resolution)
#
#     sensor.processfNIRS()
#
#     fnirs_features = sensor.getFeatures()
#
#     fNIRS_Dataframe = pd.DataFrame.from_dict(fnirs_features,orient="columns")
#
#     return fNIRS_Dataframe

# def Process_TEMP(data, fs, resolution):
#     sensor = TEMP(data, fs, resolution)
#
#     temp = sensor.filterData()
#
#     Temp_Dataframe = sensor.getFeatures(temp)
#
#     return Temp_Dataframe
