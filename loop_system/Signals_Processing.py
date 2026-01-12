import time

import pyxdf
from colorama import Fore
from sklearn import svm, naive_bayes, ensemble, feature_selection
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add other content roots to sys.path
# sys.path.append(os.path.abspath("..//Study1//Physiological_Data//"))
from physiological_data.Epochs import *
from physiological_data.Load import *
from physiological_data.Process_Features import *

# ANSI escape code for blue
BLUE = "\033[94m"


@staticmethod
def Run_files(fname):
    """

    :param fname:
    :type fname:
    :return:
    :rtype:
    """
    data, header = pyxdf.load_xdf(fname)

    return data


@staticmethod
def change_key(dictionary, key_to_remove, new_key):
    """

    :param dictionary:
    :type dictionary:
    :param key_to_remove:
    :type key_to_remove:
    :param new_key:
    :type new_key:
    """
    dictionary[new_key] = dictionary.pop(key_to_remove)


@staticmethod
def Load_Data(
    data,
    openSignals_stream_name: str,
    markers_stream_name: str,
    ratings_stream_name: str,
    sensors: list,
) -> dict:
    """

    :param data:
    :type data:
    :param openSignals_stream_name:
    :type openSignals_stream_name:
    :param markers_stream_name:
    :type markers_stream_name:
    :param ratings_stream_name:
    :type ratings_stream_name:
    :param sensors:
    :type sensors:
    :return:
    :rtype:
    """
    marker, timestamps = Load_PsychopyMarkers(data, markers_stream_name)
    opensignals_data, fs = Load_Opensignals(data, openSignals_stream_name)
    ratings = Load_Ratings(data, ratings_stream_name)

    change_key(opensignals_data, "time_Opensignals", "Time"), change_key(
        opensignals_data, "CH1", sensors[0]
    ),
    change_key(opensignals_data, "CH2", sensors[1]), change_key(
        opensignals_data, "CH3", sensors[2]
    )

    Signals = pd.DataFrame(data=opensignals_data)

    data = {
        "Signals": Signals,
        "Markers": marker,
        "Markers Timestamps": timestamps,
        "Ratings": ratings,
    }

    return data


@staticmethod
def getSignals(
    data,
    openSignals_stream_name: str,
    markers_stream_name: str,
    ratings_stream_name: str,
    sensors: list,
) -> dict:
    """

    :param data:
    :type data:
    :param openSignals_stream_name:
    :type openSignals_stream_name:
    :param markers_stream_name:
    :type markers_stream_name:
    :param ratings_stream_name:
    :type ratings_stream_name:
    :param sensors:
    :type sensors:
    :return:
    :rtype:
    """
    signals = {}

    for condition in data:
        signals[condition] = Load_Data(
            data[condition],
            openSignals_stream_name,
            markers_stream_name,
            ratings_stream_name,
            sensors,
        )

    return signals


@staticmethod
def getEvents(signals) -> (dict, dict):
    """

    :param signals:
    :type signals:
    :return:
    :rtype:
    """
    epochs_markers, ratings = {}, {}

    for condition in signals:
        onset, offset, marker = getMarkers(
            signals[condition]["Markers"], signals[condition]["Markers Timestamps"]
        )
        onset_index, offset_index = getMarkersIndex(
            onset, offset, signals[condition]["Signals"]["Time"]
        )
        events_diff = CalculateEventsDiff(onset, offset)
        epochs_markers[condition] = {
            "Onset": onset,
            "Onset_index": onset_index,
            "Offset": offset,
            "Offset_index": offset_index,
            "EventsDiff": events_diff,
            "Marker": marker,
        }

        arousal_index = getRatingsIndex(
            signals[condition]["Ratings"]["Arousal Timestamps"],
            signals[condition]["Signals"]["Time"],
        )
        valence_index = getRatingsIndex(
            signals[condition]["Ratings"]["Valence Timestamps"],
            signals[condition]["Signals"]["Time"],
        )
        ratings[condition] = {
            "Arousal": signals[condition]["Ratings"]["Arousal"],
            "Arousal Timestamps": signals[condition]["Ratings"]["Arousal Timestamps"],
            "Arousal_index": arousal_index,
            "Valence": signals[condition]["Ratings"]["Valence"],
            "Valence Timestamps": signals[condition]["Ratings"]["Valence Timestamps"],
            "Valence_index": valence_index,
        }

    return epochs_markers, ratings


@staticmethod
def getSignalsEpochs(
    signals: dict, epochs_markers: dict, ratings: dict, window: int, fs: int
) -> dict:
    """

    :param signals:
    :type signals:
    :param epochs_markers:
    :type epochs_markers:
    :param ratings:
    :type ratings:
    :param window:
    :type window:
    :param fs:
    :type fs:
    :return:
    :rtype:
    """
    epochs_data = {}

    for condition in epochs_markers:
        events = np.arange(
            epochs_markers[condition]["Onset_index"][0],
            epochs_markers[condition]["Offset_index"][-1] - window * fs,
            fs,
        )

        match condition:
            case "Baseline":
                arousal_values = ["Low" for _ in range(len(events))]
                valence_values = ["Low" for _ in range(len(events))]

                epochs_signals = nk.epochs_create(
                    pd.DataFrame.from_dict(signals[condition]["Signals"]),
                    events=events,
                    sampling_rate=fs,
                    epochs_start=0,
                    epochs_end=window,
                )
            case "NBack":
                one_back_markers = {
                    "Marker": [
                        marker
                        for marker in epochs_markers[condition]["Marker"]
                        if "1-Back" in marker
                    ],
                    "Onset_index": [
                        index
                        for index, marker in zip(
                            epochs_markers[condition]["Onset_index"],
                            epochs_markers[condition]["Marker"],
                        )
                        if "1-Back" in marker
                    ],
                    "Offset_index": [
                        index
                        for index, marker in zip(
                            epochs_markers[condition]["Offset_index"],
                            epochs_markers[condition]["Marker"],
                        )
                        if "1-Back" in marker
                    ],
                }
                two_back_markers = {
                    "Marker": [
                        marker
                        for marker in epochs_markers[condition]["Marker"]
                        if "2-Back" in marker
                    ],
                    "Onset_index": [
                        index
                        for index, marker in zip(
                            epochs_markers[condition]["Onset_index"],
                            epochs_markers[condition]["Marker"],
                        )
                        if "2-Back" in marker
                    ],
                    "Offset_index": [
                        index
                        for index, marker in zip(
                            epochs_markers[condition]["Offset_index"],
                            epochs_markers[condition]["Marker"],
                        )
                        if "2-Back" in marker
                    ],
                }
                four_back_markers = {
                    "Marker": [
                        marker
                        for marker in epochs_markers[condition]["Marker"]
                        if "4-Back" in marker
                    ],
                    "Onset_index": [
                        index
                        for index, marker in zip(
                            epochs_markers[condition]["Onset_index"],
                            epochs_markers[condition]["Marker"],
                        )
                        if "4-Back" in marker
                    ],
                    "Offset_index": [
                        index
                        for index, marker in zip(
                            epochs_markers[condition]["Offset_index"],
                            epochs_markers[condition]["Marker"],
                        )
                        if "4-Back" in marker
                    ],
                }

                one_back_events = list(
                    np.arange(
                        one_back_markers["Onset_index"][0],
                        one_back_markers["Offset_index"][-1] - window * fs,
                        fs,
                    )
                )
                two_back_events = list(
                    np.arange(
                        two_back_markers["Onset_index"][0],
                        two_back_markers["Offset_index"][-1] - window * fs,
                        fs,
                    )
                )

                four_back_events = list(
                    np.arange(
                        four_back_markers["Onset_index"][0],
                        four_back_markers["Offset_index"][-1] - window * fs,
                        fs,
                    )
                )

                n_back_events = one_back_events + two_back_events + four_back_events

                arousal_values = (
                    len(one_back_events) * ["Low"]
                    + len(two_back_events) * ["Medium"]
                    + len(four_back_events) * ["High"]
                )

                epochs_signals = nk.epochs_create(
                    pd.DataFrame.from_dict(signals[condition]["Signals"]),
                    events=n_back_events,
                    sampling_rate=fs,
                    epochs_start=0,
                    epochs_end=window,
                )
            case "PreStroop" | "PostStroop":
                stroop_markers = [0]

                for timestamp_index in epochs_markers[condition]["Onset_index"]:
                    stroop_markers.append(len(np.where(timestamp_index >= events)[0]))

                count = np.diff(stroop_markers)
                markers = [
                    (
                        "Medium"
                        if "Congruente" in value
                        else "High" if "Incongruente" in value else value
                    )
                    for count, value in zip(count, epochs_markers[condition]["Marker"])
                    for _ in range(count)
                ]

                arousal_values = markers
                valence_values = markers

                epochs_signals = nk.epochs_create(
                    pd.DataFrame.from_dict(signals[condition]["Signals"]),
                    events=events,
                    sampling_rate=fs,
                    epochs_start=0,
                    epochs_end=window,
                )
            case "VR":
                markers = np.insert(
                    ratings[condition]["Arousal_index"],
                    0,
                    epochs_markers[condition]["Onset_index"][0],
                )

                valence_values = ratings[condition]["Valence"]
                arousal_values = ratings[condition]["Arousal"]

                valence_timestamps = np.insert(
                    ratings[condition]["Valence Timestamps"],
                    len(ratings[condition]["Valence Timestamps"]),
                    epochs_markers[condition]["Offset"][-1],
                )
                arousal_timestamps = np.insert(
                    ratings[condition]["Arousal Timestamps"],
                    0,
                    epochs_markers[condition]["Onset"][0],
                )

                events_diff = valence_timestamps - arousal_timestamps
                events_diff = list(events_diff[:-1])

                epochs_signals = nk.epochs_create(
                    pd.DataFrame.from_dict(signals[condition]["Signals"]),
                    events=markers[:-1],
                    sampling_rate=fs,
                    epochs_start=0,
                    epochs_end=events_diff,
                )

        epochs_data[condition] = {
            "Signals": epochs_signals,
            "Arousal": arousal_values,
            # "Valence": valence_values,
        }

    return epochs_data


@staticmethod
def getDataframe(dataframe, fs, resolution):
    """

    :param dataframe:
    :type dataframe:
    :param fs:
    :type fs:
    :param resolution:
    :type resolution:
    :return:
    :rtype:
    """
    HRV_Dataframe = Process_HRV(dataframe["ECG"], fs, resolution)
    RESP_Dataframe = Process_RESP(dataframe["RESP"], fs, resolution)
    EDA_Dataframe = Process_EDA(dataframe["EDA"], fs, resolution)
    Dataframe = (HRV_Dataframe.join(EDA_Dataframe)).join(RESP_Dataframe)

    return Dataframe


@staticmethod
def getFeatures(epochs_data: dict, fs: int, resolution: int) -> dict:
    """

    :param epochs_data:
    :type epochs_data:
    :param fs:
    :type fs:
    :param resolution:
    :type resolution:
    :return:
    :rtype:
    """
    features = {}

    for condition in tqdm(epochs_data):
        features[condition] = {}
        for epoch in tqdm(
            epochs_data[condition]["Signals"],
            desc=Fore.BLUE + f"Extracting features for {condition}" + Fore.BLUE,
            colour="cyan",
            bar_format=BLUE
            + "{l_bar}"
            + BLUE
            + "{bar}"
            + BLUE
            + "| "
            + BLUE
            + "{n_fmt}"
            + BLUE
            + "/"
            + BLUE
            + "{total_fmt} ["
            + BLUE
            + "{elapsed}"
            + "<"
            + BLUE
            + "{remaining}"
            + ", "
            + "{rate_fmt}"
            "]",
            leave=False,
        ):
            features[condition][epoch] = getDataframe(
                epochs_data[condition]["Signals"][epoch], fs, resolution
            )

    tqdm().set_description("All Conditions Processed")

    return features


@staticmethod
def getSignalsDataframe(features: dict, epochs_data: dict, df_baseline):
    """

    :param features:
    :type features:
    :param epochs_data:
    :type epochs_data:
    :return:
    :rtype:
    """
    BLUE = "\033[94m"

    dataframe = pd.DataFrame(columns=df_baseline.columns)
    users, conditions, arousal, valence = [], [], [], []

    for condition in tqdm(features):
        arousal.extend(epochs_data[condition]["Arousal"])
        # valence.extend(epochs_data[condition]["Valence"])

        for epoch in tqdm(
            features[condition],
            desc=Fore.BLUE
            + f"Concatenating Features in Dataframe for {condition}"
            + Fore.BLUE,
            colour="cyan",
            bar_format=BLUE
            + "{l_bar}"
            + BLUE
            + "{bar}"
            + BLUE
            + "| "
            + BLUE
            + "{n_fmt}"
            + BLUE
            + "/"
            + BLUE
            + "{total_fmt} ["
            + BLUE
            + "{elapsed}"
            + "<"
            + BLUE
            + "{remaining}"
            + ", "
            + "{rate_fmt}"
            "]",
            leave=False,
        ):
            conditions.append(condition + "_" + epoch)
            dataframe = pd.concat(
                [dataframe, features[condition][epoch]], ignore_index=True
            )

    dataframe["Arousal"] = arousal
    # dataframe["Valence"] = valence
    # dataframe["Condition"] = conditions
    dataframe = dataframe.rename(
        columns={
            "AVG": "AVG_RSP_Rate",
            "Maximum": "Max_RSP_Rate",
            "Minimum": "Min_RSP_Rate",
            "STD": "STD_RSP_Rate",
        }
    )

    return dataframe


@staticmethod
def getFullDataframe(
    dataframe: pd.DataFrame, df_baseline: pd.DataFrame, columns: list
) -> pd.DataFrame:
    """

    :param dataframe:
    :type dataframe:
    :param df_baseline:
    :type df_baseline:
    :param columns:
    :type columns:
    :return:
    :rtype:
    """
    full_dataframe = pd.DataFrame(columns=columns)

    for index, row in dataframe.iterrows():
        full_dataframe = pd.concat(
            [full_dataframe, row[columns] - df_baseline[columns]], ignore_index=True
        )
    full_dataframe["Arousal"] = dataframe["Arousal"]
    # full_dataframe["Valence"] = dataframe["Valence"]
    # full_dataframe["Condition"] = dataframe["Condition"]

    return full_dataframe


@staticmethod
def gridSearchCV(X: np.array, Y: np.array) -> dict:
    """

    :param X:
    :type X:
    :param Y:
    :type Y:
    :return:
    :rtype:
    """
    start_time = time.time()

    """Models Parameters for GridSearch"""

    param_grids = {
        "SVM": {
            "model": [svm.SVC(probability=True)],
            "model__C": [0.1, 1],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
            "model__random_state": list(range(0, 50, 5)),
        },
        "Bayes": {"model": [naive_bayes.GaussianNB()]},  # No parameters for GaussianNB
        "Logistic Regression": {
            "model": [LogisticRegression()],
            "model__C": [0.1, 1],
            "model__solver": ["lbfgs", "newton-cg", "sag", "saga"],
            "model__multi_class": ["multinomial"],
            "model__random_state": list(range(0, 50, 5)),
        },
        "Random Forest": {
            "model": [ensemble.RandomForestClassifier()],
            "model__n_estimators": [50, 100, 150],
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__random_state": list(range(0, 50, 5)),
        },
        "NN": {
            "model": [MLPClassifier()],
            "model__activation": ["identity", "logistic", "tanh", "relu"],
            "model__solver": ["lbfgs", "sgd", "adam"],
            "model__random_state": list(range(0, 50, 5)),
        },
    }

    """Train Model using GridSearchCV"""

    best_models = {}

    # Loop through each model and parameter grid
    for model_name, param_grid in tqdm(param_grids.items()):
        # Create the pipeline
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                ("scaler", StandardScaler()),
                (
                    "feature_selection",
                    feature_selection.RFE(
                        ensemble.RandomForestClassifier(random_state=42), step=1
                    ),
                ),
                (
                    "model",
                    param_grid["model"][0],
                ),  # Placeholder model to set pipeline structure
            ]
        )

        # Grid search for hyperparameter tuning
        if param_grid:
            with tqdm(
                total=len(param_grid),
                desc=f"Hyperparameter tuning for {model_name}",
                leave=False,
            ) as inner_bar:
                grid_search = GridSearchCV(
                    pipeline, param_grid, cv=5, n_jobs=1, verbose=2, scoring="accuracy"
                )
                grid_search.fit(X, Y.ravel())

                inner_bar.update(len(param_grid))

                # Store the best model and score
                best_models[model_name] = {
                    "best_estimator": grid_search.best_estimator_,
                    "best_score": grid_search.best_score_,
                }
                print(
                    f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_ * 100:.2f}%"
                )
        else:
            # Fit the pipeline without grid search if no parameters
            pipeline.fit(X, Y.ravel())
            best_models[model_name] = {
                "best_estimator": pipeline,
                "best_score": pipeline.score(
                    X, Y.ravel()
                ),  # Evaluate the score directly
            }
            print(
                f"No hyperparameters to tune for {model_name}. Model fitted directly."
            )

    print(f"Time to perform GridSearchCV = {time.time() - start_time} seconds")

    return best_models


# @staticmethod
# def getVideosDict(epochs: dict, videos: dict) -> dict:
#     dict = {}
#
#     for users in epochs.keys():
#         temp_dict = {}
#         for i, keys in enumerate(epochs[users].keys()):
#             temp_dict[videos[users][i]] = epochs[users][keys]
#         dict[users] = temp_dict
#
#     return dict

# @staticmethod
# def filterEEG(data: dict, EEG_fs: int) -> dict:
#     EEG_filtered = {}
#
#     for keys in data.keys():
#         temp_dict = {}
#         for i in range(1, 33):
#             temp_dict["EEG_" + str(i)] = EEG.filterData(
#                 EEG.ICA(data[keys]["EEG"]["EEG_" + str(i)]), EEG_fs
#             )
#         EEG_filtered[keys] = temp_dict
#
#     return EEG_filtered

# @staticmethod
# def getEEGChannels(EEG_filtered: dict) -> dict:
#     for users in EEG_filtered.keys():
#         EEG_filtered[users] = pd.DataFrame.from_dict(EEG_filtered[users])
#
#     return EEG_filtered

# @staticmethod
# def getEEGBands(EEG_dict: dict, EEG_fs: int) -> dict:
#     features_EEG = {}
#
#     for users in EEG_dict.keys():
#         temp_dict = {}
#         for key in EEG_dict[users].keys():
#             new_dict = {}
#             for i in range(1, 33):
#                 band_power = EEG.frequencyAnalysis(
#                     EEG_dict[users][key]["EEG_" + str(i)], EEG_fs
#                 )
#                 new_dict["EEG_" + str(i)] = band_power
#             temp_dict[key] = new_dict
#         features_EEG[users] = temp_dict
#
#     return features_EEG


# @staticmethod
# def getEEGDict(features_EEG: dict) -> tuple:
#     features_epochs_EEG, features_baseline_EEG = {}, {}
#
#     for user in features_EEG.keys():
#         temp_dict, channel_dict = {}, {}
#         for video in features_EEG[user].keys():
#             if video != "baseline":
#                 channel_dict = dict()
#                 for channel in features_EEG[user][video].keys():
#                     band_dict = {}
#                     for feature in features_EEG[user][video][channel].keys():
#                         band_dict[feature] = (
#                                 features_EEG[user][video][channel][feature]
#                                 - features_EEG[user]["baseline"][channel][feature]
#                         )
#                     channel_dict[channel] = band_dict
#                 temp_dict[video] = channel_dict
#             elif video == "baseline":
#                 for channel in features_EEG[user][video].keys():
#                     band_dict = {}
#                     for feature in features_EEG[user][video][channel].keys():
#                         band_dict[feature] = features_EEG[user]["baseline"][channel][
#                             feature
#                         ]
#                     channel_dict[channel] = band_dict
#
#         features_epochs_EEG[user] = temp_dict
#         features_baseline_EEG[user] = channel_dict
#
#     return features_epochs_EEG, features_baseline_EEG

# @staticmethod
# def getEpochs(data: dict, onset_index: dict, events_diff: dict, fs: int) -> dict:
#     epochs = {}
#
#     for users in data.keys():
#         epochs[users] = nk.epochs_create(
#             data[users],
#             events=onset_index[users],
#             sampling_rate=fs,
#             epochs_start=0,
#             epochs_end=events_diff[users],
#         )
#
#     return epochs
# @staticmethod
# def getFeaturesEpochs(data: dict) -> tuple:
#     features_epochs, baseline_dict = {}, {}
#     df_baseline = pd.DataFrame()
#     for user in data.keys():
#         temp_dict = {}
#         for video in data[user].keys():
#             df_test = pd.DataFrame()
#
#             if video != "baseline":
#                 for feature in data[user][video].columns:
#                     df_test[feature] = (
#                             data[user][video][feature] - data[user]["baseline"][feature]
#                     )
#                 temp_dict[video] = df_test
#             elif video == "baseline":
#                 for feature in data[user][video].columns:
#                     df_baseline[feature] = data[user]["baseline"][feature]
#
#         features_epochs[user] = temp_dict
#         baseline_dict[user] = df_baseline
#
#     return features_epochs, baseline_dict

# @staticmethod
# def getEEGDataframe(features_epochs_EEG: dict, features_baseline_EEG: dict):
#     columns = list()
#     category, video = [], []
#
#     for key in features_epochs_EEG.keys():
#         user = key
#     videos = list(features_epochs_EEG[user].keys())
#     for epochs in features_epochs_EEG[user][videos[0]].keys():
#         for bands in features_epochs_EEG[user][videos[0]][epochs].keys():
#             columns.append(epochs + "_" + bands)
#     df_EEG = pd.DataFrame(columns=columns)
#     baseline_df = pd.DataFrame(columns=columns)
#
#     for users in features_epochs_EEG.keys():
#         temp_df = pd.DataFrame(columns=columns)
#         baseline_list = list()
#
#         for channel in features_baseline_EEG[users].keys():
#             for band in features_baseline_EEG[users][channel].keys():
#                 baseline_list.append(features_baseline_EEG[users][channel][band])
#
#         baseline_df.loc[0] = baseline_list
#
#         for i, videos in enumerate(features_epochs_EEG[users].keys()):
#             if videos != "baseline":
#                 temp_list = list()
#                 for channel in features_epochs_EEG[users][videos].keys():
#                     for band in features_epochs_EEG[users][videos][channel].keys():
#                         temp_list.append(
#                             features_epochs_EEG[users][videos][channel][band]
#                         )
#                 temp_df.loc[i] = temp_list
#                 category.append(videos.split("/")[1])
#                 video.append(videos.split("/")[2])
#
#         df_EEG = pd.concat([df_EEG, temp_df], ignore_index=True)
#     df_EEG["Category"] = category
#     df_EEG["Video"] = video
#
#     return df_EEG, baseline_df
