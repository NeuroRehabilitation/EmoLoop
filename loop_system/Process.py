import os.path

import joblib

from loop_system.Signals_Processing import *


class Processing:
    def __init__(self):
        self.features = None
        self.info = []
        self.data = {}

    @staticmethod
    def getOpenSignals(data: dict, info: list):
        """

        :return:
        :rtype:
        """

        dataframe, HRV_Dataframe, EDA_Dataframe, RESP_Dataframe = None, None, None, None

        for stream in info:
            if stream["Name"] == "OpenSignals":
                name = stream["Name"]
                fs = stream["Sampling Rate"]
                for key in data[name].keys():
                    if key.startswith("ECG"):
                        HRV_Dataframe = Process_HRV(data[name][key], fs, 16)
                    if key.startswith("EDA"):
                        EDA_Dataframe = Process_EDA(data[name][key], fs, 16)
                    if key.startswith("RESP"):
                        RESP_Dataframe = Process_RESP(data[name][key], fs, 16)

                dataframe = (HRV_Dataframe.join(EDA_Dataframe)).join(RESP_Dataframe)

        return dataframe

    def getOpenvibe(self):
        """

        :return:
        :rtype:
        """
        for stream in self.info:
            if stream["Name"] == "openvibeSignal":
                fs = stream["Sampling Rate"]
                name = stream["Name"]
                EEG_dict = {}
                for i, key in enumerate(self.data[name]):
                    if i > 0 and i < 33:
                        channel = "EEG_" + str(i)
                        EEG_dict[channel] = EEG.frequencyAnalysis(
                            EEG.filterData(EEG.ICA(self.data[name][key]), fs),
                            fs,
                        )

                columns, temp_list = [], []
                for channel in EEG_dict.keys():
                    for band in EEG_dict[channel].keys():
                        columns.append(channel + "_" + band)
                        temp_list.append(EEG_dict[channel][band])
                EEG_Dataframe = pd.DataFrame(columns=columns)
                EEG_Dataframe.loc[0] = temp_list

        return EEG_Dataframe

    def processData(self):
        """

        :return:
        :rtype:
        """

        self.features = pd.DataFrame()
        dataframe = None
        if "OpenSignals" in self.data.keys():
            dataframe = self.getOpenSignals(self.data, self.info)
        # if "openvibeSignal" in self.data.keys():
        #     EEG_dataframe = self.getOpenvibe()

        # self.features = pd.concat([EEG_dataframe, dataframe], axis=1)
        self.features = dataframe

        return self.features

    @staticmethod
    def loadModel(path: str):

        model_path = os.path.join(path, "model.pkl")
        model = None

        if not os.path.exists(path):
            raise FileNotFoundError(f"The directory {path} does not exist.")
            sys.exit(1)
        elif not os.path.isfile(model_path):
            raise FileNotFoundError(f"The file {model_path} does not exist.")
            sys.exit(1)
        else:
            try:
                model = joblib.load(model_path)
                print("Model loaded successfully.\n")
                print(model)
            except Exception as e:
                print(e)
                print("Error on loading Model file!")

        return model

    def predict(self, model):
        try:
            X = np.array(self.features)
            if len(X) > 0:
                prediction = model.predict(X)
                index = list(model.classes_).index(prediction[0])
                probability = model.predict_proba(X)[0][index]

                return prediction, probability
            else:
                print("X has no data!")
        except Exception as e:
            print(e)
            print("No prediction")
