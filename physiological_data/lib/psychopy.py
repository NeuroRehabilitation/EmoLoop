import numpy as np
from pandas import read_csv


class Psycho:
    def __init__(self, file_path, task=None):
        self.file_path = file_path
        if task is "N-Back":
            self.data = self.getNBack()
        elif task is "Subtraction":
            self.data = self.getSubtraction()
        elif task is "Baseline":
            self.data = self.getBaseline()
        else:
            self.data = self.getPsychoData()

    def getPsychoData(self):
        return read_csv(self.file_path)

    def getNBack(self):
        data = self.getPsychoData()
        indexes = np.where(np.array(data["Activity"]) == "N-back")[0]
        return data.loc[np.append(indexes, indexes[-1] + 1)]

    def getSubtraction(self):
        data = self.getPsychoData()
        indexes = np.sort(
            np.concatenate(
                [
                    np.where(np.array(data["Activity"]) == "o")[0],
                    np.where(np.array(data["Activity"]) == "sub_baseline")[0],
                ]
            )
        )
        return data.loc[np.append(indexes, indexes[-1] + 1)]

    def getBaseline(self):
        data = self.getPsychoData()
        indexes = np.sort(
            np.concatenate(
                [
                    np.where(np.array(data["Activity"]) == "Begin of Baseline")[0],
                    np.where(np.array(data["Activity"]) == "End of Baseline")[0],
                ]
            )
        )
        return data.loc[np.append(indexes, indexes[-1])]

    def getTimestamps(self):
        return np.array(self.data["Timestamp"])

    def getActivity(self):
        return np.array(self.data["Activity"])

    def getResult(self):
        return np.array(self.data["Result"])

    def getDifficulty(self):
        return np.array(self.data["Difficulty"])

    def getTimeAnswer(self):
        return np.array(self.data["Time to Answer"])

    def getKeyAnswer(self):
        return np.array(self.data["Key Answer"])


if __name__ == "__main__":
    psycho = Psycho(r"..\..\acquisitions\Acquisitions\03_11_2020\results_3.csv")
    print(psycho.getNBack())
    print(psycho.getSubtraction())
    print(psycho.getBaseline())
