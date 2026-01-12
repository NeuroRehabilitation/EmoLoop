import os,glob
import pandas as pd

def CSV_Load(path):
    data = {}
    os.chdir(path)
    for files in glob.glob("*.csv"):
        data[files.split(".")[0]] = pd.read_csv(files)

    return data