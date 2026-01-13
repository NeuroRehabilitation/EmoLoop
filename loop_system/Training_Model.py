import sys
import warnings

import joblib
from sklearn.ensemble import VotingClassifier

from loop_system.Baseline import select_file
from loop_system.Signals_Processing import *

warnings.filterwarnings("ignore")

def main():

    start_time = time.time()
    fs = 100
    resolution = 16
    sensors = ["ECG", "EDA", "RESP"]
    data = {}

    """Load Baseline Data file"""

    baseline_file = select_file("Select the Baseline data file", [("CSV files", "*.csv")])

    """Baseline Dataframe"""
    try:
        df_baseline = pd.read_csv(
            baseline_file,
            sep=";",
            index_col=False,
        )
        if "Unnamed: 0" in df_baseline.columns:
            df_baseline = df_baseline.drop("Unnamed: 0", axis=1)
        print("Baseline Dataframe loaded successfully.\n")
        print("This is the Baseline Dataframe:")
        print(df_baseline)
    except Exception as e:
        print(f"An error occurred loading the Baseline Dataframe: {e}")
        sys.exit(1)

    """Load N-Back Data file"""

    nback_file = select_file("Select the N-Back data folder", [("XDF files", "*.xdf")])

    if not nback_file:
        print("No file selected. Exiting...")
        sys.exit(1)

    try:
        condition = (os.path.basename(nback_file)).split("_")[1].split(".")[0]

        print(f"Loading N-Back data from {nback_file}...")

        data[condition] = Run_files(nback_file)

        print(f"N-Back data loaded successfully from {nback_file}.")

    except Exception as e:
        print(f"An error occurred loading the N-Back data: {e}")
        sys.exit(1)


    """Signals Processing"""
    signals = getSignals(
        data, "OpenSignals", "PsychoPy Markers", "PsychoPy Ratings", sensors=sensors
    )
    epochs_markers, ratings = getEvents(signals)

    """Create epochs from signals"""
    epochs_data = getSignalsEpochs(signals, epochs_markers, ratings, window=60, fs=fs)
    print("Epochs Created Successfully.")

    """Extract features from Epochs"""
    features = getFeatures(epochs_data, fs=fs, resolution=resolution)

    """Features Dataframe"""
    dataframe = getSignalsDataframe(features, epochs_data, df_baseline)
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

    try:
        path = os.path.dirname(os.path.dirname(nback_file))
        file_path = os.path.join(path, "dataframe.csv")
        dataframe.to_csv(file_path, sep=";")
        print(f"Features Dataframe saved to {path}.")
    except Exception as e:
        print(f"An error occurred saving the Features Dataframe: {e}")

    """Subtract baseline features to dataframe"""
    columns = dataframe.columns[: len(dataframe.columns) - 1]
    full_dataframe = getFullDataframe(dataframe, df_baseline, columns)

    try:
        path = os.path.dirname(os.path.dirname(nback_file))
        file_path = os.path.join(path, "full_dataframe.csv")
        full_dataframe.to_csv(file_path, sep=";")
        print(f"Full Dataframe saved to {path}.")
    except Exception as e:
        print(f"An error occurred saving the Full Dataframe: {e}")

    """Input Data for Models"""
    X = np.array(full_dataframe[columns])
    Y = np.array(full_dataframe[["Arousal"]])

    """GridSearchCV"""
    print("Performing GridSearchCV to find the best models...")
    best_models = gridSearchCV(X, Y)


    print("GridSearch completed.")
    # Sort the models by their best score in descending order
    sorted_models = sorted(
        best_models.items(), key=lambda item: item[1]["best_score"], reverse=True
    )

    # Select the top two models
    best_two_models = sorted_models[:2]

    print("Best two models selected for VotingClassifier:")
    for model_name, info in best_two_models:
        print(f"{model_name} with accuracy: {info['best_score']:.2f}")

    estimators = [
        (model_name, info["best_estimator"]) for model_name, info in best_two_models
    ]

    # Create a VotingClassifier with the best two models
    voting_clf = VotingClassifier(
        estimators=estimators, voting="soft"
    )  # You can also use 'hard' voting
    voting_clf.fit(X, Y.ravel())

    try:
        path = os.path.dirname(os.path.dirname(nback_file))
        joblib.dump(voting_clf, f"{path}/model.pkl")
        print(f"VotingClassifier saved successfully to {path}")
    except Exception as e:
        print(f"Error saving the model: {e}")

    print(f"Elapsed time = {(time.time()-start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()
