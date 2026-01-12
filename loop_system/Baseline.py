import sys
import warnings
from loop_system.Signals_Processing import *
import tkinter as tk
from tkinter import filedialog

warnings.filterwarnings("ignore")

def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()  # hide the main Tkinter window

    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )

    if not file_path:
        print("No file selected. Exiting...")
        sys.exit(1)

    return file_path

def main():
    start_time = time.time()

    # Prompt user to select the XDF file
    file_path = select_file("Select the Baseline data file",[("XDF files", "*.xdf")])

    # Extract folder and condition name from the file path
    folder = os.path.dirname(file_path)
    condition = os.path.basename(file_path).split("_")[1].split(".")[0]

    print(f"Loading {file_path}...")

    data = {}
    fs = 100
    resolution = 16
    sensors = ["ECG", "EDA", "RESP"]

    try:
        data[condition] = Run_files(file_path)
        print("File loaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    """Signals Processing"""
    signals = getSignals(
        data, "OpenSignals", "PsychoPy Markers", "PsychoPy Ratings", sensors=sensors
    )
    epochs_markers, ratings = getEvents(signals)

    """Baseline Dataframe"""
    baseline_signals = nk.epochs_create(
        pd.DataFrame.from_dict(signals["baseline"]["Signals"]),
        events=epochs_markers["baseline"]["Onset_index"],
        sampling_rate=fs,
        epochs_start=0,
        epochs_end=epochs_markers["baseline"]["EventsDiff"],
    )
    df_baseline = getDataframe(baseline_signals["1"], fs, resolution)

    try:
        save_folder = os.path.dirname(file_path)
        save_path = os.path.join(save_folder, "df_baseline.csv")
        df_baseline.to_csv(save_path, sep=";")
        print(f"Time elapsed = {(time.time()-start_time):.2f} seconds.")
        print(f"Baseline Dataframe saved to {save_path}.")
    except Exception as e:
        print(f"An error occurred saving the Baseline Dataframe: {e}")

if __name__ == "__main__":
    main()
