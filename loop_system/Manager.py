import os.path
import sys

from loop_system.Baseline import select_file
from loop_system.ModelTrainer import *
from loop_system.Process import *

warnings.filterwarnings("ignore")

import builtins

original_print = builtins.print
counter = 0


def custom_print(*args, **kwargs):
    global counter
    counter += 1
    original_print(f"[{counter}]", *args, **kwargs)


builtins.print = custom_print


class TeeLogger:
    """A logger that writes output to both console and a file."""

    def __init__(self, file):
        self.terminal = sys.stdout  # Save original stdout
        self.log = file  # Log file

    def write(self, message):
        self.terminal.write(message)  # Print to console
        self.log.write(message)  # Write to file

    def flush(self):  # Needed for compatibility
        self.terminal.flush()
        self.log.flush()


class Manager(multiprocessing.Process):
    def __init__(self):
        self.outlet_stream = None
        self.data_to_stream = None
        self.data_queue = multiprocessing.Queue()
        self.training_df = None
        self.baseline = None
        self.model = None
        self.model_version = 1
        self.model_lock = multiprocessing.Lock()

    def run(self):
        """ """

        # Instantiate object from class Sync and Processing
        sync = Sync(buffer_window=60)
        process = Processing()
        modelTrainer = ModelTrainer()

        """Load and Select Baseline Data file"""
        baseline_file = select_file("Select the Baseline data file.", [("CSV files", "*.csv")])

        if not baseline_file:
            print("No baseline file selected. Exiting...")
            sys.exit(1)

        try:
            self.baseline = pd.read_csv(baseline_file, sep=";", index_col=False)
            if "Unnamed: 0" in self.baseline.columns:
                self.baseline = self.baseline.drop("Unnamed: 0", axis=1)
            print("Baseline Dataframe loaded successfully.\n")
            print(f"This is the Baseline Dataframe: \n {self.baseline}")
        except Exception as e:
            print(f"An error occurred loading the Baseline Dataframe: {e}")
            sys.exit(1)

        """Load and Select Training Data file"""

        training_data = select_file("Select the training data file.", [("CSV files", "*.csv")])

        if not training_data:
            print("No training data file selected. Exiting...")
            sys.exit(1)

        try:
            training_df = pd.read_csv(training_data, sep=";", index_col=False)
            if "Unnamed: 0" in training_df.columns:
                training_df = training_df.drop("Unnamed: 0", axis=1)
            print("Training Dataframe loaded successfully.\n")
            print(f"This is the Training Dataframe: \n {training_df}")
        except Exception as e:
            print(f"An error occurred loading the Training Dataframe: {e}")
            sys.exit(1)

        """Load and Select Model file"""
        model_file = select_file("Select the model file.", [("Pickle files", "*.pkl")])

        if not model_file:
            print("No model file selected. Exiting...")
            sys.exit(1)

        try:
            self.model = joblib.load(model_file)
            print(f"Model loaded successfully from {model_file}.")
        except Exception as e:
            print(f"Error loading model: {e}.")
            sys.exit(1)

        input("Press Enter to start acquisition...")

        base_dir = os.path.dirname(model_file)

        log_folder = os.path.join(base_dir, "Console_logs")
        os.makedirs(log_folder, exist_ok=True)  # Ensure directory exists
        log_file_path = os.path.join(log_folder, "output.txt")

        print("Logging output to {log_file_path}...".format(log_file_path=log_file_path))

        log_file = open(log_file_path, "w")
        sys.stdout = TeeLogger(log_file)
        sys.stderr = TeeLogger(log_file)

        # Start process Sync and put flag startAcquisition as True
        sync.start()
        modelTrainer.start()
        sync.startAcquisition.value = 1
        modelTrainer.startAcquisition.value = 1
        sync.sendBuffer.value = 1

        print("Acquisition Started!")

        i = 0
        previous_df = None

        # Get streams information
        process.info = sync.info_queue.get()

        try:
            data_sender = DataSender(
                stream_name="loopsystem",
                stream_type="stress",
                channel_count=2,
                sampling_rate=IRREGULAR_RATE,
                channel_format=cf_string,
                source_id="id1",
                data_queue=self.data_queue,
                delta_time=1,
            )
            data_sender.start()
            with modelTrainer.lock:
                print("Sending Initial Model and Training Dataframe to Model Trainer.")
                modelTrainer.model_queue.put((self.model, self.training_df))

            print("Acquisition Running...")
            print("Press Ctrl+C to stop.")

            # While it is acquiring data
            while bool(sync.startAcquisition.value):
                isDataAvailable = sync.data_available_event.wait(timeout=0.1)
                isModelRetrained = modelTrainer.model_retrained_event.wait(timeout=0.1)

                if isDataAvailable:
                    with sync.train_lock:
                        data_to_train = sync.data_train_queue.get()
                        # print(
                        #     f'Len =  {len(data_to_train["OpenSignals"]["Timestamps"])}'
                        # )

                        # print(data_to_train)
                        print("Getting Training Data from Sync Queue.")

                        new_sample = process.getOpenSignals(data_to_train, process.info)
                        new_sample -= self.baseline

                        X = np.array(new_sample)
                        predicted_label = self.model.predict(X)[0]
                        print(f"Predicted Label = {predicted_label}")

                        arousal = int(sync.arousal_queue.get())

                        if arousal <= 3:
                            true_label = "Low"
                        elif arousal >= 7:
                            true_label = "High"
                        else:
                            true_label = "Medium"

                        new_sample["Arousal"] = true_label
                        print(new_sample)

                        sync.data_available_event.clear()
                        sync.clear_data.value = 1
                        if true_label != predicted_label:
                            with modelTrainer.lock:
                                modelTrainer.new_sample_queue.put(new_sample)
                                modelTrainer.sample_available_event.set()
                                print("Sending new data to Model Trainer Queue.")
                        else:
                            print("No need to retrain model!")
                            continue

                if isModelRetrained:
                    with self.model_lock:
                        if not modelTrainer.model_queue.empty():
                            self.model = modelTrainer.model_queue.get()
                            self.model_version += 1
                            # print(self.model)
                            print("Updating retrained model.")
                            modelTrainer.model_retrained_event.clear()

                # If there is data in the buffer queue from Sync, send to Process.
                if sync.buffer_queue.qsize() > 0:
                    sync.sendBuffer.value = 0
                    with sync.lock:
                        process.data = sync.buffer_queue.get()
                        i += 1
                        features = process.processData()
                        process.features = features - self.baseline
                        if previous_df is not None and features is not None:
                            if not np.allclose(
                                features.values, previous_df.values, atol=1e-3
                            ):
                                with self.model_lock:
                                    predicted_sample, probability = process.predict(
                                        self.model
                                    )
                                    self.data_queue.put(
                                        [predicted_sample[0], str(probability)]
                                    )
                                    print(
                                        f"Prediction = {predicted_sample[0]}, Probability = {probability}, Model v{self.model_version}."
                                    )

                        sync.sendBuffer.value = 1
                        previous_df = features

        except Exception as e:
            print(f"An error occurred: {e}")
            pass
        finally:
            try:
                model_path = os.path.join(base_dir, f"model_v{self.model_version}.pkl" )
                joblib.dump(
                    self.model,
                    model_path
                )
                print(f"Model saved successfully to {model_path}")
            except Exception as e:
                print(f"Error saving the model: {e}")
                pass

            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_file.close()
            print("Acquisition Stopped.")
            print(f"Log saved in: {log_file_path}")
            sync.terminate()
            sync.join()
            modelTrainer.terminate()
            modelTrainer.join()
            data_sender.terminate()
            data_sender.join()
            print("Closed all processes.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    manager = Manager()
    process = multiprocessing.Process(target=manager.run())
    process.start()
    process.join()
