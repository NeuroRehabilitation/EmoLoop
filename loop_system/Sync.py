import time
from collections import deque


from loop_system.ReceiveStreams import *


class Sync(multiprocessing.Process):
    def __init__(self, buffer_window: int) -> None:
        """
        :param buffer_window: time of the window to fill with buffers of data in seconds;
        :type buffer_window: int
        """

        super().__init__()

        # info_queue - Queue with the information of the streams
        # buffer_queue - Queue with each buffer of data to send to Manager.py
        # data_queue - Queue with the data of the streams from ReceiveStreams.py
        (
            self.data_queue,
            self.buffer_queue,
            self.arousal_queue,
            self.data_train_queue,
            self.info_queue,
        ) = (
            multiprocessing.Queue(),
            multiprocessing.Queue(),
            multiprocessing.Queue(),
            multiprocessing.Queue(),
            multiprocessing.Queue(),
        )

        # List with the stream info
        self.streams_info = []

        # Dictionaries to organize the data of all the streams
        self.synced_dict, self.data_to_train, self.information, self.timestamps = (
            {},
            {},
            {},
            {},
        )

        # Flag to check if buffers are synced, if it is the first buffer of data
        self.isSync, self.isFirstBuffer = False, True

        # Flag to start the acquisition of data by the Manager.
        self.startAcquisition = multiprocessing.Value("i", 0)
        self.sendBuffer = multiprocessing.Value("i", 0)
        self.clear_data = multiprocessing.Value("i", 0)

        # Locks of the system
        self.lock, self.train_lock = multiprocessing.Lock(), multiprocessing.Lock()

        self.data_available_event = multiprocessing.Event()

        # Number of full buffers from all the streams
        self.n_full_buffers = 0

        # Buffer window in seconds
        self.buffer_window = buffer_window

        # Timestamp to synchronize the streams
        self.first_timestamp = 0

    def startStreams(self):
        """
        Function to instantiate class ReceiveStreams and start the process ReceiveStreams;

        :return: returns object of class ReceiveStreams();

        """
        streams_receiver = ReceiveStreams()
        streams_receiver.start()

        return streams_receiver

    def getStreamsInfo(self, streams_receiver) -> None:
        """
        Function to get information about the streams and pass that information to a Queue

        :param streams_receiver: object of class ReceiveStreams()

        """

        self.streams_info = streams_receiver.info_queue.get()
        self.info_queue.put(self.streams_info)

    def createDict(self, stream_info: dict) -> dict:
        """
        Function to create dictionary for each stream with the channels names. columns_labels are the channels names.

        :param stream_info: dictionary with information about the stream
        :type stream_info: dict
        :return: dictionary with all the channels from the stream
        :rtype: dict
        """

        columns_labels = ["Timestamps"]
        for key in stream_info["Channels Info"].keys():
            columns_labels.append(stream_info["Channels Info"][key][0])
        dict = {key: deque() for key in columns_labels}

        return dict

    def clearDict(self, stream_name: str):
        # print("Clearing Data.")
        for key in self.data_to_train[stream_name].keys():
            self.data_to_train[stream_name][key].clear()

    def fill_TrainingData(self, data: tuple, stream_name: str) -> None:

        self.data_to_train[stream_name]["Timestamps"].append(data[1])
        for i, key in enumerate(self.data_to_train[stream_name].keys()):
            if key != "Timestamps":
                self.data_to_train[stream_name][key].append(data[0][i - 1])

    def fillData(self, data: tuple, stream_name: str) -> None:
        """Function to fill the dictionary synced_dict with the data from each stream

        :param data: data from the streams
        :type data: tuple -[(nseq,value),timestamp]
        :param stream_name: name of the stream
        :type stream_name: str
        """
        self.synced_dict[stream_name]["Timestamps"].append(data[1])

        for i, key in enumerate(self.synced_dict[stream_name].keys()):
            if key != "Timestamps":
                self.synced_dict[stream_name][key].append(data[0][i - 1])

    def getBufferMaxSize(self, stream_name: str) -> int:
        """Set the maximum size of the buffer to fill with data

        :param stream_name: name of the stream
        :type stream_name: str
        :return: max buffer size - window*fs
        :rtype: int
        """

        return int(self.buffer_window * self.information[stream_name]["Sampling Rate"])

    def checkBufferSize(self, max_size: int, stream_name: str) -> None:
        """Check if buffers are synchronized and then check the size of the buffers for every stream

        :param max_size: maximum size of the buffer
        :type max_size:int
        :param stream_name: name of the stream
        :type stream_name: str
        """
        buffer_len = [len(value) for value in self.synced_dict[stream_name].values()]
        if all(i == max_size for i in buffer_len):
            if not self.information[stream_name]["Full Buffer"]:
                self.n_full_buffers += 1
            else:
                if self.n_full_buffers == len(self.synced_dict.keys()):
                    self.isFirstBuffer = False
            self.information[stream_name]["Full Buffer"] = True

    def slidingWindow(self, stream_name: str) -> None:
        """Create Sliding window to remove the oldest element of the array (index 0) and add the newest sample value

        :param stream_name: name of the stream
        :type stream_name: str
        """
        for key in self.synced_dict[stream_name].keys():
            self.synced_dict[stream_name][key].popleft()

    def syncStreams(self, data: tuple, stream_name: str) -> None:
        """Synchronize streams:
            check if the timestamps from all the streams are higher than the minimum timestamp.
            Streams are synchronized if condition is true.

        :param first_timestamp: minimum timestamp of all the streams
        :type first_timestamp: int
        """
        if not self.isSync:
            self.timestamps[stream_name] = data[1]
            if len(self.timestamps.values()) >= 1:
                if self.first_timestamp == 0:
                    self.first_timestamp = max(self.timestamps.values())
                else:
                    if all(i >= self.first_timestamp for i in self.timestamps.values()):
                        self.isSync = True
                        print("\nStreams are Synced.")

    def getBuffers(self, data: tuple, stream_name: str) -> None:
        """

        :param data: data from the streams
        :type data: tuple
        :param stream_name: name of the streams
        :type stream_name: str

        """
        # In case of the data being synchronized
        if self.isSync:
            # print("Get Buffer")
            if "PsychoPy" not in stream_name:
                # If buffers are not full keep checking the size of the buffers
                self.checkBufferSize(
                    self.information[stream_name]["Max Size"],
                    stream_name,
                )

                # In case it is the first buffer of data
                if not self.information[stream_name]["Full Buffer"]:
                    # Fills the first buffer of data with data
                    self.fillData(data, stream_name)
                    self.fill_TrainingData(data, stream_name)
                else:
                    # If it is not the first buffer (buffers are with max size)
                    # Start sliding window and put buffers on the Queue to send to process
                    self.slidingWindow(stream_name)
                    self.fillData(data, stream_name)
                    self.fill_TrainingData(data, stream_name)

    def getPsychoPyData(self, data: tuple, stream_name: str) -> None:
        """
        :param data:
        :type data:
        :param stream_name:
        :type stream_name:
        """
        if "Ratings" in stream_name:
            if "Arousal" in data[0][0]:
                self.arousal_queue.put(data[0][1])
                # print(f"Arousal = {data[0][1]}")

    def run(self):
        """ """
        # Start all the available streams
        streams_receiver = self.startStreams()

        # Get the information from the streams
        self.getStreamsInfo(streams_receiver)

        # For every streams available create and fill the dictionary synced_dict with:
        # Name of the stream, Maximum Size of the buffers, Number of channels filled with data
        for stream in self.streams_info:
            if "PsychoPy" not in stream["Name"]:
                self.synced_dict[stream["Name"]] = self.createDict(stream)
                self.data_to_train[stream["Name"]] = self.createDict(stream)
                self.information[stream["Name"]] = stream
                self.information[stream["Name"]]["Max Size"] = self.getBufferMaxSize(
                    stream["Name"]
                )
                self.information[stream["Name"]]["Full Buffer"] = False

        start_time = time.time()
        # Loop to receive the data - start acquisition is true
        while bool(self.startAcquisition.value):
            if self.isFirstBuffer:
                elapsed = time.time() - start_time
                print(f"Elapsed Time = {elapsed:.2f} seconds.")
            # Synchronize the data
            if not self.isSync:
                if streams_receiver.data_queue.qsize() > 0:
                    stream_name, data = streams_receiver.data_queue.get()
                    if "PsychoPy" not in stream_name:
                        self.syncStreams(data, stream_name)
                    else:
                        self.getPsychoPyData(data, stream_name)
            if self.isSync:
                if streams_receiver.data_queue.qsize() > 0:
                    stream_name, data = streams_receiver.data_queue.get()

                    if "Markers" in stream_name:
                        if data[0][0] == "end":
                            self.startAcquisition.value = 0
                    if "Ratings" in stream_name:
                        self.getPsychoPyData(data, stream_name)
                        with self.train_lock:
                            buffer_len = [
                                len(value)
                                for value in self.data_to_train["OpenSignals"].values()
                            ]
                            cropped_data = {"OpenSignals": {}}
                            if all(
                                i > self.information["OpenSignals"]["Max Size"]
                                for i in buffer_len
                            ):
                                for key in self.data_to_train["OpenSignals"].keys():
                                    cropped_data["OpenSignals"][key] = deque(
                                        self.data_to_train["OpenSignals"][key],
                                        maxlen=self.information["OpenSignals"][
                                            "Max Size"
                                        ],
                                    )
                                self.data_train_queue.put(cropped_data)
                            else:
                                self.data_train_queue.put(self.data_to_train)
                            print("Putting Training data in Manager Queue.")
                            self.data_available_event.set()
                    else:
                        self.getBuffers(data, stream_name)
                        if bool(self.clear_data.value):
                            self.clearDict(stream_name)
                            self.clear_data.value = 0
                            # print(self.data_to_train)

            if not self.isFirstBuffer and self.sendBuffer.value == 1:
                with self.lock:
                    self.buffer_queue.put(self.synced_dict)
                    # print(f"Queue size = {self.buffer_queue.qsize()}")

        # Stop all running child processes
        streams_receiver.stopChildProcesses()
        streams_receiver.terminate()
        streams_receiver.join()
