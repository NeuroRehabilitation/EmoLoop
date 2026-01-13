from loop_system.Stream import *


class ReceiveStreams(multiprocessing.Process):
    """ """

    def __init__(self):
        super().__init__()
        # List of names,information and inlet of the streams
        (
            self.stream_names,
            self.streams_info,
        ) = ([], [])

        # data_queue - Queue to put the data to send to Sync
        # info_queue - Queue to put the information of the streams to send to Sync
        self.data_queue = multiprocessing.Queue()
        self.info_queue = multiprocessing.Queue()

    def availableStreams(self) -> list:
        """
        :return: list with names of the available streams
        :rtype: list
        """

        streams = resolve_streams()

        for stream in streams:
            self.stream_names.append(stream.name())

        return self.stream_names

    def startProcess(self, name: str):
        """

        :param name: name of the stream
        :type name: str
        :return: stream and inlet_info
        :rtype: Stream, list
        """

        stream = Streams(name, self.data_queue)
        inlet = stream.getInlet()
        inlet_info = stream.getInletInfo(inlet)
        stream.start()
        print(f"Process Started {stream.name}")

        return inlet_info

    def stopChildProcesses(self):
        # get all active child processes
        active = multiprocessing.active_children()
        # terminate all active children
        for child in active:
            print("Process finished")
            child.terminate()
        for child in active:
            child.join()

    def run(self) -> None:
        # Get the names of the streams available
        self.stream_names = self.availableStreams()

        # For each stream available start the process of the stream and put the information in the info_queue
        for name in self.stream_names:
            inlet_info = self.startProcess(name)
            self.streams_info.append(inlet_info)
        self.info_queue.put(self.streams_info)
