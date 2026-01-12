from loop_system.Sync import *
import warnings

warnings.filterwarnings("ignore")


class DataSender(multiprocessing.Process):
    def __init__(
        self,
        stream_name: str,
        stream_type: str,
        channel_count: int,
        sampling_rate,
        channel_format,
        source_id: str,
        data_queue,
        delta_time: int,
    ):
        super().__init__()
        self.stream_info = None
        self.stream_outlet = None
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.channel_count = channel_count
        self.sampling_rate = sampling_rate
        self.channel_format = channel_format
        self.source_id = source_id
        self.data_queue = data_queue
        self.delta_time = delta_time

    def run(self):
        print("Sender Started")
        self.stream_info = StreamInfo(
            self.stream_name,
            self.stream_type,
            self.channel_count,
            self.sampling_rate,
            self.channel_format,
            self.source_id,
        )
        info_channels = self.stream_info.desc().append_child("channels")
        info_channels.append_child("channel").append_child_value("label", "Stress")
        self.stream_outlet = StreamOutlet(self.stream_info)

        while True:
            if not self.data_queue.empty():
                data = self.data_queue.get()
                self.stream_outlet.push_chunk(data)
