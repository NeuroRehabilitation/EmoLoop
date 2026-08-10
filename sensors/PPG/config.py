from dataclasses import dataclass


@dataclass
class PPG_Config:
    lowpass_freq: float = 5
    highpass_freq: float = 0.1
    lowpass_Order: int = 2
    highpass_Order: int = 2
    filter_type: str = "bandpass"
    sampling_rate: float = 1000
