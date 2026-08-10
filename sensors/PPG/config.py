from dataclasses import dataclass


@dataclass
class PPG_Config:
    lowpass_freq: float = 5
    highpass_freq: float = 0.1
    butter_order: int = 2
    filter_type: str = "bandpass"
    sampling_rate: float = 1000
    threshold: float = 0.7
    window: int = 5
