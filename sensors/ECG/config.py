from dataclasses import dataclass


@dataclass
class ECG_Config:
    VCC: int = 3000
    gain: int = 1000
    resolution: int = 16
    filter_type: str = "bandpass"
    butter_order: int = 4
    lowpass_freq: int = 5
    highpass_freq: int = 15
    mph: int = None
    mpd: int = 35
    threshold: int = 0
    edge: str = "rising"
    kpsh: bool = False
    valley: bool = False
    library: str = "neurokit"
    discard_window: float = 0.15
    sampling_rate: int = 1000
