from dataclasses import dataclass


@dataclass
class ECG_Config:
    VCC: int = 3000
    gain: int = 1000
    resolution: int = 16
    filter_type: str = "bandpass"
    butter_order: int = 5
    lowpass_freq: int = 5
    highpass_freq: int = 15
    minimum_peak_height: int = None
    minimum_peak_distance: int = 35
    edge: str = 'rising'
    library: str = "neurokit"
    sampling_rate: int = 100
    