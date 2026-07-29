from dataclasses import dataclass


@dataclass
class HRV_Config:
    ectopy_threshold: float = 0.2  # Threshold for ectopic beat detection
    window: str = "hann"
    interpolation_rate: int = 4
    vlf_lfreq: float = 0.0033  # Very Low Frequency lower bound
    vlf_hfreq: float = 0.04  # Very Low Frequency upper bound
    lf_lfreq: float = 0.04  # Low Frequency lower bound
    lf_hfreq: float = 0.15  # Low Frequency upper bound
    hf_lfreq: float = 0.15  # High Frequency lower bound
    hf_hfreq: float = 0.4  # High Frequency upper bound
