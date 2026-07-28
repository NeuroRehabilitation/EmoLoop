from dataclasses import dataclass


@dataclass
class HRV_Config:
    ectopy_threshold: float = 0.2  # Threshold for ectopic beat detection
    window: str = "hann"
    interpolation_rate: int = 4
