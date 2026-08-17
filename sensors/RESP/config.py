from dataclasses import dataclass


@dataclass
class RESP_Config:
    sampling_rate: int = 1000  # Sampling frequency in Hz

    # ADC / hardware conversion parameters
    VCC: float = 3  # Reference voltage for analog-to-digital conversion (Volts)
    resolution: float = 16  # ADC resolution in bits
    gain: float = 1
    method: str = "biosppy"