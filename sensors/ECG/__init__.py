"""
ECG Signal Processing Module for EmoLoop.

Provides high-level ECG processing with algorithm-independent interface.
"""

# Expose main classes for easy importing
from sensors.ECG.processor import ECG
from sensors.ECG.base import ECG_base
from sensors.ECG.config import ECG_Config
from sensors.ECG.algorithms import PanTompkinsAlgorithm
from sensors.ECG.lib_selector import ECGAlgorithmSelector

# # Package version (optional)
# __version__ = "1.0.0"