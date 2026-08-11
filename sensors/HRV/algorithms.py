
"""
sensors.HRV.algorithms

HRVAlgorithm - a reference implementation of the HRV_base interface.

This module provides a concrete HRVAlgorithm that implements common HRV
processing routines used by the higher-level `sensors.HRV.processor.HRV`
wrapper. It expects RR intervals (in seconds) and corresponding RR timestamps
(in seconds) where needed.

Implemented features:
 - Ectopy removal heuristic
 - Basic RR / heart-rate summaries
 - Time-domain HRV features (SDNN, RMSSD, NN50/pNN50, NN20/pNN20, etc.)
 - Frequency-domain analysis using interpolation + Welch PSD
 - Frequency-domain summary features (VLF/LF/HF band powers, normalized units)
 - Non-linear features (SDSD, SD1, SD2, SD2/SD1)

Notes:
 - RR interval inputs are expected in seconds (float). Many returned "RR"
   values are converted to milliseconds (multiplied by 1000) to match common
   HRV reporting conventions.
 - Frequency analysis uses SciPy for spline interpolation and Welch PSD.
 - Several functions return np.nan when input size is insufficient for the
   computation (e.g., less than 2 intervals for RMSSD).
 - The algorithm relies on configuration values provided by `HRV_Config`,
   such as `interpolation_rate`, frequency band limits, ectopy threshold, and
   PSD parameters.
"""

from typing import Dict, Any

from sensors.HRV.base import HRV_base
from sensors.HRV.config import HRV_Config

import numpy as np
import scipy as sc


class HRVAlgorithm(HRV_base):
    """
    Reference HRV algorithm implementation.

    Parameters
    ----------
    config : HRV_Config
        Configuration object containing algorithm parameters:
          - ectopy_threshold: relative difference threshold used in ectopy removal
          - interpolation_rate: sampling rate (Hz) used to create uniformly sampled RR series
          - window: window name/length for Welch PSD
          - vlf_lfreq, vlf_hfreq, lf_lfreq, lf_hfreq, hf_lfreq, hf_hfreq: band limits (Hz)
          - ... (other config attributes used by frequencyAnalysis / PSD)
    """

    def __init__(self, config: HRV_Config):
        self.config = config

    def remove_ectopy_beats(self, rr_intervals: np.ndarray) -> np.ndarray:
        """
        Heuristic ectopy removal.

        Marks an RR interval as ectopic if its relative difference to the previous
        interval exceeds `config.ectopy_threshold`. When an ectopic beat is found,
        this implementation also drops the subsequent interval (common heuristic
        to remove the pair around an ectopic event).

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals in seconds.

        Returns
        -------
        np.ndarray
            Filtered RR intervals (subset of input) with ectopic candidates removed.

        Notes
        -----
        - Very small arrays (size < 2) are returned unchanged.
        - The algorithm uses a simple local relative-difference rule; more
          sophisticated methods (adaptive filters, template matching) can be
          substituted by providing a different algorithm implementation.
        """
        if rr_intervals.size < 2:
            return rr_intervals.copy()

        keep = np.ones(len(rr_intervals), dtype=bool)

        for i in range(1, len(rr_intervals)):
            prev = rr_intervals[i - 1]
            # Skip non-positive previous intervals (invalid)
            if prev <= 0:
                continue
            # Relative difference check
            if abs(rr_intervals[i] - prev) / prev > self.config.ectopy_threshold:
                keep[i] = False
                # Also drop the following interval (pair removal heuristic)
                if i + 1 < len(rr_intervals):
                    keep[i + 1] = False

        return rr_intervals[keep]

    def rr_intervals(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Basic RR interval summary statistics.

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in seconds.

        Returns
        -------
        dict
            Dictionary with average, min, max and standard deviation of RR (in seconds).
            Returns np.nan for all values if input is empty.
        """
        if rr_intervals.size == 0:
            return {
                "Avg RR": np.nan,
                "Min RR": np.nan,
                "Max RR": np.nan,
                "SD RR": np.nan,
            }

        return {
            "Avg RR": float(np.nanmean(rr_intervals)),
            "Min RR": float(np.nanmin(rr_intervals)),
            "Max RR": float(np.nanmax(rr_intervals)),
            "SD RR": float(np.nanstd(rr_intervals)),
        }

    def heart_rate(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Compute basic heart-rate summaries derived from RR intervals.

        Converts RR intervals (s) to instantaneous heart rate (beats per minute)
        as 60 / RR, then computes mean/min/max/std of that series.

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in seconds.

        Returns
        -------
        dict
            Dictionary with Avg/Min/Max/SD heart rate (in beats per minute).
            Returns np.nan for empty input.
        """
        if rr_intervals.size == 0:
            return {
                "Avg HR": np.nan,
                "Min HR": np.nan,
                "Max HR": np.nan,
                "SD HR": np.nan,
            }

        heart_rate = 60 / rr_intervals

        return {
            "Avg HR": float(np.nanmean(heart_rate)),
            "Min HR": float(np.nanmin(heart_rate)),
            "Max HR": float(np.nanmax(heart_rate)),
            "SD HR": float(np.nanstd(heart_rate)),
        }

    def time_domain_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """
        Compute time-domain HRV metrics.

        Typical outputs include:
         - Avg/Min/Max/SD RR (reported in milliseconds)
         - SDNN (ms): standard deviation of NN intervals
         - RMSSD (ms): root mean square of successive differences
         - NN50 / pNN50 : counts and percentage of successive diffs > 50 ms
         - NN20 / pNN20 : counts and percentage of successive diffs > 20 ms

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in seconds.

        Returns
        -------
        dict
            Dictionary with the time-domain metrics. For empty input, returns
            NaNs for all metrics.
        """
        if rr_intervals.size == 0:
            return {
                "Avg RR": np.nan,
                "Min RR": np.nan,
                "Max RR": np.nan,
                "SD RR": np.nan,
                "SDNN": np.nan,
                "RMSSD": np.nan,
                "NN50": np.nan,
                "pNN50": np.nan,
                "NN20": np.nan,
                "pNN20": np.nan,
            }

        nn50 = self.NN50(rr_intervals)
        nn20 = self.NN20(rr_intervals)

        return {
            # Convert RR statistics to milliseconds for readability / convention
            "Avg RR": float(np.nanmean(rr_intervals) * 1000),
            "Min RR": float(np.nanmin(rr_intervals) * 1000),
            "Max RR": float(np.nanmax(rr_intervals) * 1000),
            "SD RR": float(np.nanstd(rr_intervals) * 1000),
            "SDNN": self.SDNN(rr_intervals),
            "RMSSD": self.RMSSD(rr_intervals),
            "NN50": nn50,
            "pNN50": self.pNN50(nn50, rr_intervals),
            "NN20": nn20,
            "pNN20": self.pNN20(nn20, rr_intervals),
        }

    @staticmethod
    def SDNN(rr_intervals: np.ndarray) -> float:
        """
        Standard deviation of NN intervals (SDNN).

        Returns value in milliseconds, rounded to 4 decimal places.

        Returns np.nan if input is empty.
        """
        return (
            float(round(np.std(rr_intervals) * 1000, 4))
            if rr_intervals.size > 0
            else np.nan
        )

    @staticmethod
    def RMSSD(rr_intervals: np.ndarray) -> float:
        """
        Root mean square of successive differences (RMSSD).

        RMSSD is computed as sqrt( sum(diff^2) / (N-1) ) and returned in ms.
        Returns np.nan if fewer than 2 intervals are available.
        """
        return (
            float(
                round(
                    np.sqrt(
                        np.sum((np.diff(rr_intervals)) ** 2) / (len(rr_intervals) - 1)
                    )
                    * 1000,
                    4,
                )
            )
            if not rr_intervals.size < 2
            else np.nan
        )

    @staticmethod
    def NN50(rr_intervals: np.ndarray) -> int:
        """
        Count of successive RR interval differences greater than 50 ms.

        Returns np.nan if fewer than 2 intervals exist (to be consistent with other
        methods that require at least two points).
        """
        rr_interval_diff = np.diff(rr_intervals)
        rr_interval_abs = np.abs(rr_interval_diff)

        # Note: diffs are in seconds, so use 0.05 (s) threshold for 50 ms.
        return (
            sum(1 for i in rr_interval_abs if i > 0.05)
            if not rr_intervals.size < 2
            else np.nan
        )

    @staticmethod
    def pNN50(nn50: float, rr_intervals: np.ndarray) -> float:
        """
        Percentage of NN50 relative to the number of intervals (as %).

        Returns np.nan if nn50 is np.nan.
        """
        return (
            float(round((float(nn50) / len(rr_intervals)) * 100, 4))
            if not np.isnan(nn50)
            else np.nan
        )

    @staticmethod
    def NN20(rr_intervals: np.ndarray) -> int:
        """
        Count of successive RR interval differences greater than 20 ms.

        Returns np.nan if fewer than 2 intervals exist.
        """
        rr_interval_diff = np.diff(rr_intervals)
        rr_interval_abs = np.abs(rr_interval_diff)

        # 0.02 s threshold for 20 ms
        return (
            sum(1 for i in rr_interval_abs if i > 0.02)
            if not rr_intervals.size < 2
            else np.nan
        )

    @staticmethod
    def pNN20(nn20: float, rr_intervals: np.ndarray) -> float:
        """
        Percentage of NN20 relative to the number of intervals (as %).

        Returns np.nan if nn20 is np.nan.
        """
        return (
            float(round((float(nn20) / len(rr_intervals)) * 100, 4))
            if not np.isnan(nn20)
            else np.nan
        )

    def frequencyAnalysis(
        self,
        rr_intervals: np.ndarray,
        rr_time: np.ndarray,
    ):
        """
        Perform frequency analysis (PSD) of RR intervals.

        Steps:
         1. Interpolate the irregularly-sampled RR series onto a uniform grid
            using cubic spline (`scipy.interpolate.splrep` / `splev`).
            The new time axis is created from rr_time[0] to rr_time[-1] with
            spacing 1 / interpolation_rate (Hz).
         2. Demean the interpolated series (zero-centre) to remove DC bias.
         3. Compute PSD using Welch's method (`scipy.signal.welch`).
         4. Return frequency bins and power spectral density values below 0.5 Hz
            (typical upper limit for HRV spectral analysis).

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals (s).
        rr_time : np.ndarray
            Corresponding timestamps (s). Required for interpolation.

        Returns
        -------
        tuple (freq_axis, power_axis)
            Both are numpy arrays. If inputs are too short for a reliable
            analysis (less than 4 samples here), returns two empty arrays.

        Notes
        -----
        - `self.config.interpolation_rate` sets the sampling rate for the uniform grid.
        - `self.config.window` and `nperseg` determine spectral resolution.
        - The returned PSD is not scaled to ms; callers that compute band power
          might multiply by 1e6 (as done in `frequency_domain_features`) to
          express power in µs^2/Hz-like units depending on convention.
        """
        # Require a minimum number of samples for a reasonable PSD
        if len(rr_time) < 4 or len(rr_intervals) < 4:
            return np.array([]), np.array([])

        # Create uniformly sampled time axis at interpolation_rate Hz
        t_new = np.arange(rr_time[0], rr_time[-1], 1.0 / self.config.interpolation_rate)

        # Spline interpolation of rr_intervals vs rr_time
        # s=0 requests an exact fit through the points
        tck = sc.interpolate.splrep(rr_time, rr_intervals, s=0)
        rr_even = sc.interpolate.splev(t_new, tck)

        # Remove DC component for spectral analysis
        rr_even = rr_even - np.mean(rr_even)

        # Use Welch's method to estimate PSD. nperseg is bounded by len(rr_even)
        freq_axis, power_axis = sc.signal.welch(
            rr_even,
            fs=self.config.interpolation_rate,
            window=sc.signal.get_window(self.config.window, min(len(rr_even), 1000)),
            nperseg=min(len(rr_even), 1000),
        )

        # Trim to typical HRV frequencies (< 0.5 Hz)
        mask = freq_axis < 0.5
        return freq_axis[mask], power_axis[mask]

    def frequency_domain_features(
        self, freqs: np.ndarray, power: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute frequency-domain HRV band powers and normalized units.

        The method integrates PSD values inside configured band ranges (VLF/LF/HF)
        using trapezoidal integration and applies scaling (multiply by 10^6)
        to present results in micro-second-squared-like units (convention).

        Parameters
        ----------
        freqs : np.ndarray
            Frequency axis (Hz).
        power : np.ndarray
            PSD values corresponding to `freqs`.

        Returns
        -------
        dict
            Dictionary containing:
              - VLF_Power, LF_Power, HF_Power, Total_Power
              - LF_(nu), HF_(nu) : normalized units (percentage)
              - LF/HF : ratio of normalized LF to HF

        Notes
        -----
        - If the PSD or frequency arrays are empty or the integration yields
          non-finite results, entries will be set to np.nan.
        """
        def band_power(fmin, fmax):
            """
            Integrate PSD between fmin and fmax using trapezoidal rule.
            Returns np.nan if no frequencies fall in the band.
            """
            idx = (freqs >= fmin) & (freqs < fmax)
            return (
                sc.integrate.trapezoid(power[idx], freqs[idx])
                if np.any(idx)
                else np.nan
            )

        # Compute band powers (scale by 10^6 to match upstream expectations)
        vlf = float(
            band_power(self.config.vlf_lfreq, self.config.vlf_hfreq) * pow(10, 6)
        )
        lf = float(band_power(self.config.lf_lfreq, self.config.lf_hfreq) * pow(10, 6))
        hf = float(band_power(self.config.hf_lfreq, self.config.hf_hfreq) * pow(10, 6))
        total_power = float(
            band_power(self.config.vlf_lfreq, self.config.hf_hfreq) * pow(10, 6)
        )

        # Compute normalized units for LF and HF (exclude VLF from denominator)
        if np.isfinite(total_power) and (total_power - vlf) > 0:
            lf_norm = lf / (total_power - vlf) * 100
            hf_norm = hf / (total_power - vlf) * 100
        else:
            lf_norm = np.nan
            hf_norm = np.nan

        # LF/HF ratio in normalized units (guard against division by zero)
        ratio = (
            lf_norm / hf_norm
            if np.isfinite(lf_norm) and np.isfinite(hf_norm) and hf_norm > 0
            else np.nan
        )

        return {
            "VLF_Power": vlf,
            "LF_Power": lf,
            "HF_Power": hf,
            "Total_Power": total_power,
            "LF_(nu)": lf_norm,
            "HF_(nu)": hf_norm,
            "LF/HF": ratio,
        }

    def non_linear_features(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """
        Compute a small set of non-linear HRV measures derived from the Poincaré plot.

        Features returned:
         - STD : standard deviation of RR intervals (seconds rounded)
         - SDSD : standard deviation of successive differences (seconds)
         - SD1 : Poincaré SD1 (ms)
         - SD2 : Poincaré SD2 (ms)
         - SD2/SD1 : ratio

        Notes
        -----
        - SD1/SD2 are scaled to milliseconds to match typical reporting.
        - Uses helper static methods SDSD, SD1, SD2, SD2_SD1.
        """
        STD = round(float(np.std(rr_intervals)), 4)
        SDSD = float(self.SDSD(rr_intervals))
        SD2 = float(self.SD2(SDSD, STD))
        SD1 = float(self.SD1(SDSD))
        SD2_SD1 = float(self.SD2_SD1(SD1, SD2))

        return {
            "STD": STD,
            "SDSD": SDSD,
            "SD2": SD2,
            "SD1": SD1,
            "SD2/SD1": SD2_SD1,
        }

    @staticmethod
    def SDSD(rr_intervals: np.ndarray) -> float:
        """
        Standard deviation of successive differences (SDSD) in seconds.

        Returns np.nan if fewer than 2 intervals exist.
        """
        diff_rr = np.diff(rr_intervals)
        return float(round(np.std(diff_rr), 4)) if rr_intervals.size >= 2 else np.nan

    @staticmethod
    def SD2(SDSD: float, STD: float) -> float:
        """
        Compute SD2 (Poincaré long-axis) in milliseconds.

        Formula: SD2 = sqrt(2 * STD^2 - 0.5 * SDSD^2) * 1000

        Returns a rounded float (2 decimals).
        """
        return float(round(np.sqrt(2 * STD**2 - 0.5 * SDSD**2) * 1000, 2))

    @staticmethod
    def SD1(SDSD: float) -> float:
        """
        Compute SD1 (Poincaré short-axis) in milliseconds.

        Formula: SD1 = sqrt(0.5 * SDSD^2) * 1000

        Returns a rounded float (2 decimals).
        """
        return float(round(np.sqrt(0.5 * SDSD**2) * 1000, 2))

    @staticmethod
    def SD2_SD1(SD1: float, SD2: float) -> float:
        """
        Ratio SD2/SD1. Returns np.nan if SD1 is zero (to avoid division by zero).
        """
        return float(round(SD2 / SD1, 4)) if SD1 != 0 else np.nan