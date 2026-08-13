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

    A concrete implementation of the `HRV_base` interface that computes
    HRV metrics using standard signal processing and spectral analysis
    techniques. This is a good starting point for HRV analysis; alternative
    implementations can be substituted by providing a different class that
    also implements `HRV_base`.

    Attributes
    ----------
    config : HRV_Config
        Configuration object containing algorithm parameters such as:
          - ectopy_threshold: relative difference threshold for ectopy detection
          - interpolation_rate: sampling rate (Hz) for uniform RR resampling
          - window: window function name for Welch PSD (e.g., "hann", "hamming")
          - vlf_lfreq, vlf_hfreq: Very Low Frequency band limits (Hz)
          - lf_lfreq, lf_hfreq: Low Frequency band limits (Hz)
          - hf_lfreq, hf_hfreq: High Frequency band limits (Hz)

    Parameters
    ----------
    config : HRV_Config
        Configuration instance for the algorithm. If not provided by the caller,
        the algorithm may create a default `HRV_Config()`.
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
        - Very small arrays (size < 2) are returned unchanged (as a copy).
        - The algorithm uses a simple local relative-difference rule; more
          sophisticated methods (adaptive filters, template matching) can be
          substituted by providing a different algorithm implementation.
        - The ectopy threshold is read from `self.config.ectopy_threshold`.
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

        Keys
        ----
        "Avg RR" : float
            Mean RR interval (seconds)
        "Min RR" : float
            Minimum RR interval (seconds)
        "Max RR" : float
            Maximum RR interval (seconds)
        "SD RR" : float
            Standard deviation of RR intervals (seconds)
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

        Keys
        ----
        "Avg HR" : float
            Mean heart rate (beats per minute)
        "Min HR" : float
            Minimum heart rate (beats per minute)
        "Max HR" : float
            Maximum heart rate (beats per minute)
        "SD HR" : float
            Standard deviation of heart rate (beats per minute)
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

        Keys
        ----
        "Avg RR" : float
            Mean RR interval (milliseconds)
        "Min RR" : float
            Minimum RR interval (milliseconds)
        "Max RR" : float
            Maximum RR interval (milliseconds)
        "SD RR" : float
            Standard deviation of RR intervals (milliseconds)
        "SDNN" : float
            Standard deviation of NN intervals (milliseconds)
        "RMSSD" : float
            Root mean square of successive differences (milliseconds)
        "NN50" : int or float
            Count of successive differences > 50 ms
        "pNN50" : float
            Percentage of NN50 (%)
        "NN20" : int or float
            Count of successive differences > 20 ms
        "pNN20" : float
            Percentage of NN20 (%)
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

        SDNN is a fundamental time-domain HRV metric representing the standard
        deviation of the normal-to-normal (NN) RR interval series. It reflects
        overall heart rate variability.

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals in seconds.

        Returns
        -------
        float
            SDNN value in milliseconds, rounded to 4 decimal places.
            Returns np.nan if input is empty.

        Notes
        -----
        - SDNN captures variability over the entire recording.
        - Higher SDNN generally indicates higher HRV (healthier autonomic function).
        - Result is scaled to milliseconds (multiplied by 1000).
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

        RMSSD is computed as sqrt( sum(diff^2) / (N-1) ) and measures the
        variability between successive RR intervals. It is sensitive to
        short-term heart rate variability and parasympathetic activity.

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals in seconds.

        Returns
        -------
        float
            RMSSD value in milliseconds, rounded to 4 decimal places.
            Returns np.nan if fewer than 2 intervals are available.

        Notes
        -----
        - RMSSD is a time-domain measure reflecting beat-to-beat variability.
        - It is particularly sensitive to parasympathetic (vagal) activity.
        - Often reported as one of the most reliable HRV metrics.
        - Result is scaled to milliseconds (multiplied by 1000).
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

        NN50 counts the number of pairs of successive RR intervals that differ
        by more than 50 milliseconds. It is used to compute pNN50 (percentage).

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals in seconds.

        Returns
        -------
        int or float
            Count of successive differences > 50 ms.
            Returns np.nan if fewer than 2 intervals exist.

        Notes
        -----
        - Threshold is 0.05 seconds (50 ms) for input in seconds.
        - Sensitive to parasympathetic activity and vagal tone.
        - Often paired with pNN50 (percentage metric).
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

        Converts the NN50 count to a percentage, normalizing by the total
        number of RR intervals. Often more interpretable than raw counts.

        Parameters
        ----------
        nn50 : float
            NN50 count (output from NN50()).
        rr_intervals : np.ndarray
            Array of RR intervals (used to get total count).

        Returns
        -------
        float
            pNN50 value as a percentage (0–100), rounded to 4 decimal places.
            Returns np.nan if nn50 is np.nan.

        Notes
        -----
        - Formula: (NN50 / N) * 100, where N is the total number of intervals.
        - pNN50 > 50% is sometimes considered a marker of good parasympathetic tone.
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

        NN20 is similar to NN50 but uses a lower threshold (20 ms). It captures
        more subtle beat-to-beat variations and is used to compute pNN20.

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals in seconds.

        Returns
        -------
        int or float
            Count of successive differences > 20 ms.
            Returns np.nan if fewer than 2 intervals exist.

        Notes
        -----
        - Threshold is 0.02 seconds (20 ms) for input in seconds.
        - More sensitive to variability than NN50 (lower threshold).
        - Can capture additional information about vagal modulation.
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

        Converts the NN20 count to a percentage, normalizing by the total
        number of RR intervals (analogous to pNN50).

        Parameters
        ----------
        nn20 : float
            NN20 count (output from NN20()).
        rr_intervals : np.ndarray
            Array of RR intervals (used to get total count).

        Returns
        -------
        float
            pNN20 value as a percentage (0–100), rounded to 4 decimal places.
            Returns np.nan if nn20 is np.nan.

        Notes
        -----
        - Formula: (NN20 / N) * 100, where N is the total number of intervals.
        - Generally, pNN20 > pNN50 due to the lower threshold.
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

        Computes the power spectral density (PSD) of the RR interval series by:
         1. Interpolating the irregularly-sampled RR series onto a uniform grid
            using cubic spline (`scipy.interpolate.splrep` / `splev`).
            The new time axis is created from rr_time[0] to rr_time[-1] with
            spacing 1 / interpolation_rate (Hz).
         2. Demeaning the interpolated series (zero-centering) to remove DC bias.
         3. Computing PSD using Welch's method (`scipy.signal.welch`).
         4. Returning frequency bins and power spectral density values below 0.5 Hz
            (typical upper limit for HRV spectral analysis).

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in seconds. Shape: (N,).
        rr_time : np.ndarray
            Corresponding timestamps in seconds. Shape: (N,).
            Required for interpolation to establish time-domain correspondence.

        Returns
        -------
        tuple (freq_axis, power_axis)
            Both are numpy arrays with matching lengths.
            - freq_axis: 1D array of frequency bin centres (Hz)
            - power_axis: 1D array of PSD values corresponding to freq_axis
            If inputs are too short for a reliable analysis (less than 4 samples),
            returns two empty arrays.

        Notes
        -----
        - `self.config.interpolation_rate` sets the sampling rate for the uniform grid.
        - `self.config.window` specifies the window function for Welch's method.
        - Segment size (nperseg) is capped at 1000 samples by default.
        - The returned PSD is not scaled; callers that compute band power
          might multiply by 1e6 to express power in µs^2/Hz-like units.
        - Requires: scipy.interpolate and scipy.signal
        """
        # Require a minimum number of samples for a reasonable PSD
        if len(rr_time) < 4 or len(rr_intervals) < 4:
            return np.array([]), np.array([])

        # Create uniformly sampled time axis at interpolation_rate Hz
        t_new = np.arange(rr_time[0], rr_time[-1], 1.0 / self.config.interpolation_rate)

        # Spline interpolation of rr_intervals vs rr_time
        # s=0 requests an exact fit through the points (no smoothing)
        tck = sc.interpolate.splrep(rr_time, rr_intervals, s=0)
        rr_even = sc.interpolate.splev(t_new, tck)

        # Remove DC component for spectral analysis (mean subtraction)
        rr_even = rr_even - np.mean(rr_even)

        # Use Welch's method to estimate PSD. nperseg is bounded by len(rr_even)
        # and capped at 1000 samples for computational efficiency.
        freq_axis, power_axis = sc.signal.welch(
            rr_even,
            fs=self.config.interpolation_rate,
            window=sc.signal.get_window(self.config.window, min(len(rr_even), 1000)),
            nperseg=min(len(rr_even), 1000),
        )

        # Trim to typical HRV frequencies (< 0.5 Hz). Frequencies above this
        # are typically noise or respiratory artifacts in HRV analysis.
        mask = freq_axis < 0.5
        return freq_axis[mask], power_axis[mask]

    def frequency_domain_features(
        self, freqs: np.ndarray, power: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute frequency-domain HRV band powers and normalized units.

        Integrates the power spectral density over standard HRV frequency bands
        (Very Low Frequency, Low Frequency, High Frequency) and computes
        normalized units and ratios. Band limits and scaling factors are read
        from the configuration.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency axis in Hz. Shape: (M,).
        power : np.ndarray
            PSD values corresponding to `freqs`. Shape: (M,).

        Returns
        -------
        dict
            Dictionary containing frequency-domain features:

            Keys
            ----
            "VLF_Power" : float
                Very Low Frequency band power (0.0033–0.04 Hz, scaled by 10^6)
            "LF_Power" : float
                Low Frequency band power (0.04–0.15 Hz, scaled by 10^6)
            "HF_Power" : float
                High Frequency band power (0.15–0.4 Hz, scaled by 10^6)
            "Total_Power" : float
                Sum of VLF, LF, and HF powers (scaled by 10^6)
            "LF_(nu)" : float
                LF power in normalized units (%), computed as LF / (Total - VLF) * 100
            "HF_(nu)" : float
                HF power in normalized units (%), computed as HF / (Total - VLF) * 100
            "LF/HF" : float
                Ratio of normalized LF to HF. np.nan if HF is 0 or invalid.

        Notes
        -----
        - Integration uses trapezoidal rule (`scipy.integrate.trapezoid`).
        - Band limits (vlf_lfreq, vlf_hfreq, etc.) are read from `self.config`.
        - Scaling by 10^6 is applied to match upstream HRV reporting conventions.
        - If the PSD or frequency arrays are empty or integration yields
          non-finite results, entries are set to np.nan.
        - Normalized units exclude VLF from the denominator (LF + HF basis).
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

        # Compute band powers. Scale by 10^6 for convention (µs^2/Hz-like units).
        vlf = float(
            band_power(self.config.vlf_lfreq, self.config.vlf_hfreq) * pow(10, 6)
        )
        lf = float(band_power(self.config.lf_lfreq, self.config.lf_hfreq) * pow(10, 6))
        hf = float(band_power(self.config.hf_lfreq, self.config.hf_hfreq) * pow(10, 6))
        total_power = float(
            band_power(self.config.vlf_lfreq, self.config.hf_hfreq) * pow(10, 6)
        )

        # Compute normalized units for LF and HF (exclude VLF from denominator).
        # This normalization emphasizes the balance between sympathetic and
        # parasympathetic nervous system activity.
        if np.isfinite(total_power) and (total_power - vlf) > 0:
            lf_norm = lf / (total_power - vlf) * 100
            hf_norm = hf / (total_power - vlf) * 100
        else:
            lf_norm = np.nan
            hf_norm = np.nan

        # LF/HF ratio in normalized units (guard against division by zero).
        # This ratio is often used as a marker of sympatho-vagal balance.
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

        Non-linear methods capture complex, chaotic, or scaling properties of
        the RR sequence. The Poincaré plot is a 2D scatter plot of (RR_n, RR_{n+1}),
        and SD1/SD2 measure the spread perpendicular and parallel to the line
        of identity y=x.

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals in seconds.

        Returns
        -------
        dict
            Dictionary of non-linear HRV features:

            Keys
            ----
            "STD" : float
                Standard deviation of RR intervals (seconds, rounded)
            "SDSD" : float
                Standard deviation of successive differences (seconds)
            "SD1" : float
                Poincaré short-axis scatter (milliseconds)
            "SD2" : float
                Poincaré long-axis scatter (milliseconds)
            "SD2/SD1" : float
                Ratio of SD2 to SD1. np.nan if SD1 is 0.

        Notes
        -----
        - SD1 relates to short-term variability (parasympathetic activity).
        - SD2 relates to long-term variability (overall complexity).
        - SD1 and SD2 are scaled to milliseconds by the helper methods.
        - Non-linear metrics are sensitive to signal quality and length.
        - Formula: SD1 = sqrt(0.5 * SDSD^2), SD2 = sqrt(2*STD^2 - 0.5*SDSD^2)
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

        Computes the standard deviation of the first-order differences of RR
        intervals (i.e., consecutive changes in heart rate).

        Parameters
        ----------
        rr_intervals : np.ndarray
            Array of RR intervals in seconds.

        Returns
        -------
        float
            SDSD value in seconds, rounded to 4 decimal places.
            Returns np.nan if fewer than 2 intervals exist.

        Notes
        -----
        - SDSD is related to short-term variability and parasympathetic tone.
        - Used as an intermediate value in Poincaré plot calculations (SD1/SD2).
        """
        diff_rr = np.diff(rr_intervals)
        return float(round(np.std(diff_rr), 4)) if rr_intervals.size >= 2 else np.nan

    @staticmethod
    def SD2(SDSD: float, STD: float) -> float:
        """
        Compute SD2 (Poincaré long-axis) in milliseconds.

        SD2 measures the variability along the line of identity (y=x) in the
        Poincaré plot and represents long-term HRV and overall complexity.

        Parameters
        ----------
        SDSD : float
            Standard deviation of successive differences (from SDSD method).
        STD : float
            Standard deviation of RR intervals (in seconds).

        Returns
        -------
        float
            SD2 value in milliseconds, rounded to 2 decimal places.

        Notes
        -----
        - Formula: SD2 = sqrt(2 * STD^2 - 0.5 * SDSD^2) * 1000
        - SD2 is typically larger than SD1.
        - Reflects long-term variability and overall heart rate complexity.
        """
        return float(round(np.sqrt(2 * STD**2 - 0.5 * SDSD**2) * 1000, 2))

    @staticmethod
    def SD1(SDSD: float) -> float:
        """
        Compute SD1 (Poincaré short-axis) in milliseconds.

        SD1 measures the variability perpendicular to the line of identity in
        the Poincaré plot and represents short-term, beat-to-beat variability
        (closely related to RMSSD and parasympathetic tone).

        Parameters
        ----------
        SDSD : float
            Standard deviation of successive differences (from SDSD method).

        Returns
        -------
        float
            SD1 value in milliseconds, rounded to 2 decimal places.

        Notes
        -----
        - Formula: SD1 = sqrt(0.5 * SDSD^2) * 1000 = SDSD / sqrt(2) * 1000
        - SD1 is typically smaller than SD2.
        - Strongly correlated with RMSSD; captures parasympathetic activity.
        """
        return float(round(np.sqrt(0.5 * SDSD**2) * 1000, 2))

    @staticmethod
    def SD2_SD1(SD1: float, SD2: float) -> float:
        """
        Ratio SD2/SD1.

        Computes the ratio of the Poincaré long-axis to short-axis. This ratio
        is often used as a single metric reflecting the balance between long-term
        and short-term variability.

        Parameters
        ----------
        SD1 : float
            Poincaré short-axis (from SD1 method, in milliseconds).
        SD2 : float
            Poincaré long-axis (from SD2 method, in milliseconds).

        Returns
        -------
        float
            SD2/SD1 ratio, rounded to 4 decimal places.
            Returns np.nan if SD1 is zero (to avoid division by zero).

        Notes
        -----
        - Typically SD2/SD1 > 1 due to SD2 > SD1.
        - Used as a single marker combining short-term and long-term variability.
        - Higher values indicate more complex, variable heart rate patterns.
        """
        return float(round(SD2 / SD1, 4)) if SD1 != 0 else np.nan

    def get_config(self) -> Dict[str, Any]:
        """
        Retrieve the configuration parameters used by the algorithm.

        Returns a dictionary containing all configuration settings used by the
        algorithm (e.g., sampling rate, filter parameters, peak detection
        thresholds). This allows callers to inspect how the algorithm is
        configured and serialize/persist the configuration if needed.

        Returns
        -------
        Dict[str, Any]
            Dictionary of configuration parameters. Keys and values are
            derived from the `EDA_Config` instance.
        """
        return self.config.__dict__
