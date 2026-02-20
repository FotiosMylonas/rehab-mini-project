# signal_processing.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def robust_scale(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    return float(scale if scale > 1e-12 else (np.std(x) + 1e-12))


def infer_time_and_fs(df: pd.DataFrame, fallback_fs: float = 100.0) -> tuple[np.ndarray, float]:
    candidate_time_cols = ["time", "timestamp", "t", "Time", "Timestamp"]
    time_col = next((c for c in candidate_time_cols if c in df.columns), None)

    n = len(df)
    if time_col is None:
        fs = float(fallback_fs)
        t = np.arange(n) / fs
        return t, fs

    raw = df[time_col]

    if np.issubdtype(raw.dtype, np.datetime64):
        t = (raw - raw.iloc[0]).dt.total_seconds().to_numpy()
    else:
        if raw.dtype == object:
            parsed = pd.to_datetime(raw, errors="coerce")
            if parsed.notna().mean() > 0.9:
                t = (parsed - parsed.iloc[0]).dt.total_seconds().to_numpy()
            else:
                t = pd.to_numeric(raw, errors="coerce").to_numpy()
        else:
            t = pd.to_numeric(raw, errors="coerce").to_numpy()

        t = t - np.nanmin(t)
        if np.nanmax(t) > 1e5:
            t = t / 1000.0

    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) < 10:
        fs = float(fallback_fs)
        return np.arange(n) / fs, fs

    med_dt = np.median(dt)
    if med_dt <= 0:
        fs = float(fallback_fs)
        return np.arange(n) / fs, fs

    fs = 1.0 / med_dt
    if fs < 10 or fs > 500:
        fs = float(fallback_fs)
        t = np.arange(n) / fs

    return t, float(fs)


def choose_signal(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    cols_lower = {c.lower(): c for c in df.columns}

    ax = cols_lower.get("acc_x") or cols_lower.get("ax")
    ay = cols_lower.get("acc_y") or cols_lower.get("ay")
    az = cols_lower.get("acc_z") or cols_lower.get("az")

    if ax and ay and az:
        x = pd.to_numeric(df[ax], errors="coerce").to_numpy()
        y = pd.to_numeric(df[ay], errors="coerce").to_numpy()
        z = pd.to_numeric(df[az], errors="coerce").to_numpy()
        mag = np.sqrt(x**2 + y**2 + z**2)
        return mag, "acc_magnitude"

    time_like = {"time", "timestamp", "t"}
    numeric_cols = []
    for c in df.columns:
        if c.lower() in time_like:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.8:
            numeric_cols.append(c)

    if not numeric_cols:
        if df.shape[1] >= 2:
            c = df.columns[1]
            return pd.to_numeric(df[c], errors="coerce").to_numpy(), str(c)
        raise ValueError("No suitable numeric signal column found.")

    c = numeric_cols[0]
    return pd.to_numeric(df[c], errors="coerce").to_numpy(), str(c)


def detect_steps(
    filtered: np.ndarray,
    fs: float,
    time_s: np.ndarray,
    min_step_hz: float = 0.4,
    max_step_hz: float = 3.0,
    prominence_k: float = 1.2,
    height_k: float = 0.5,
) -> tuple[np.ndarray, dict]:
    x = np.asarray(filtered, dtype=float)
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy()

    xc = x - np.median(x)
    scale = robust_scale(xc)

    prominence = prominence_k * scale
    height = height_k * scale
    min_distance = max(1, int(fs / max_step_hz))

    peaks_pos, _ = find_peaks(xc, distance=min_distance, prominence=prominence, height=height)
    peaks_neg, _ = find_peaks(-xc, distance=min_distance, prominence=prominence, height=height)

    def score(peaks: np.ndarray) -> float:
        if len(peaks) < 3:
            return -np.inf
        pt = time_s[peaks]
        intervals = np.diff(pt)
        if len(intervals) < 2:
            return -np.inf
        mean_interval = np.mean(intervals)
        if mean_interval <= 0:
            return -np.inf
        freq = 1.0 / mean_interval
        if not (min_step_hz <= freq <= max_step_hz):
            return -np.inf
        cv = np.std(intervals) / (np.mean(intervals) + 1e-12)
        return float(len(peaks) - 20.0 * cv)

    score_pos = score(peaks_pos)
    score_neg = score(peaks_neg)

    if score_pos >= score_neg:
        peaks = peaks_pos
        polarity = "positive"
    else:
        peaks = peaks_neg
        polarity = "negative"

    duration = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else float(len(x) / fs)
    step_count = int(len(peaks))
    mean_freq_hz = float(step_count / duration) if duration > 0 else float("nan")
    cadence_spm = float(mean_freq_hz * 60.0)

    if step_count >= 3:
        pt = time_s[peaks]
        intervals = np.diff(pt)
        interval_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-12))
    else:
        interval_cv = float("nan")

    metrics = {
        "chosen_peak_polarity": polarity,
        "step_count": step_count,
        "duration_s": duration,
        "mean_frequency_hz": mean_freq_hz,
        "cadence_steps_per_min": cadence_spm,
        "step_interval_cv": interval_cv,
        "min_peak_distance_samples": int(min_distance),
        "peak_prominence_used": float(prominence),
        "peak_height_used": float(height),
    }
    return peaks, metrics