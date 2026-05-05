"""Windowed feature extraction for LFP-like time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, skew


FEATURE_NAMES = (
    "mean",
    "std",
    "variance",
    "rms",
    "abs_mean",
    "ptp",
    "minimum",
    "maximum",
    "skew",
    "kurtosis",
    "line_length",
    "diff_std",
    "max_abs_diff",
    "zero_crossing_rate",
    "slope",
    "energy",
    "band_delta_0p5_4_hz",
    "band_theta_4_8_hz",
    "band_alpha_8_13_hz",
    "band_beta_13_30_hz",
    "band_gamma_30_80_hz",
    "band_10_40_hz",
    "high_to_low_power_ratio",
    "spectral_entropy",
    "dominant_freq_hz",
)


@dataclass
class WindowedFeatures:
    """Feature matrix and aligned window metadata."""

    x: np.ndarray
    y: np.ndarray
    centers_s: np.ndarray
    starts_s: np.ndarray
    ends_s: np.ndarray
    record_ids: list[str]
    record_types: list[str]
    seeds: list[int]


def sampling_rate_hz(t_s: np.ndarray) -> float:
    if t_s.size < 2:
        raise ValueError("at least two time samples are required")
    dt = float(np.median(np.diff(t_s)))
    if dt <= 0:
        raise ValueError("time vector must be strictly increasing")
    return 1.0 / dt


def _band_power(freqs_hz: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs_hz >= low_hz) & (freqs_hz < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freqs_hz[mask]))


def extract_window_feature_vector(window: np.ndarray, fs_hz: float) -> np.ndarray:
    """Extract interpretable time and spectral features from one window."""
    window = np.asarray(window, dtype=float)
    if window.ndim != 1 or window.size < 8:
        raise ValueError("window must be one-dimensional with at least 8 samples")
    if not np.all(np.isfinite(window)):
        raise ValueError("window contains non-finite values")

    centered = window - float(np.mean(window))
    diffs = np.diff(window)
    duration_s = window.size / fs_hz
    time_axis = np.arange(window.size, dtype=float) / fs_hz
    slope = float(np.polyfit(time_axis, window, deg=1)[0]) if window.size > 1 else 0.0

    taper = np.hanning(window.size)
    fft = np.fft.rfft(centered * taper)
    freqs = np.fft.rfftfreq(window.size, d=1.0 / fs_hz)
    power = (np.abs(fft) ** 2) / max(float(np.sum(taper**2)) * fs_hz, 1e-12)
    total_power = float(np.sum(power)) + 1e-12
    prob_power = power / total_power
    spectral_entropy = float(-np.sum(prob_power * np.log2(prob_power + 1e-12)) / np.log2(prob_power.size))
    dominant_freq = float(freqs[int(np.argmax(power[1:]) + 1)]) if power.size > 1 else 0.0

    low_power = _band_power(freqs, power, 0.5, 13.0)
    high_power = _band_power(freqs, power, 13.0, 80.0)

    values = [
        float(np.mean(window)),
        float(np.std(window)),
        float(np.var(window)),
        float(np.sqrt(np.mean(window**2))),
        float(np.mean(np.abs(window))),
        float(np.ptp(window)),
        float(np.min(window)),
        float(np.max(window)),
        float(skew(window, bias=False)),
        float(kurtosis(window, fisher=True, bias=False)),
        float(np.sum(np.abs(diffs)) / max(duration_s, 1e-12)),
        float(np.std(diffs)),
        float(np.max(np.abs(diffs))) if diffs.size else 0.0,
        float(np.mean(np.diff(np.signbit(centered)) != 0)) if diffs.size else 0.0,
        slope,
        float(np.mean(window**2)),
        _band_power(freqs, power, 0.5, 4.0),
        _band_power(freqs, power, 4.0, 8.0),
        _band_power(freqs, power, 8.0, 13.0),
        _band_power(freqs, power, 13.0, 30.0),
        _band_power(freqs, power, 30.0, 80.0),
        _band_power(freqs, power, 10.0, 40.0),
        float(high_power / (low_power + 1e-12)),
        spectral_entropy,
        dominant_freq,
    ]
    out = np.asarray(values, dtype=float)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def window_signal(
    t_s: np.ndarray,
    signal: np.ndarray,
    window_s: float,
    step_s: float,
    positive_interval_s: tuple[float, float] | None,
    record_id: str,
    record_type: str,
    seed: int,
    min_positive_overlap: float = 0.50,
) -> WindowedFeatures:
    """Convert a single record into labeled overlapping windows."""
    t_s = np.asarray(t_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if t_s.shape != signal.shape:
        raise ValueError("time and signal arrays must have matching shape")
    fs_hz = sampling_rate_hz(t_s)
    window_n = max(8, int(round(window_s * fs_hz)))
    step_n = max(1, int(round(step_s * fs_hz)))
    if window_n > signal.size:
        raise ValueError("window is longer than signal")

    rows: list[np.ndarray] = []
    labels: list[int] = []
    centers: list[float] = []
    starts: list[float] = []
    ends: list[float] = []
    ids: list[str] = []
    types: list[str] = []
    seeds: list[int] = []

    for start_idx in range(0, signal.size - window_n + 1, step_n):
        end_idx = start_idx + window_n
        start_s = float(t_s[start_idx])
        end_s = float(t_s[end_idx - 1])
        center_s = 0.5 * (start_s + end_s)
        label = 0
        if positive_interval_s is not None:
            pos_start, pos_end = positive_interval_s
            overlap = max(0.0, min(end_s, pos_end) - max(start_s, pos_start))
            label = int(overlap / max(end_s - start_s, 1e-12) >= min_positive_overlap)

        rows.append(extract_window_feature_vector(signal[start_idx:end_idx], fs_hz))
        labels.append(label)
        centers.append(center_s)
        starts.append(start_s)
        ends.append(end_s)
        ids.append(record_id)
        types.append(record_type)
        seeds.append(int(seed))

    return WindowedFeatures(
        x=np.vstack(rows),
        y=np.asarray(labels, dtype=int),
        centers_s=np.asarray(centers, dtype=float),
        starts_s=np.asarray(starts, dtype=float),
        ends_s=np.asarray(ends, dtype=float),
        record_ids=ids,
        record_types=types,
        seeds=seeds,
    )


def concatenate_feature_sets(parts: list[WindowedFeatures]) -> WindowedFeatures:
    if not parts:
        raise ValueError("at least one feature set is required")
    return WindowedFeatures(
        x=np.vstack([p.x for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        centers_s=np.concatenate([p.centers_s for p in parts]),
        starts_s=np.concatenate([p.starts_s for p in parts]),
        ends_s=np.concatenate([p.ends_s for p in parts]),
        record_ids=[rid for p in parts for rid in p.record_ids],
        record_types=[rtype for p in parts for rtype in p.record_types],
        seeds=[seed for p in parts for seed in p.seeds],
    )
