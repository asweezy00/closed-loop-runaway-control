"""Line-length detector for runaway activity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .config import Config


@dataclass
class FeatureTrace:
    t_s: np.ndarray
    values: np.ndarray


@dataclass
class DetectionResult:
    detection_times_s: list[float]
    threshold: float
    baseline_mean: float
    baseline_std: float
    feature: FeatureTrace
    k: float
    feature_kind: str


def _window_params(t_s: np.ndarray, lfp_z: np.ndarray, cfg: Config) -> tuple[float, int, int]:
    if t_s.size != lfp_z.size:
        raise ValueError("time and LFP arrays must have the same length")
    if t_s.size < 2:
        raise ValueError("at least two samples are required")
    dt_s = float(np.median(np.diff(t_s)))
    win = max(2, int(round(cfg.line_length_window_s / dt_s)))
    step = max(1, int(round(cfg.feature_step_s / dt_s)))
    return dt_s, win, step


def line_length_feature(t_s: np.ndarray, lfp_z: np.ndarray, cfg: Config) -> FeatureTrace:
    """Trailing-window mean absolute derivative feature."""
    _, win, step = _window_params(t_s, lfp_z, cfg)
    vals: list[float] = []
    times: list[float] = []
    for start in range(0, lfp_z.size - win + 1, step):
        seg = lfp_z[start : start + win]
        vals.append(float(np.mean(np.abs(np.diff(seg)))))
        times.append(float(t_s[start + win - 1]))
    return FeatureTrace(t_s=np.asarray(times), values=np.asarray(vals))


def variance_feature(t_s: np.ndarray, lfp_z: np.ndarray, cfg: Config) -> FeatureTrace:
    """Trailing-window variance feature."""
    _, win, step = _window_params(t_s, lfp_z, cfg)
    vals: list[float] = []
    times: list[float] = []
    for start in range(0, lfp_z.size - win + 1, step):
        seg = lfp_z[start : start + win]
        vals.append(float(np.var(seg)))
        times.append(float(t_s[start + win - 1]))
    return FeatureTrace(t_s=np.asarray(times), values=np.asarray(vals))


def band_power_feature(t_s: np.ndarray, lfp_z: np.ndarray, cfg: Config) -> FeatureTrace:
    """Trailing-window 10-40 Hz power using an FFT periodogram estimate."""
    dt_s, win, step = _window_params(t_s, lfp_z, cfg)
    fs_hz = 1.0 / dt_s
    freqs = np.fft.rfftfreq(win, d=dt_s)
    band = (freqs >= cfg.band_power_low_hz) & (freqs <= cfg.band_power_high_hz)
    taper = np.hanning(win)
    taper_norm = float(np.sum(taper**2))
    vals: list[float] = []
    times: list[float] = []
    for start in range(0, lfp_z.size - win + 1, step):
        seg = lfp_z[start : start + win]
        centered = seg - float(np.mean(seg))
        spectrum = np.fft.rfft(centered * taper)
        power = np.square(np.abs(spectrum)) / max(taper_norm * fs_hz, 1e-12)
        vals.append(float(np.mean(power[band])) if np.any(band) else 0.0)
        times.append(float(t_s[start + win - 1]))
    return FeatureTrace(t_s=np.asarray(times), values=np.asarray(vals))


def compute_feature(t_s: np.ndarray, lfp_z: np.ndarray, cfg: Config, feature_kind: str | None = None) -> FeatureTrace:
    """Compute one detector feature trace."""
    kind = cfg.feature_kind if feature_kind is None else feature_kind
    if kind == "line_length":
        return line_length_feature(t_s, lfp_z, cfg)
    if kind == "band_power":
        return band_power_feature(t_s, lfp_z, cfg)
    if kind == "variance":
        return variance_feature(t_s, lfp_z, cfg)
    valid = ", ".join(cfg.feature_kinds)
    raise ValueError(f"unknown feature_kind {kind!r}; expected one of: {valid}")


def _detector_prefilter(t_s: np.ndarray, lfp_z: np.ndarray, cfg: Config) -> np.ndarray:
    """Causal low-pass filter used before line-length extraction.

    The LFP proxy is already low-passed at 100 Hz. This additional detector
    filter limits broadband measurement noise from dominating the line-length
    baseline while preserving seizure-band changes.
    """
    if t_s.size < 3:
        return lfp_z
    fs_hz = 1.0 / float(np.median(np.diff(t_s)))
    cutoff = min(cfg.detector_lowpass_hz, fs_hz * 0.45)
    sos = butter(cfg.detector_filter_order, cutoff, btype="low", fs=fs_hz, output="sos")
    zi = sosfilt_zi(sos) * lfp_z[0]
    filtered, _ = sosfilt(sos, lfp_z, zi=zi)
    return filtered


def detect_feature(
    t_s: np.ndarray,
    lfp_z: np.ndarray,
    cfg: Config,
    k: float | None = None,
    feature_kind: str | None = None,
) -> DetectionResult:
    """Detect threshold crossings with persistence and cooldown."""
    k = cfg.default_k if k is None else float(k)
    kind = cfg.feature_kind if feature_kind is None else feature_kind
    lfp_use = _detector_prefilter(t_s, lfp_z, cfg)
    feature = compute_feature(t_s, lfp_use, cfg, feature_kind=kind)
    baseline = feature.t_s < cfg.baseline_end_s
    if np.any(baseline):
        baseline_mean = float(np.mean(feature.values[baseline]))
        baseline_std = float(np.std(feature.values[baseline]))
    else:
        baseline_mean = 0.0
        baseline_std = 0.0
    threshold = baseline_mean + k * baseline_std + 1e-12

    feature_dt_s = cfg.feature_step_s
    persist_steps = max(1, int(round(cfg.detector_persistence_s / feature_dt_s)))
    cooldown_steps = max(1, int(round(cfg.detector_cooldown_s / feature_dt_s)))

    detection_times: list[float] = []
    counter = 0
    cooldown = 0
    for idx, value in enumerate(feature.values):
        if cooldown > 0:
            cooldown -= 1
            counter = 0
            continue
        if value > threshold:
            counter += 1
            if counter >= persist_steps:
                detection_times.append(float(feature.t_s[idx]))
                counter = 0
                cooldown = cooldown_steps
        else:
            counter = 0

    return DetectionResult(
        detection_times_s=detection_times,
        threshold=threshold,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        feature=feature,
        k=k,
        feature_kind=kind,
    )


def detect_line_length(t_s: np.ndarray, lfp_z: np.ndarray, cfg: Config, k: float | None = None) -> DetectionResult:
    """Backward-compatible line-length detector."""
    return detect_feature(t_s, lfp_z, cfg, k=k, feature_kind="line_length")
