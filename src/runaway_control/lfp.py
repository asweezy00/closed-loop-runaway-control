"""Simulated LFP proxy from recorded synaptic conductances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from brian2 import nS, second
from scipy.signal import butter, sosfilt, sosfilt_zi

from .config import Config


@dataclass
class LfpResult:
    t_s: np.ndarray
    lfp_z: np.ndarray
    filtered_raw: np.ndarray
    raw: np.ndarray


def _lowpass(signal: np.ndarray, fs_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    sos = butter(order, cutoff_hz, btype="low", fs=fs_hz, output="sos")
    zi = sosfilt_zi(sos) * signal[0]
    filtered, _ = sosfilt(sos, signal, zi=zi)
    return filtered


def compute_lfp(state_monitor, cfg: Config) -> LfpResult:
    """Compute weighted, low-passed, baseline-z-scored LFP proxy."""
    t_s = np.asarray(state_monitor.t / second, dtype=float)
    if t_s.size == 0:
        raise ValueError("state monitor is empty")

    g_e = np.asarray(state_monitor.g_e / nS, dtype=float)
    g_i = np.asarray(state_monitor.g_i / nS, dtype=float)
    if g_e.ndim == 1:
        g_e = g_e[np.newaxis, :]
        g_i = g_i[np.newaxis, :]

    raw = cfg.alpha_lfp * np.sum(g_e, axis=0) + cfg.beta_lfp * np.sum(g_i, axis=0)
    fs_hz = 1.0 / (cfg.record_dt_ms / 1000.0)
    filtered = _lowpass(raw, fs_hz, cfg.lfp_lowpass_hz, cfg.lfp_filter_order)

    baseline = t_s < cfg.baseline_end_s
    if not np.any(baseline):
        raise ValueError("no baseline samples available for LFP z-score")
    mu = float(np.mean(filtered[baseline]))
    sigma = float(np.std(filtered[baseline]))
    lfp_z = (filtered - mu) / (sigma + 1e-12)
    return LfpResult(t_s=t_s, lfp_z=lfp_z, filtered_raw=filtered, raw=raw)


def add_gaussian_noise(signal: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Add Gaussian measurement noise at the requested SNR."""
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(np.square(signal)))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-12)
    return signal + rng.standard_normal(signal.size) * np.sqrt(noise_power)


def add_artifact_burst(t_s: np.ndarray, signal: np.ndarray, seed: int, amplitude: float = 1.0) -> np.ndarray:
    """Add a deterministic post-calibration artifact burst for false-alarm testing."""
    rng = np.random.default_rng(seed)
    start = float(rng.uniform(2.25, 2.75))
    duration = 0.30
    freq = float(rng.uniform(24.0, 34.0))
    out = signal.copy()
    mask = (t_s >= start) & (t_s < start + duration)
    out[mask] += amplitude * np.sin(2.0 * np.pi * freq * t_s[mask])
    return out


def add_artifact_bursts(
    t_s: np.ndarray,
    signal: np.ndarray,
    seed: int,
    rate_per_min: float,
    duration_s: float,
    amplitude: float,
    min_start_s: float,
) -> tuple[np.ndarray, list[float]]:
    """Add randomized post-calibration artifact bursts for long false-alarm testing."""
    if t_s.size == 0:
        return signal.copy(), []
    rng = np.random.default_rng(seed)
    t_end = float(t_s[-1])
    available_s = max(0.0, t_end - min_start_s - duration_s)
    expected = max(0.0, rate_per_min * available_s / 60.0)
    n_bursts = int(rng.poisson(expected))
    out = signal.copy()
    starts: list[float] = []
    if n_bursts == 0 or available_s <= 0.0:
        return out, starts
    for start in sorted(rng.uniform(min_start_s, min_start_s + available_s, size=n_bursts)):
        freq = float(rng.uniform(24.0, 34.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        mask = (t_s >= start) & (t_s < start + duration_s)
        out[mask] += amplitude * np.sin(2.0 * np.pi * freq * t_s[mask] + phase)
        starts.append(float(start))
    return out, starts
