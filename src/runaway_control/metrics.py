"""Scoring and suppression metrics for the final project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from brian2 import Hz, second

from .config import Config
from .controller import StimSchedule
from .lfp import LfpResult
from .network import NetworkRun


@dataclass
class DetectionScore:
    sensitivity: float
    false_alarms_per_hour: float
    latency_ms: float
    tp_events: int
    fp_events: int
    n_events: int


@dataclass
class SuppressionMetrics:
    lfp_power_suppression_pct: float
    e_rate_suppression_pct: float
    runaway_duration_reduction_pct: float
    open_eval_rate_hz: float
    closed_eval_rate_hz: float
    open_runaway_duration_s: float
    closed_runaway_duration_s: float
    n_pulses: int
    duty_cycle: float
    burden_nS_s: float


def _merge_events(times: list[float], tolerance_s: float = 0.30) -> list[tuple[float, float]]:
    times = sorted(float(t) for t in times)
    if not times:
        return []
    events = [[times[0], times[0]]]
    for t in times[1:]:
        if t - events[-1][1] <= tolerance_s:
            events[-1][1] = t
        else:
            events.append([t, t])
    return [(float(a), float(b)) for a, b in events]


def score_detections(detection_times_s: list[float], cfg: Config) -> DetectionScore:
    """Event-style score for one simulated runaway episode."""
    events = _merge_events(detection_times_s)
    tp_events = [ev for ev in events if ev[1] >= cfg.runaway_onset_s]
    fp_events = [ev for ev in events if ev[1] < cfg.runaway_onset_s]
    sensitivity = 1.0 if tp_events else 0.0
    latency_ms = float("nan")
    if tp_events:
        latency_ms = max(0.0, tp_events[0][0] - cfg.runaway_onset_s) * 1000.0
    false_alarm_hours = max(cfg.baseline_end_s / 3600.0, 1e-12)
    false_alarms_per_hour = len(fp_events) / false_alarm_hours
    return DetectionScore(
        sensitivity=sensitivity,
        false_alarms_per_hour=false_alarms_per_hour,
        latency_ms=latency_ms,
        tp_events=len(tp_events),
        fp_events=len(fp_events),
        n_events=len(events),
    )


def score_false_alarms(detection_times_s: list[float], total_s: float) -> DetectionScore:
    """Score detections when no runaway event exists in the record."""
    events = _merge_events(detection_times_s)
    non_event_hours = max(float(total_s) / 3600.0, 1e-12)
    return DetectionScore(
        sensitivity=float("nan"),
        false_alarms_per_hour=len(events) / non_event_hours,
        latency_ms=float("nan"),
        tp_events=0,
        fp_events=len(events),
        n_events=len(events),
    )


def _rate_window_hz(run: NetworkRun, start_s: float, end_s: float) -> float:
    spikes = np.asarray(run.monitors["spike_E"].t / second, dtype=float)
    n_spikes = int(np.sum((spikes >= start_s) & (spikes < end_s)))
    duration = max(end_s - start_s, 1e-12)
    return n_spikes / run.config.n_e / duration


def rate_trace(run: NetworkRun, width_ms: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    monitor = run.monitors["rate_E"]
    t_s = np.asarray(monitor.t / second, dtype=float)
    rate_hz = np.asarray(monitor.smooth_rate(window="gaussian", width=width_ms * second / 1000.0) / Hz, dtype=float)
    return t_s, rate_hz


def _runaway_duration_from_rate(run: NetworkRun, cfg: Config) -> float:
    t_s, rate_hz = rate_trace(run)
    baseline = t_s < cfg.baseline_end_s
    post = t_s >= cfg.runaway_onset_s
    if not np.any(baseline) or not np.any(post):
        return 0.0
    threshold = float(np.mean(rate_hz[baseline]) + 3.0 * np.std(rate_hz[baseline]))
    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else cfg.dt_ms / 1000.0
    return float(np.sum(rate_hz[post] > threshold) * dt)


def _safe_pct_reduction(open_value: float, closed_value: float) -> float:
    if not np.isfinite(open_value) or abs(open_value) < 1e-12:
        return 0.0
    return 100.0 * (open_value - closed_value) / open_value


def compute_suppression_metrics(
    open_run: NetworkRun,
    closed_run: NetworkRun,
    open_lfp: LfpResult,
    closed_lfp: LfpResult,
    schedule: StimSchedule,
) -> SuppressionMetrics:
    cfg = open_run.config
    eval_start = cfg.runaway_onset_s
    eval_end = cfg.t_total_s

    open_mask = (open_lfp.t_s >= eval_start) & (open_lfp.t_s < eval_end)
    closed_mask = (closed_lfp.t_s >= eval_start) & (closed_lfp.t_s < eval_end)
    open_power = float(np.mean(np.square(open_lfp.lfp_z[open_mask]))) if np.any(open_mask) else 0.0
    closed_power = float(np.mean(np.square(closed_lfp.lfp_z[closed_mask]))) if np.any(closed_mask) else 0.0

    open_rate = _rate_window_hz(open_run, eval_start, eval_end)
    closed_rate = _rate_window_hz(closed_run, eval_start, eval_end)
    open_duration = _runaway_duration_from_rate(open_run, cfg)
    closed_duration = _runaway_duration_from_rate(closed_run, cfg)

    return SuppressionMetrics(
        lfp_power_suppression_pct=_safe_pct_reduction(open_power, closed_power),
        e_rate_suppression_pct=_safe_pct_reduction(open_rate, closed_rate),
        runaway_duration_reduction_pct=_safe_pct_reduction(open_duration, closed_duration),
        open_eval_rate_hz=open_rate,
        closed_eval_rate_hz=closed_rate,
        open_runaway_duration_s=open_duration,
        closed_runaway_duration_s=closed_duration,
        n_pulses=schedule.n_pulses,
        duty_cycle=schedule.duty_cycle,
        burden_nS_s=schedule.burden_nS_s,
    )
