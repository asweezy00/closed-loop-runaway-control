"""Safety-limited intervention scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass
class StimSchedule:
    requested_times_s: list[float]
    safe_times_s: list[float]
    n_pulses: int
    duty_cycle: float
    burden_nS_s: float
    skipped_cooldown: int
    skipped_duty: int
    skipped_max_pulses: int


def filter_stim_times(
    detection_times_s: list[float],
    cfg: Config,
    added_latency_s: float = 0.0,
) -> StimSchedule:
    """Apply amplitude, pulse-count, spacing, and duty constraints."""
    requested = [float(t) + float(added_latency_s) for t in detection_times_s]
    requested = [t for t in requested if 0.0 <= t <= cfg.t_total_s - cfg.stim_duration_s]
    safe: list[float] = []
    last_start = -1e9
    total_stim_time = 0.0
    skipped_cooldown = 0
    skipped_duty = 0
    skipped_max = 0

    for t in sorted(requested):
        if len(safe) >= cfg.max_pulses:
            skipped_max += 1
            continue
        if safe and t < last_start + cfg.min_pulse_spacing_s:
            skipped_cooldown += 1
            continue
        projected_duty = (total_stim_time + cfg.stim_duration_s) / cfg.t_total_s
        if projected_duty > cfg.max_duty_cycle:
            skipped_duty += 1
            continue
        safe.append(t)
        last_start = t
        total_stim_time += cfg.stim_duration_s

    burden = len(safe) * cfg.stim_amplitude_cap_nS * cfg.stim_duration_s * cfg.stim_fraction
    return StimSchedule(
        requested_times_s=requested,
        safe_times_s=safe,
        n_pulses=len(safe),
        duty_cycle=total_stim_time / cfg.t_total_s,
        burden_nS_s=burden,
        skipped_cooldown=skipped_cooldown,
        skipped_duty=skipped_duty,
        skipped_max_pulses=skipped_max,
    )
