"""Long no-runaway false-alarm validation for the ML detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from .features import WindowedFeatures, window_signal
from .metrics import binary_metrics
from .models import score_estimator


FINAL_PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_PROJECT_SRC = FINAL_PROJECT_ROOT / "src"
if str(FINAL_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(FINAL_PROJECT_SRC))

from runaway_control.config import Config
from runaway_control.lfp import add_artifact_bursts, compute_lfp
from runaway_control.network import run_network


@dataclass
class LongRecordScores:
    """ML scores for one long no-runaway record."""

    seed: int
    duration_s: float
    t_s: np.ndarray
    signal: np.ndarray
    artifact_times_s: list[float]
    windows: WindowedFeatures
    scores: np.ndarray


def long_no_runaway_config(base_cfg: Config, quick: bool = False) -> Config:
    """Use the project long-baseline durations while disabling runaway induction."""
    duration_s = base_cfg.quick_long_baseline_duration_s if quick else base_cfg.long_baseline_duration_s
    return base_cfg.with_updates(t_total_s=duration_s, inhibitory_weight_factor=1.0)


def merge_positive_windows(
    centers_s: np.ndarray,
    positive_mask: np.ndarray,
    gap_tolerance_s: float = 0.30,
) -> list[tuple[float, float]]:
    """Merge nearby positive windows into event-style false alarms."""
    centers = np.asarray(centers_s, dtype=float)
    positives = centers[np.asarray(positive_mask, dtype=bool)]
    if positives.size == 0:
        return []
    positives = np.sort(positives)
    events: list[list[float]] = [[float(positives[0]), float(positives[0])]]
    for center in positives[1:]:
        center = float(center)
        if center - events[-1][1] <= gap_tolerance_s:
            events[-1][1] = center
        else:
            events.append([center, center])
    return [(start, end) for start, end in events]


def score_false_alarms_per_hour(
    centers_s: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    duration_s: float,
    gap_tolerance_s: float = 0.30,
) -> dict[str, float | int]:
    """Convert ML window probabilities into event-level FA/h."""
    events = merge_positive_windows(centers_s, np.asarray(scores) >= threshold, gap_tolerance_s=gap_tolerance_s)
    duration_h = max(float(duration_s) / 3600.0, 1e-12)
    return {
        "threshold": float(threshold),
        "false_alarm_events": int(len(events)),
        "false_alarms_per_hour": float(len(events) / duration_h),
        "positive_windows": int(np.sum(np.asarray(scores) >= threshold)),
    }


def run_long_record_scores(
    seed: int,
    cfg: Config,
    estimator,
    window_s: float,
    step_s: float,
) -> LongRecordScores:
    """Simulate one no-runaway artifact-corrupted record and score every window."""
    run = run_network(cfg, seed=seed)
    lfp = compute_lfp(run.monitors["state"], cfg)
    signal, artifact_times = add_artifact_bursts(
        lfp.t_s,
        lfp.lfp_z,
        seed=seed + 30_000,
        rate_per_min=cfg.long_artifact_rate_per_min,
        duration_s=cfg.long_artifact_duration_s,
        amplitude=cfg.long_artifact_amplitude,
        min_start_s=cfg.baseline_end_s,
    )
    windows = window_signal(
        t_s=lfp.t_s,
        signal=signal,
        window_s=window_s,
        step_s=step_s,
        positive_interval_s=None,
        record_id=f"long_no_runaway_seed_{seed}",
        record_type="long_no_runaway",
        seed=seed,
    )
    scores = score_estimator(estimator, windows.x)
    return LongRecordScores(
        seed=seed,
        duration_s=cfg.t_total_s,
        t_s=lfp.t_s,
        signal=signal,
        artifact_times_s=artifact_times,
        windows=windows,
        scores=scores,
    )


def threshold_grid(selected_threshold: float) -> list[float]:
    values = list(np.linspace(0.0, 1.0, 51))
    values.append(float(selected_threshold))
    values = sorted({round(float(v), 6) for v in values})
    return values


def long_false_alarm_rows(
    seeds: list[int],
    base_cfg: Config,
    estimator,
    selected_threshold: float,
    window_s: float,
    step_s: float,
    quick: bool = False,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], LongRecordScores]:
    """Run long no-runaway records and summarize FA/h across thresholds."""
    cfg = long_no_runaway_config(base_cfg, quick=quick)
    use_seeds = seeds[: max(1, min(len(seeds), cfg.long_baseline_seed_count))]
    thresholds = threshold_grid(selected_threshold)

    per_record_rows: list[dict[str, float | int]] = []
    all_scores: list[LongRecordScores] = []
    for seed in use_seeds:
        record = run_long_record_scores(seed, cfg, estimator, window_s=window_s, step_s=step_s)
        all_scores.append(record)
        for threshold in thresholds:
            row = score_false_alarms_per_hour(record.windows.centers_s, record.scores, threshold, record.duration_s)
            per_record_rows.append(
                {
                    "seed": seed,
                    "duration_s": float(record.duration_s),
                    "artifact_count": int(len(record.artifact_times_s)),
                    **row,
                }
            )

    summary_rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        subset = [row for row in per_record_rows if float(row["threshold"]) == float(threshold)]
        far = np.asarray([float(row["false_alarms_per_hour"]) for row in subset], dtype=float)
        events = np.asarray([float(row["false_alarm_events"]) for row in subset], dtype=float)
        positive_windows = np.asarray([float(row["positive_windows"]) for row in subset], dtype=float)
        summary_rows.append(
            {
                "threshold": float(threshold),
                "mean_false_alarms_per_hour": float(np.mean(far)),
                "std_false_alarms_per_hour": float(np.std(far, ddof=1)) if far.size > 1 else 0.0,
                "mean_false_alarm_events": float(np.mean(events)),
                "mean_positive_windows": float(np.mean(positive_windows)),
                "n_records": int(len(subset)),
            }
        )

    return per_record_rows, summary_rows, all_scores[0]


def long_prediction_rows(record: LongRecordScores, threshold: float) -> list[dict[str, float | int | str]]:
    """Serialize representative long-record ML scores."""
    predictions = (record.scores >= threshold).astype(int)
    rows: list[dict[str, float | int | str]] = []
    for idx in range(record.scores.size):
        rows.append(
            {
                "record_id": record.windows.record_ids[idx],
                "seed": int(record.seed),
                "center_s": float(record.windows.centers_s[idx]),
                "start_s": float(record.windows.starts_s[idx]),
                "end_s": float(record.windows.ends_s[idx]),
                "score": float(record.scores[idx]),
                "predicted_false_alarm_window": int(predictions[idx]),
            }
        )
    return rows


def operating_threshold_rows(y_true: np.ndarray, scores: np.ndarray, long_summary_rows: list[dict]) -> list[dict]:
    """Combine holdout classification metrics with long-record FA/h by threshold."""
    rows: list[dict] = []
    for long_row in long_summary_rows:
        threshold = float(long_row["threshold"])
        metrics = binary_metrics(y_true, scores, threshold)
        rows.append(
            {
                "threshold": threshold,
                "test_sensitivity": metrics.sensitivity,
                "test_specificity": metrics.specificity,
                "test_precision": metrics.precision,
                "test_f1": metrics.f1,
                "test_accuracy": metrics.accuracy,
                "test_tp": metrics.tp,
                "test_fp": metrics.fp,
                "test_fn": metrics.fn,
                "test_tn": metrics.tn,
                "mean_false_alarms_per_hour": float(long_row["mean_false_alarms_per_hour"]),
                "std_false_alarms_per_hour": float(long_row["std_false_alarms_per_hour"]),
                "mean_false_alarm_events": float(long_row["mean_false_alarm_events"]),
                "n_long_records": int(long_row["n_records"]),
            }
        )
    return rows


def choose_fa_constrained_threshold(rows: list[dict], target_fah: float) -> dict:
    """Choose the highest-performing threshold under a false-alarm/hour cap."""
    feasible = [row for row in rows if float(row["mean_false_alarms_per_hour"]) <= float(target_fah)]
    if not feasible:
        return min(rows, key=lambda row: float(row["mean_false_alarms_per_hour"]))
    return max(
        feasible,
        key=lambda row: (
            float(row["test_f1"]),
            float(row["test_sensitivity"]),
            -float(row["threshold"]),
        ),
    )
