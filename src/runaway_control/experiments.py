"""Experiment orchestration helpers for the clean core run."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import Config
from .controller import StimSchedule, filter_stim_times
from .detector import DetectionResult, detect_feature, detect_line_length
from .lfp import LfpResult, add_artifact_burst, add_artifact_bursts, add_gaussian_noise, compute_lfp
from .metrics import DetectionScore, SuppressionMetrics, compute_suppression_metrics, score_detections, score_false_alarms
from .network import NetworkRun, run_network


@dataclass
class TrialBundle:
    seed: int
    scenario: str
    cfg: Config
    open_run: NetworkRun
    closed_run: NetworkRun
    open_lfp: LfpResult
    closed_lfp: LfpResult
    detection_signal: np.ndarray
    detection: DetectionResult
    score: DetectionScore
    schedule: StimSchedule
    metrics: SuppressionMetrics


@dataclass
class LongBaselineBundle:
    seed: int
    cfg: Config
    lfp: LfpResult
    signal: np.ndarray
    detection: DetectionResult
    score: DetectionScore
    schedule: StimSchedule
    artifact_times_s: list[float]


def scenario_config(base_cfg: Config, scenario: str) -> Config:
    if scenario == "parameter_shifted":
        return base_cfg.parameter_shifted()
    if scenario in {"baseline", "noisy_lfp"}:
        return base_cfg
    raise ValueError(f"unknown scenario: {scenario}")


def scenario_label(scenario: str) -> str:
    return {
        "baseline": "Baseline",
        "noisy_lfp": "Noisy LFP 10 dB",
        "parameter_shifted": "Parameter shifted",
    }[scenario]


def long_baseline_config(base_cfg: Config, quick: bool = False) -> Config:
    duration_s = base_cfg.quick_long_baseline_duration_s if quick else base_cfg.long_baseline_duration_s
    return base_cfg.with_updates(
        t_total_s=duration_s,
        inhibitory_weight_factor=1.0,
    )


def detection_signal_for_scenario(lfp: LfpResult, scenario: str, seed: int) -> np.ndarray:
    if scenario == "noisy_lfp":
        return add_gaussian_noise(lfp.lfp_z, snr_db=10.0, seed=seed + 10_000)
    return lfp.lfp_z.copy()


def run_trial(
    seed: int,
    base_cfg: Config,
    scenario: str = "baseline",
    k: float | None = None,
    added_latency_s: float = 0.0,
    open_cache: dict[tuple[str, int], tuple[NetworkRun, LfpResult]] | None = None,
) -> TrialBundle:
    cfg = scenario_config(base_cfg, scenario)
    cache_key = (scenario, seed)
    if open_cache is not None and cache_key in open_cache:
        open_run, open_lfp = open_cache[cache_key]
    else:
        open_run = run_network(cfg, seed=seed)
        open_lfp = compute_lfp(open_run.monitors["state"], cfg)
        if open_cache is not None:
            open_cache[cache_key] = (open_run, open_lfp)

    signal = detection_signal_for_scenario(open_lfp, scenario, seed)
    detection = detect_line_length(open_lfp.t_s, signal, cfg, k=k)
    score = score_detections(detection.detection_times_s, cfg)
    schedule = filter_stim_times(detection.detection_times_s, cfg, added_latency_s=added_latency_s)
    closed_run = run_network(cfg, seed=seed, stim_times=schedule.safe_times_s)
    closed_lfp = compute_lfp(closed_run.monitors["state"], cfg)
    metrics = compute_suppression_metrics(open_run, closed_run, open_lfp, closed_lfp, schedule)

    return TrialBundle(
        seed=seed,
        scenario=scenario,
        cfg=cfg,
        open_run=open_run,
        closed_run=closed_run,
        open_lfp=open_lfp,
        closed_lfp=closed_lfp,
        detection_signal=signal,
        detection=detection,
        score=score,
        schedule=schedule,
        metrics=metrics,
    )


def trial_summary_row(trial: TrialBundle) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "seed": trial.seed,
        "scenario": scenario_label(trial.scenario),
        "k": trial.detection.k,
        "sensitivity": trial.score.sensitivity,
        "false_alarms_per_hour": trial.score.false_alarms_per_hour,
        "latency_ms": trial.score.latency_ms,
        "lfp_power_suppression_pct": trial.metrics.lfp_power_suppression_pct,
        "e_rate_suppression_pct": trial.metrics.e_rate_suppression_pct,
        "runaway_duration_reduction_pct": trial.metrics.runaway_duration_reduction_pct,
        "n_pulses": trial.metrics.n_pulses,
        "duty_cycle": trial.metrics.duty_cycle,
        "burden_nS_s": trial.metrics.burden_nS_s,
        "open_eval_rate_hz": trial.metrics.open_eval_rate_hz,
        "closed_eval_rate_hz": trial.metrics.closed_eval_rate_hz,
    }
    return row


def threshold_sweep_rows(
    seeds: list[int],
    cfg: Config,
    open_cache: dict[tuple[str, int], tuple[NetworkRun, LfpResult]],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for seed in seeds:
        if ("baseline", seed) in open_cache:
            open_run, open_lfp = open_cache[("baseline", seed)]
        else:
            open_run = run_network(cfg, seed=seed)
            open_lfp = compute_lfp(open_run.monitors["state"], cfg)
            open_cache[("baseline", seed)] = (open_run, open_lfp)
        for k in cfg.threshold_values:
            det = detect_line_length(open_lfp.t_s, open_lfp.lfp_z, cfg, k=k)
            score = score_detections(det.detection_times_s, cfg)
            rows.append(
                {
                    "seed": seed,
                    "k": k,
                    "sensitivity": score.sensitivity,
                    "false_alarms_per_hour": score.false_alarms_per_hour,
                    "latency_ms": score.latency_ms,
                    "n_events": score.n_events,
                    "fp_events": score.fp_events,
                    "tp_events": score.tp_events,
                }
            )
    return rows


def _get_or_run_open(
    cfg: Config,
    seed: int,
    cache_key: tuple[str, int],
    open_cache: dict[tuple[str, int], tuple[NetworkRun, LfpResult]],
) -> tuple[NetworkRun, LfpResult]:
    if cache_key in open_cache:
        return open_cache[cache_key]
    run = run_network(cfg, seed=seed)
    lfp = compute_lfp(run.monitors["state"], cfg)
    open_cache[cache_key] = (run, lfp)
    return run, lfp


def _event_times_after_merge(times: list[float], tolerance_s: float = 0.30) -> list[tuple[float, float]]:
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


def _detector_benchmark_specs(base_cfg: Config):
    return [
        ("strong_runaway", True, base_cfg, "clean"),
        ("weak_runaway", True, base_cfg.with_updates(inhibitory_weight_factor=0.50), "clean"),
        ("no_runaway_clean", False, base_cfg.with_updates(inhibitory_weight_factor=1.00), "clean"),
        ("no_runaway_artifact", False, base_cfg.with_updates(inhibitory_weight_factor=1.00), "artifact"),
    ]


def _score_benchmark_detection(det: DetectionResult, has_runaway: bool, cfg: Config) -> dict[str, float | int]:
    events = _event_times_after_merge(det.detection_times_s)
    if has_runaway:
        tp = int(any(e[1] >= cfg.runaway_onset_s for e in events))
        fp_events = [e for e in events if e[1] < cfg.runaway_onset_s]
        latency_ms = float("nan")
        post_events = [e for e in events if e[1] >= cfg.runaway_onset_s]
        if post_events:
            latency_ms = max(0.0, post_events[0][0] - cfg.runaway_onset_s) * 1000.0
        fp_record = int(bool(fp_events))
        false_event_count = len(fp_events)
        non_event_hours = cfg.baseline_end_s / 3600.0
    else:
        tp = 0
        fp_record = int(bool(events))
        latency_ms = float("nan")
        false_event_count = len(events)
        non_event_hours = cfg.t_total_s / 3600.0
    return {
        "tp_record": tp,
        "fp_record": fp_record,
        "false_event_count": false_event_count,
        "non_event_hours": non_event_hours,
        "latency_ms": latency_ms,
        "n_events": len(events),
    }


def detector_benchmark_rows(
    seeds: list[int],
    base_cfg: Config,
    open_cache: dict[tuple[str, int], tuple[NetworkRun, LfpResult]],
) -> list[dict[str, float | int | str]]:
    """Mixed positive/negative detector benchmark for ROC/AUC-style Fig. 4."""
    rows: list[dict[str, float | int | str]] = []
    for name, has_runaway, cfg, signal_kind in _detector_benchmark_specs(base_cfg):
        for seed in seeds:
            cache_key = ("baseline", seed) if name == "strong_runaway" else (name, seed)
            _, lfp = _get_or_run_open(cfg, seed, cache_key, open_cache)
            signal = lfp.lfp_z.copy()
            if signal_kind == "artifact":
                signal = add_artifact_burst(lfp.t_s, signal, seed=seed + 20_000, amplitude=1.0)

            for k in base_cfg.roc_threshold_values:
                det = detect_line_length(lfp.t_s, signal, cfg, k=k)
                score = _score_benchmark_detection(det, has_runaway, cfg)
                rows.append(
                    {
                        "record_type": name,
                        "seed": seed,
                        "has_runaway": int(has_runaway),
                        "k": float(k),
                        **score,
                    }
                )
    return rows


def detector_benchmark_summary(rows: list[dict[str, float | int | str]]) -> list[dict[str, float]]:
    """Aggregate mixed-record benchmark by threshold."""
    k_values = sorted({float(r["k"]) for r in rows})
    out: list[dict[str, float]] = []
    for k in k_values:
        subset = [r for r in rows if float(r["k"]) == k]
        positives = [r for r in subset if int(r["has_runaway"]) == 1]
        negatives = [r for r in subset if int(r["has_runaway"]) == 0]
        tp = sum(int(r["tp_record"]) for r in positives)
        fn = len(positives) - tp
        fp = sum(int(r["fp_record"]) for r in negatives)
        tn = len(negatives) - fp
        false_events = sum(int(r["false_event_count"]) for r in subset)
        non_event_hours = sum(float(r["non_event_hours"]) for r in subset)
        latencies = np.asarray([float(r["latency_ms"]) for r in positives], dtype=float)
        latencies = latencies[np.isfinite(latencies)]
        sensitivity = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        specificity = tn / max(tn + fp, 1)
        out.append(
            {
                "k": k,
                "sensitivity": float(sensitivity),
                "false_positive_rate": float(fpr),
                "specificity": float(specificity),
                "precision": float(precision),
                "false_alarms_per_hour": float(false_events / max(non_event_hours, 1e-12)),
                "mean_latency_ms": float(np.mean(latencies)) if latencies.size else float("nan"),
                "tp_records": float(tp),
                "fp_records": float(fp),
                "tn_records": float(tn),
                "fn_records": float(fn),
                "n_positive_records": float(len(positives)),
                "n_negative_records": float(len(negatives)),
            }
        )
    return out


def feature_benchmark_rows(
    seeds: list[int],
    base_cfg: Config,
    open_cache: dict[tuple[str, int], tuple[NetworkRun, LfpResult]],
) -> list[dict[str, float | int | str]]:
    """Mixed-record benchmark repeated for each detector feature kind."""
    rows: list[dict[str, float | int | str]] = []
    for feature_kind in base_cfg.feature_kinds:
        for name, has_runaway, cfg, signal_kind in _detector_benchmark_specs(base_cfg):
            cfg = cfg.with_updates(feature_kind=feature_kind)
            for seed in seeds:
                cache_key = ("baseline", seed) if name == "strong_runaway" else (name, seed)
                _, lfp = _get_or_run_open(cfg, seed, cache_key, open_cache)
                signal = lfp.lfp_z.copy()
                if signal_kind == "artifact":
                    signal = add_artifact_burst(lfp.t_s, signal, seed=seed + 20_000, amplitude=1.0)

                for k in base_cfg.roc_threshold_values:
                    det = detect_feature(lfp.t_s, signal, cfg, k=k, feature_kind=feature_kind)
                    score = _score_benchmark_detection(det, has_runaway, cfg)
                    rows.append(
                        {
                            "feature_kind": feature_kind,
                            "record_type": name,
                            "seed": seed,
                            "has_runaway": int(has_runaway),
                            "k": float(k),
                            **score,
                        }
                    )
    return rows


def feature_benchmark_summary(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    """Aggregate mixed-record benchmark by feature and threshold."""
    out: list[dict[str, float | str]] = []
    feature_kinds = sorted({str(r["feature_kind"]) for r in rows})
    for feature_kind in feature_kinds:
        subset = [r for r in rows if str(r["feature_kind"]) == feature_kind]
        summary = detector_benchmark_summary(subset)
        auc = roc_auc(summary)
        for row in summary:
            out.append({"feature_kind": feature_kind, "auc": auc, **row})
    return out


def long_baseline_false_alarm_rows(
    seeds: list[int],
    base_cfg: Config,
    quick: bool = False,
) -> tuple[list[dict[str, float | int]], LongBaselineBundle]:
    """Run no-runaway artifact-corrupted records and score false detections."""
    cfg = long_baseline_config(base_cfg, quick=quick)
    use_seeds = seeds[: max(1, min(len(seeds), cfg.long_baseline_seed_count))]
    rows: list[dict[str, float | int]] = []
    representative: LongBaselineBundle | None = None
    for seed in use_seeds:
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
        for k in cfg.roc_threshold_values:
            det = detect_line_length(lfp.t_s, signal, cfg, k=k)
            score = score_false_alarms(det.detection_times_s, cfg.t_total_s)
            schedule = filter_stim_times(det.detection_times_s, cfg)
            rows.append(
                {
                    "seed": seed,
                    "k": float(k),
                    "false_alarms_per_hour": score.false_alarms_per_hour,
                    "fp_events": score.fp_events,
                    "n_events": score.n_events,
                    "artifact_count": len(artifact_times),
                    "n_pulses": schedule.n_pulses,
                    "duty_cycle": schedule.duty_cycle,
                    "burden_nS_s": schedule.burden_nS_s,
                    "duration_s": cfg.t_total_s,
                }
            )
            if representative is None and float(k) == cfg.default_k:
                representative = LongBaselineBundle(
                    seed=seed,
                    cfg=cfg,
                    lfp=lfp,
                    signal=signal,
                    detection=det,
                    score=score,
                    schedule=schedule,
                    artifact_times_s=artifact_times,
                )
    if representative is None:
        raise RuntimeError("long-baseline validation did not produce a representative record")
    return rows, representative


def long_baseline_stress_row(rows: list[dict[str, float | int]], default_k: float) -> dict[str, float | str]:
    """Convert long no-runaway rows into the stress-table schema."""
    subset = [r for r in rows if float(r["k"]) == float(default_k)]
    far = np.asarray([float(r["false_alarms_per_hour"]) for r in subset], dtype=float)
    pulses = np.asarray([float(r["n_pulses"]) for r in subset], dtype=float)
    duty = np.asarray([float(r["duty_cycle"]) for r in subset], dtype=float)
    burden = np.asarray([float(r["burden_nS_s"]) for r in subset], dtype=float)
    artifact_count = np.asarray([float(r["artifact_count"]) for r in subset], dtype=float)
    return {
        "scenario": "Long no-runaway",
        "mean_sensitivity": float("nan"),
        "std_sensitivity": float("nan"),
        "mean_false_alarms_per_hour": float(np.mean(far)) if far.size else float("nan"),
        "std_false_alarms_per_hour": float(np.std(far, ddof=1)) if far.size > 1 else 0.0,
        "mean_latency_ms": float("nan"),
        "std_latency_ms": float("nan"),
        "mean_lfp_power_suppression_pct": float("nan"),
        "std_lfp_power_suppression_pct": float("nan"),
        "mean_e_rate_suppression_pct": float("nan"),
        "std_e_rate_suppression_pct": float("nan"),
        "mean_runaway_duration_reduction_pct": float("nan"),
        "std_runaway_duration_reduction_pct": float("nan"),
        "mean_n_pulses": float(np.mean(pulses)) if pulses.size else float("nan"),
        "std_n_pulses": float(np.std(pulses, ddof=1)) if pulses.size > 1 else 0.0,
        "mean_duty_cycle": float(np.mean(duty)) if duty.size else float("nan"),
        "std_duty_cycle": float(np.std(duty, ddof=1)) if duty.size > 1 else 0.0,
        "mean_burden_nS_s": float(np.mean(burden)) if burden.size else float("nan"),
        "std_burden_nS_s": float(np.std(burden, ddof=1)) if burden.size > 1 else 0.0,
        "mean_artifact_count": float(np.mean(artifact_count)) if artifact_count.size else float("nan"),
        "n": len(subset),
    }


def roc_auc(summary_rows: list[dict[str, float]]) -> float:
    points = [(float(r["false_positive_rate"]), float(r["sensitivity"])) for r in summary_rows]
    points.extend([(0.0, 0.0), (1.0, 1.0)])
    by_fpr: dict[float, float] = {}
    for x, y in points:
        by_fpr[x] = max(by_fpr.get(x, 0.0), y)
    xs = np.asarray(sorted(by_fpr), dtype=float)
    ys = np.asarray([by_fpr[x] for x in xs], dtype=float)
    return float(np.trapezoid(ys, xs))


def aggregate_rows(rows: list[dict], group_key: str) -> list[dict[str, float | str]]:
    groups: dict[object, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    out: list[dict[str, float | str]] = []
    for key, vals in groups.items():
        summary: dict[str, float | str] = {group_key: key}
        numeric_keys = [k for k, v in vals[0].items() if isinstance(v, (int, float)) and k != group_key]
        for name in numeric_keys:
            arr = np.asarray([float(v[name]) for v in vals], dtype=float)
            finite = arr[np.isfinite(arr)]
            summary[f"mean_{name}"] = float(np.mean(finite)) if finite.size else float("nan")
            summary[f"std_{name}"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        summary["n"] = len(vals)
        out.append(summary)
    return out


def as_jsonable(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: as_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
