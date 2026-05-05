#!/usr/bin/env python3
"""Run the clean final-project experiment suite."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runaway_control.config import Config, parse_seed_spec
from runaway_control.experiments import (
    aggregate_rows,
    as_jsonable,
    detector_benchmark_rows,
    detector_benchmark_summary,
    feature_benchmark_rows,
    feature_benchmark_summary,
    long_baseline_false_alarm_rows,
    long_baseline_stress_row,
    run_trial,
    roc_auc,
    scenario_label,
    threshold_sweep_rows,
    trial_summary_row,
)
from runaway_control.lfp import compute_lfp
from runaway_control.metrics import compute_suppression_metrics
from runaway_control.network import run_network
from runaway_control.plotting import (
    fig_block_diagram,
    fig_closed_loop,
    fig_detector_performance,
    fig_detector_trace,
    fig_feature_comparison,
    fig_latency_sensitivity,
    fig_long_fa_vs_k,
    fig_long_record_false_alarms,
    fig_runaway_example,
)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def write_stress_markdown(path: Path, rows: list[dict]):
    headers = [
        "Scenario",
        "Sensitivity",
        "False alarms/hour",
        "Latency (ms)",
        "LFP suppression (%)",
        "E-rate suppression (%)",
        "Duration reduction (%)",
        "Pulses",
        "Duty cycle",
        "Burden (nS*s)",
    ]
    keys = [
        "scenario",
        "mean_sensitivity",
        "mean_false_alarms_per_hour",
        "mean_latency_ms",
        "mean_lfp_power_suppression_pct",
        "mean_e_rate_suppression_pct",
        "mean_runaway_duration_reduction_pct",
        "mean_n_pulses",
        "mean_duty_cycle",
        "mean_burden_nS_s",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(k, "")) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n")


def run_latency_sensitivity(rep_trial, cfg: Config) -> list[dict]:
    """Single-seed latency sweep (kept for backwards compatibility)."""
    return _latency_sweep_for_trial(rep_trial, cfg)


def _latency_sweep_for_trial(trial, cfg: Config) -> list[dict]:
    from runaway_control.controller import filter_stim_times

    rows: list[dict] = []
    for latency_ms in cfg.latency_sweep_ms:
        latency_s = latency_ms / 1000.0
        schedule = filter_stim_times(trial.detection.detection_times_s, cfg, added_latency_s=latency_s)
        closed_run = run_network(cfg, seed=trial.seed, stim_times=schedule.safe_times_s)
        closed_lfp = compute_lfp(closed_run.monitors["state"], cfg)
        metrics = compute_suppression_metrics(trial.open_run, closed_run, trial.open_lfp, closed_lfp, schedule)
        rows.append(
            {
                "seed": trial.seed,
                "added_latency_ms": latency_ms,
                "lfp_power_suppression_pct": metrics.lfp_power_suppression_pct,
                "e_rate_suppression_pct": metrics.e_rate_suppression_pct,
                "runaway_duration_reduction_pct": metrics.runaway_duration_reduction_pct,
                "n_pulses": metrics.n_pulses,
                "duty_cycle": metrics.duty_cycle,
                "burden_nS_s": metrics.burden_nS_s,
            }
        )
    return rows


def run_latency_sensitivity_across_seeds(baseline_trials: list, cfg: Config) -> list[dict]:
    """Run the latency sweep on every supplied baseline trial; one row per (seed, latency)."""
    rows: list[dict] = []
    for trial in baseline_trials:
        rows.extend(_latency_sweep_for_trial(trial, cfg))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results" / "core_run"), help="output directory")
    parser.add_argument("--seeds", default="42-51", help="seed spec, e.g. 42-51 or 42,43")
    parser.add_argument("--quick", action="store_true", help="run 2-seed smoke version")
    parser.add_argument("--report", action="store_true", help="show Brian2 progress reports")
    args = parser.parse_args(argv)

    cfg = Config()
    seeds = parse_seed_spec(args.seeds)
    if args.quick:
        seeds = seeds[:2] if len(seeds) >= 2 else seeds
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    tables = out / "tables"

    open_cache = {}
    rep_seed = seeds[0]

    print(f"[core] Running representative baseline trial seed={rep_seed}")
    rep_trial = run_trial(rep_seed, cfg, scenario="baseline", k=cfg.default_k, open_cache=open_cache)

    print("[core] Running threshold sweep")
    threshold_rows = threshold_sweep_rows(seeds, cfg, open_cache)
    threshold_summary = aggregate_rows(threshold_rows, "k")

    print("[core] Running mixed-record detector benchmark")
    detector_rows = detector_benchmark_rows(seeds, cfg, open_cache)
    detector_summary = detector_benchmark_summary(detector_rows)
    detector_auc = roc_auc(detector_summary)

    print("[core] Running feature comparison benchmark")
    feature_rows = feature_benchmark_rows(seeds, cfg, open_cache)
    feature_summary = feature_benchmark_summary(feature_rows)

    print("[core] Running stress scenarios")
    trial_rows = [trial_summary_row(rep_trial)]
    scenario_trials = [rep_trial]
    baseline_trials = [rep_trial]
    scenarios = ["baseline", "noisy_lfp", "parameter_shifted"]
    for scenario in scenarios:
        for seed in seeds:
            if scenario == "baseline" and seed == rep_seed:
                continue
            trial = run_trial(seed, cfg, scenario=scenario, k=cfg.default_k, open_cache=open_cache)
            scenario_trials.append(trial)
            trial_rows.append(trial_summary_row(trial))
            if scenario == "baseline":
                baseline_trials.append(trial)
    stress_summary = aggregate_rows(trial_rows, "scenario")

    print("[core] Running long no-runaway false-alarm validation")
    long_rows, long_rep = long_baseline_false_alarm_rows(seeds, cfg, quick=bool(args.quick))
    long_summary = aggregate_rows(long_rows, "k")
    stress_summary.append(long_baseline_stress_row(long_rows, cfg.default_k))

    print(f"[core] Running latency sensitivity across {len(baseline_trials)} baseline seeds")
    latency_rows = run_latency_sensitivity_across_seeds(baseline_trials, cfg)
    latency_summary = aggregate_rows(latency_rows, "added_latency_ms")

    print("[core] Writing tables")
    write_csv(tables / "threshold_sweep.csv", threshold_rows)
    write_csv(tables / "threshold_summary.csv", threshold_summary)
    write_csv(tables / "detector_benchmark.csv", detector_rows)
    write_csv(tables / "detector_benchmark_summary.csv", detector_summary)
    write_csv(tables / "feature_benchmark.csv", feature_rows)
    write_csv(tables / "feature_benchmark_summary.csv", feature_summary)
    write_csv(tables / "trial_summary.csv", trial_rows)
    write_csv(tables / "long_baseline_false_alarms.csv", long_rows)
    write_csv(tables / "long_baseline_fa_by_k.csv", long_summary)
    write_csv(tables / "table1_stress_summary.csv", stress_summary)
    write_csv(tables / "latency_sensitivity.csv", latency_rows)
    write_csv(tables / "latency_sensitivity_summary.csv", latency_summary)
    write_stress_markdown(tables / "table1_stress_summary.md", stress_summary)

    print("[core] Writing figures")
    fig_block_diagram(figures / "fig1_block_diagram.png")
    fig_runaway_example(figures / "fig2_runaway_example.png", rep_trial.open_run, rep_trial.open_lfp)
    fig_detector_trace(figures / "fig3_detector_trace.png", rep_trial.open_lfp, rep_trial.detection_signal, rep_trial.detection, cfg)
    fig_detector_performance(figures / "fig4_detector_performance.png", detector_summary, detector_auc)
    fig_feature_comparison(figures / "fig4b_feature_comparison.png", feature_summary)
    fig_closed_loop(figures / "fig5_closed_loop_suppression.png", rep_trial, baseline_trials=baseline_trials)
    fig_latency_sensitivity(figures / "fig6_latency_sensitivity.png", latency_rows)
    fig_long_record_false_alarms(figures / "fig7_long_record_fa.png", long_rep)
    fig_long_fa_vs_k(figures / "fig8_long_fa_vs_k.png", long_summary)

    summary = {
        "quick": bool(args.quick),
        "seeds": seeds,
        "representative_seed": rep_seed,
        "config": as_jsonable(cfg),
        "required_outputs": {
            "figures": [str(p.relative_to(out)) for p in sorted(figures.glob("*.png"))],
            "stress_table_csv": str((tables / "table1_stress_summary.csv").relative_to(out)),
            "stress_table_md": str((tables / "table1_stress_summary.md").relative_to(out)),
        },
        "representative_metrics": as_jsonable(rep_trial.metrics),
        "representative_detection": as_jsonable(rep_trial.score),
        "mixed_record_detector_auc": detector_auc,
        "feature_comparison_auc": {
            str(row["feature_kind"]): row["auc"]
            for row in feature_summary
            if float(row["k"]) == float(cfg.roc_threshold_values[0])
        },
        "long_baseline_default_k": {
            "duration_s": long_rep.cfg.t_total_s,
            "seed_count": len({int(r["seed"]) for r in long_rows}),
            "default_k": cfg.default_k,
            "representative": {
                "seed": long_rep.seed,
                "false_alarms_per_hour": long_rep.score.false_alarms_per_hour,
                "fp_events": long_rep.score.fp_events,
                "artifact_count": len(long_rep.artifact_times_s),
                "hypothetical_pulses": long_rep.schedule.n_pulses,
                "hypothetical_burden_nS_s": long_rep.schedule.burden_nS_s,
            },
            "stress_table_row": stress_summary[-1],
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[core] Done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
