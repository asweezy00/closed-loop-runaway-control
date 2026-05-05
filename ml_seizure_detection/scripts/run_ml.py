#!/usr/bin/env python3
"""Train and evaluate the ML seizure-like event detector."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_PROJECT_ROOT = ROOT.parent
SRC = ROOT / "src"
FINAL_SRC = FINAL_PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
for path in (SRC, FINAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from runaway_control.config import Config, parse_seed_spec

from ml_seizure_detection.dataset import build_simulated_window_dataset, dataset_rows, quick_ml_config
from ml_seizure_detection.long_validation import (
    choose_fa_constrained_threshold,
    long_false_alarm_rows,
    long_prediction_rows,
    operating_threshold_rows,
)
from ml_seizure_detection.metrics import binary_metrics
from ml_seizure_detection.models import (
    candidate_models,
    feature_importance_rows,
    fit_best_model,
    prediction_rows,
    score_estimator,
)
from ml_seizure_detection.plotting import (
    fig_confusion_matrix,
    fig_feature_importance,
    fig_long_fa_vs_threshold,
    fig_long_record_scores,
    fig_operating_tradeoff,
    fig_precision_recall,
    fig_roc,
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


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def build_config(quick: bool) -> Config:
    if quick:
        return quick_ml_config()
    return Config()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/ml_run", help="output directory")
    parser.add_argument("--seeds", default="42-51", help="seed spec such as 42-51 or 42,43")
    parser.add_argument("--quick", action="store_true", help="faster smoke-test run")
    parser.add_argument("--window-s", type=float, default=0.500)
    parser.add_argument("--step-s", type=float, default=0.050)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--target-fah", type=float, default=60.0, help="target long-record false alarms/hour")
    args = parser.parse_args()

    out = Path(args.out)
    fig_dir = out / "figures"
    table_dir = out / "tables"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args.quick)
    seeds = parse_seed_spec(args.seeds)
    if args.quick:
        seeds = seeds[:4]

    print(f"Building ML window dataset from {len(seeds)} seeds...")
    dataset = build_simulated_window_dataset(seeds, cfg, window_s=args.window_s, step_s=args.step_s)
    write_csv(table_dir / "window_dataset.csv", dataset_rows(dataset))

    print("Training candidate models and selecting validation ROC-AUC winner...")
    fitted = fit_best_model(dataset, random_state=args.random_state)
    write_csv(table_dir / "model_comparison.csv", fitted.model_rows)
    write_csv(table_dir / "test_predictions.csv", prediction_rows(dataset, fitted))
    write_csv(table_dir / "test_metrics.csv", [fitted.metrics.as_row()])

    print("Computing feature importance...")
    importance = feature_importance_rows(dataset, fitted, random_state=args.random_state)
    write_csv(table_dir / "feature_importance.csv", importance)

    print("Running long no-runaway false-alarm validation...")
    long_rows, long_summary_rows, representative_long = long_false_alarm_rows(
        seeds=seeds,
        base_cfg=cfg,
        estimator=fitted.estimator,
        selected_threshold=fitted.threshold,
        window_s=args.window_s,
        step_s=args.step_s,
        quick=args.quick,
    )
    write_csv(table_dir / "long_record_false_alarms.csv", long_rows)
    write_csv(table_dir / "long_record_fa_by_threshold.csv", long_summary_rows)
    write_csv(table_dir / "long_record_predictions.csv", long_prediction_rows(representative_long, fitted.threshold))

    y_test = dataset.y[fitted.split.test_idx]
    operating_rows = operating_threshold_rows(y_test, fitted.test_scores, long_summary_rows)
    write_csv(table_dir / "operating_thresholds.csv", operating_rows)
    fa_constrained = choose_fa_constrained_threshold(operating_rows, target_fah=args.target_fah)

    fig_roc(fig_dir / "fig1_roc_curve.png", y_test, fitted.test_scores_by_model, fitted.name)
    fig_precision_recall(fig_dir / "fig2_precision_recall.png", y_test, fitted.test_scores_by_model, fitted.name)
    fig_confusion_matrix(
        fig_dir / "fig3_confusion_matrix.png",
        fitted.metrics.tn,
        fitted.metrics.fp,
        fitted.metrics.fn,
        fitted.metrics.tp,
        title=f"{fitted.name.replace('_', ' ').title()} Confusion Matrix",
    )
    fig_feature_importance(fig_dir / "fig4_feature_importance.png", importance)
    fig_long_fa_vs_threshold(fig_dir / "fig5_long_fa_vs_threshold.png", long_summary_rows, fitted.threshold)
    fig_long_record_scores(
        fig_dir / "fig6_long_record_scores.png",
        representative_long.t_s,
        representative_long.signal,
        representative_long.windows.centers_s,
        representative_long.scores,
        fitted.threshold,
        representative_long.artifact_times_s,
    )
    fig_operating_tradeoff(fig_dir / "fig7_operating_tradeoff.png", operating_rows, target_fah=args.target_fah)

    selected_long_rows = [row for row in long_summary_rows if abs(float(row["threshold"]) - fitted.threshold) < 1e-6]
    selected_long = selected_long_rows[0] if selected_long_rows else min(
        long_summary_rows,
        key=lambda row: abs(float(row["threshold"]) - fitted.threshold),
    )

    summary = {
        "selected_model": fitted.name,
        "selected_threshold": fitted.threshold,
        "test_metrics": fitted.metrics.as_row(),
        "dataset": {
            "n_windows": int(dataset.y.size),
            "n_positive_windows": int(np.sum(dataset.y == 1)),
            "n_negative_windows": int(np.sum(dataset.y == 0)),
            "n_records": int(len(set(dataset.record_ids))),
            "seeds": seeds,
            "quick": bool(args.quick),
            "window_s": args.window_s,
            "step_s": args.step_s,
        },
        "model_comparison": fitted.model_rows,
        "long_no_runaway_false_alarm_validation": {
            "threshold": float(selected_long["threshold"]),
            "mean_false_alarms_per_hour": float(selected_long["mean_false_alarms_per_hour"]),
            "std_false_alarms_per_hour": float(selected_long["std_false_alarms_per_hour"]),
            "mean_false_alarm_events": float(selected_long["mean_false_alarm_events"]),
            "n_records": int(selected_long["n_records"]),
        },
        "fa_constrained_operating_point": {
            "target_false_alarms_per_hour": float(args.target_fah),
            **fa_constrained,
        },
        "required_outputs": {
            "figures": sorted(str(p.relative_to(out)) for p in fig_dir.glob("*.png")),
            "tables": sorted(str(p.relative_to(out)) for p in table_dir.glob("*.csv")),
        },
    }
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))

    metrics = fitted.metrics
    print("\nML seizure-like detector complete")
    print(f"Selected model: {fitted.name}")
    print(f"ROC-AUC: {metrics.roc_auc:.3f}")
    print(f"PR-AUC: {metrics.pr_auc:.3f}")
    print(f"Sensitivity: {metrics.sensitivity:.3f}")
    print(f"Specificity: {metrics.specificity:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"F1: {metrics.f1:.3f}")
    print(
        "Long no-runaway FA/h: "
        f"{float(selected_long['mean_false_alarms_per_hour']):.1f} "
        f"+/- {float(selected_long['std_false_alarms_per_hour']):.1f}"
    )
    print(
        "FA-constrained operating point: "
        f"threshold={float(fa_constrained['threshold']):.3f}, "
        f"FA/h={float(fa_constrained['mean_false_alarms_per_hour']):.1f}, "
        f"sensitivity={float(fa_constrained['test_sensitivity']):.3f}, "
        f"specificity={float(fa_constrained['test_specificity']):.3f}"
    )
    print(f"Outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
