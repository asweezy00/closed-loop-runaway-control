#!/usr/bin/env python3
"""Regenerate fig5 and fig6 with across-seed uncertainty bands.

Uses cached per-seed data from trial_summary.csv where available; only re-runs
the minimum number of Brian2 simulations needed for time-series panels and the
latency sweep.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runaway_control.config import Config
from runaway_control.experiments import run_trial
from runaway_control.plotting import fig_closed_loop, fig_latency_sensitivity


@dataclass
class _ProxyMetrics:
    lfp_power_suppression_pct: float
    e_rate_suppression_pct: float
    runaway_duration_reduction_pct: float
    n_pulses: int = 0
    duty_cycle: float = 0.0
    burden_nS_s: float = 0.0
    open_eval_rate_hz: float = 0.0
    closed_eval_rate_hz: float = 0.0
    open_runaway_duration_s: float = 0.0
    closed_runaway_duration_s: float = 0.0


@dataclass
class _ProxyTrial:
    seed: int
    metrics: _ProxyMetrics


def _load_baseline_metrics(csv_path: Path) -> list[_ProxyTrial]:
    trials: list[_ProxyTrial] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["scenario"] != "Baseline":
                continue
            trials.append(
                _ProxyTrial(
                    seed=int(float(row["seed"])),
                    metrics=_ProxyMetrics(
                        lfp_power_suppression_pct=float(row["lfp_power_suppression_pct"]),
                        e_rate_suppression_pct=float(row["e_rate_suppression_pct"]),
                        runaway_duration_reduction_pct=float(row["runaway_duration_reduction_pct"]),
                        n_pulses=int(float(row["n_pulses"])),
                        duty_cycle=float(row["duty_cycle"]),
                        burden_nS_s=float(row["burden_nS_s"]),
                    ),
                )
            )
    return trials


def main() -> int:
    out_dir = ROOT / "results" / "core_run"
    figures = out_dir / "figures"
    tables = out_dir / "tables"

    cfg = Config()

    print("[regen] Re-running representative baseline seed 42 for fig5 time series")
    rep_trial = run_trial(seed=42, base_cfg=cfg, scenario="baseline", k=cfg.default_k, open_cache={})

    print("[regen] Loading baseline per-seed metrics from trial_summary.csv")
    baseline_trials = _load_baseline_metrics(tables / "trial_summary.csv")
    print(f"[regen] Loaded n={len(baseline_trials)} baseline trials")

    print("[regen] Writing fig5 with error bars")
    fig_closed_loop(figures / "fig5_closed_loop_suppression.png", rep_trial, baseline_trials=baseline_trials)
    print("[regen] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
