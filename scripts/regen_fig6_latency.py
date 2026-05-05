#!/usr/bin/env python3
"""Run latency sweep across multiple baseline seeds and regenerate fig6 with error bars."""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runaway_control.config import Config
from runaway_control.controller import filter_stim_times
from runaway_control.experiments import run_trial
from runaway_control.lfp import compute_lfp
from runaway_control.metrics import compute_suppression_metrics
from runaway_control.network import run_network
from runaway_control.plotting import fig_latency_sensitivity


def latency_sweep(trial, cfg: Config) -> list[dict]:
    rows = []
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


def main(action: str, *args: str) -> int:
    out_dir = ROOT / "results" / "core_run"
    figures = out_dir / "figures"
    tables = out_dir / "tables"
    accum_path = tables / "_latency_per_seed_accum.csv"

    if action == "sweep":
        seed = int(args[0])
        cfg = Config()
        t0 = time.time()
        trial = run_trial(seed=seed, base_cfg=cfg, scenario="baseline", k=cfg.default_k, open_cache={})
        rows = latency_sweep(trial, cfg)
        write_header = not accum_path.exists()
        with accum_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"[regen6] seed {seed} done in {time.time()-t0:.1f}s; appended {len(rows)} rows to {accum_path.name}")
        return 0

    if action == "plot":
        if not accum_path.exists():
            print("[regen6] No accumulator file found")
            return 1
        with accum_path.open() as f:
            rows = [{k: float(v) if k != "seed" else int(float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
        # Promote accumulator to canonical CSV
        canonical = tables / "latency_sensitivity.csv"
        with canonical.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        fig_latency_sensitivity(figures / "fig6_latency_sensitivity.png", rows)
        n_seeds = len({int(r["seed"]) for r in rows})
        print(f"[regen6] Wrote fig6 from {len(rows)} rows across {n_seeds} seeds")
        return 0

    if action == "reset":
        if accum_path.exists():
            accum_path.unlink()
        print("[regen6] reset accumulator")
        return 0

    print(f"[regen6] Unknown action: {action}. Use 'sweep <seed>', 'plot', or 'reset'.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: regen_fig6_latency.py <sweep <seed> | plot | reset>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], *sys.argv[2:]))
