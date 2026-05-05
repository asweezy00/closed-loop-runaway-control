"""Required figures for the clean final project."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from brian2 import Hz, second
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import Config
from .detector import DetectionResult
from .experiments import LongBaselineBundle, TrialBundle
from .lfp import LfpResult
from .metrics import rate_trace
from .network import NetworkRun


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _shade_runaway(ax, cfg: Config):
    ax.axvspan(cfg.runaway_onset_s, cfg.runaway_offset_s, color="#d95f02", alpha=0.16, label="induction window")
    ax.axvline(cfg.runaway_onset_s, color="#b3261e", ls="--", lw=1.0)


def fig_block_diagram(path: Path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    boxes = [
        (0.08, 0.46, "E/I network", "baseline -> runaway"),
        (0.32, 0.46, "LFP proxy", "weighted synaptic conductance"),
        (0.56, 0.46, "Detector", "line length > k sigma"),
        (0.80, 0.46, "Intervention", "safe inhibitory pulses"),
    ]
    for x, y, title, subtitle in boxes:
        patch = FancyBboxPatch((x, y), 0.15, 0.18, boxstyle="round,pad=0.02", fc="#eef4ed", ec="#0b2545", lw=1.5)
        ax.add_patch(patch)
        ax.text(x + 0.075, y + 0.12, title, ha="center", va="center", fontsize=12, weight="bold", color="#0b2545")
        ax.text(x + 0.075, y + 0.055, subtitle, ha="center", va="center", fontsize=8.5, color="#4b5563")
    for x0, x1 in [(0.23, 0.32), (0.47, 0.56), (0.71, 0.80)]:
        ax.add_patch(FancyArrowPatch((x0, 0.55), (x1, 0.55), arrowstyle="-|>", mutation_scale=16, lw=1.7, color="#1c7293"))
    ax.add_patch(FancyArrowPatch((0.875, 0.46), (0.155, 0.46), arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#8a5a00", connectionstyle="arc3,rad=-0.35"))
    ax.text(0.50, 0.18, "Closed-loop replay: sense -> decide -> intervene, with amplitude, duty-cycle, and pulse-count limits", ha="center", fontsize=11)
    ax.set_title("Closed-loop block diagram: controlled signals suppress runaway activity", fontsize=14, weight="bold")
    _save(fig, path)


def fig_runaway_example(path: Path, run: NetworkRun, lfp: LfpResult):
    cfg = run.config
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    sp = run.monitors["spike_E"]
    t_sp = np.asarray(sp.t / second)
    i_sp = np.asarray(sp.i)
    mask = i_sp < min(120, cfg.n_e)
    axes[0].plot(t_sp[mask], i_sp[mask], ".", ms=1.2, color="#111827", alpha=0.55)
    _shade_runaway(axes[0], cfg)
    axes[0].set_ylabel("E neuron index")
    axes[0].set_title("Excitatory raster: baseline state transitions into runaway activity")

    t_rate, rate_hz = rate_trace(run)
    axes[1].plot(t_rate, rate_hz, color="#1f4e79", lw=1.2)
    _shade_runaway(axes[1], cfg)
    axes[1].set_ylabel("E rate (Hz)")
    axes[1].set_title("Population firing rate")

    axes[2].plot(lfp.t_s, lfp.lfp_z, color="#111827", lw=0.9)
    _shade_runaway(axes[2], cfg)
    axes[2].set_ylabel("LFP proxy (z)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Simulated LFP proxy")
    _save(fig, path)


def fig_detector_trace(path: Path, lfp: LfpResult, detection_signal: np.ndarray, detection: DetectionResult, cfg: Config):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=False)
    axes[0].plot(lfp.t_s, detection_signal, color="#111827", lw=0.9)
    _shade_runaway(axes[0], cfg)
    for idx, t in enumerate(detection.detection_times_s):
        axes[0].axvline(t, color="#2e7d32", ls=":", lw=1.2, label="detections" if idx == 0 else "_nolegend_")
    axes[0].set_ylabel("LFP proxy (z)")
    axes[0].set_title("Detector input signal")

    axes[1].plot(detection.feature.t_s, detection.feature.values, color="#1f4e79", lw=1.2, label="line length")
    axes[1].axhline(detection.threshold, color="#b3261e", ls="--", lw=1.2, label=f"threshold k={detection.k:g}")
    _shade_runaway(axes[1], cfg)
    for idx, t in enumerate(detection.detection_times_s):
        axes[1].axvline(t, color="#2e7d32", ls=":", lw=1.0, label="detections" if idx == 0 else "_nolegend_")
    axes[1].set_ylabel("Line length (z/sample)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Detector feature trace and threshold")
    axes[1].legend(loc="upper right")
    _save(fig, path)


def fig_detector_performance(path: Path, benchmark_summary: list[dict], auc: float):
    rows = sorted(benchmark_summary, key=lambda r: float(r["k"]))
    k = np.asarray([float(r["k"]) for r in rows])
    sens = np.asarray([float(r["sensitivity"]) for r in rows])
    fpr = np.asarray([float(r["false_positive_rate"]) for r in rows])
    precision = np.asarray([float(r["precision"]) for r in rows])
    far = np.asarray([float(r["false_alarms_per_hour"]) for r in rows])
    lat = np.asarray([float(r["mean_latency_ms"]) for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    sc0 = axes[0].scatter(fpr, sens, c=k, cmap="viridis", s=58, zorder=3)
    axes[0].plot(fpr, sens, color="#1f4e79", alpha=0.65)
    axes[0].plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1.0)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title(f"ROC-style curve (AUC={auc:.2f})")
    label_offsets = {
        2.0: (6, 6),
        3.0: (6, -12),
        4.0: (6, -12),
        6.0: (6, 6),
        8.0: (6, 6),
    }
    for row in rows:
        kval = float(row["k"])
        if kval in label_offsets:
            axes[0].annotate(
                f"k={kval:g}",
                (float(row["false_positive_rate"]), float(row["sensitivity"])),
                fontsize=7,
                xytext=label_offsets[kval],
                textcoords="offset points",
            )

    axes[1].scatter(sens, precision, c=k, cmap="viridis", s=58, zorder=3)
    axes[1].plot(sens, precision, color="#2e7d32", alpha=0.65)
    axes[1].set_xlabel("Recall / sensitivity")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(-0.03, 1.03)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Precision-recall tradeoff")

    axes[2].scatter(far, lat, c=k, cmap="viridis", s=58, zorder=3)
    axes[2].plot(far, lat, color="#8a5a00", alpha=0.65)
    axes[2].set_xlabel("False alarms/hour")
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_title("Latency vs false-alarm burden")
    cbar = fig.colorbar(sc0, ax=axes.tolist(), location="right", shrink=0.82, fraction=0.025, pad=0.035)
    cbar.set_label("Threshold k")
    fig.suptitle("Detector benchmark on strong, weak, and no-runaway records", fontsize=13, weight="bold")
    _save(fig, path)


def fig_closed_loop(path: Path, trial: TrialBundle, baseline_trials: list[TrialBundle] | None = None):
    cfg = trial.cfg
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=False)
    axes[0].plot(trial.open_lfp.t_s, trial.open_lfp.lfp_z, color="#6b7280", lw=0.8, label="open-loop")
    axes[0].plot(trial.closed_lfp.t_s, trial.closed_lfp.lfp_z, color="#1f4e79", lw=0.9, label="closed-loop")
    _shade_runaway(axes[0], cfg)
    axes[0].set_ylabel("LFP proxy (z)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title(f"LFP: open-loop vs closed-loop (representative seed {trial.seed})")
    axes[0].legend(loc="upper right")
    axes[0].set_xlim(0, cfg.t_total_s)

    t_open, r_open = rate_trace(trial.open_run)
    t_closed, r_closed = rate_trace(trial.closed_run)
    axes[1].plot(t_open, r_open, color="#6b7280", lw=0.9, label="open-loop")
    axes[1].plot(t_closed, r_closed, color="#1f4e79", lw=0.9, label="closed-loop")
    _shade_runaway(axes[1], cfg)
    axes[1].set_ylabel("E rate (Hz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Population rate suppression")
    axes[1].set_xlim(0, cfg.t_total_s)

    for t in trial.schedule.safe_times_s:
        axes[2].broken_barh([(t, cfg.stim_duration_s)], (0.15, 0.7), facecolors="#2e7d32")
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_ylabel("Stim")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_xlim(0, cfg.t_total_s)
    axes[2].set_title(f"Safety-limited pulses: n={trial.metrics.n_pulses}, duty={trial.metrics.duty_cycle*100:.1f}%, burden={trial.metrics.burden_nS_s:.2f} nS*s")

    names = ["LFP power", "E rate", "duration"]
    if baseline_trials:
        lfp_arr = np.asarray([t.metrics.lfp_power_suppression_pct for t in baseline_trials], dtype=float)
        rate_arr = np.asarray([t.metrics.e_rate_suppression_pct for t in baseline_trials], dtype=float)
        dur_arr = np.asarray([t.metrics.runaway_duration_reduction_pct for t in baseline_trials], dtype=float)
        means = [float(np.mean(lfp_arr)), float(np.mean(rate_arr)), float(np.mean(dur_arr))]
        stds = [
            float(np.std(lfp_arr, ddof=1)) if lfp_arr.size > 1 else 0.0,
            float(np.std(rate_arr, ddof=1)) if rate_arr.size > 1 else 0.0,
            float(np.std(dur_arr, ddof=1)) if dur_arr.size > 1 else 0.0,
        ]
        axes[3].bar(
            names,
            means,
            yerr=stds,
            color=["#1f4e79", "#2e7d32", "#8a5a00"],
            capsize=6,
            error_kw={"elinewidth": 1.4, "ecolor": "#111827"},
        )
        axes[3].set_title(f"Closed-loop suppression metrics (mean ± SD across n={len(baseline_trials)} baseline seeds)")
    else:
        vals = [
            trial.metrics.lfp_power_suppression_pct,
            trial.metrics.e_rate_suppression_pct,
            trial.metrics.runaway_duration_reduction_pct,
        ]
        axes[3].bar(names, vals, color=["#1f4e79", "#2e7d32", "#8a5a00"])
        axes[3].set_title(f"Closed-loop suppression metrics (representative seed {trial.seed})")
    axes[3].axhline(0, color="#111827", lw=0.8)
    axes[3].set_ylabel("Reduction (%)")
    axes[3].set_xlabel("Suppression metric")
    _save(fig, path)


def fig_latency_sensitivity(path: Path, rows: list[dict]):
    """Plot mean suppression vs added latency with across-seed SD bands.

    `rows` may be either per-seed records (containing a 'seed' key) or already
    aggregated rows containing 'mean_*' / 'std_*' fields. Per-seed input is
    preferred so the figure can show variability bands.
    """
    rows = sorted(rows, key=lambda r: float(r["added_latency_ms"]))
    is_per_seed = any("seed" in r for r in rows)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))

    if is_per_seed:
        latencies = sorted({float(r["added_latency_ms"]) for r in rows})
        x = np.asarray(latencies, dtype=float)

        def _stats(field: str) -> tuple[np.ndarray, np.ndarray]:
            means: list[float] = []
            sds: list[float] = []
            for lat in latencies:
                vals = np.asarray(
                    [float(r[field]) for r in rows if float(r["added_latency_ms"]) == lat],
                    dtype=float,
                )
                vals = vals[np.isfinite(vals)]
                means.append(float(np.mean(vals)) if vals.size else float("nan"))
                sds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            return np.asarray(means), np.asarray(sds)

        lfp_m, lfp_sd = _stats("lfp_power_suppression_pct")
        rate_m, rate_sd = _stats("e_rate_suppression_pct")
        pulses_m, pulses_sd = _stats("n_pulses")
        n_seeds = len({int(r["seed"]) for r in rows if "seed" in r})

        ax1.errorbar(x, lfp_m, yerr=lfp_sd, fmt="o-", color="#1f4e79", capsize=5, label="LFP power reduction")
        ax1.fill_between(x, lfp_m - lfp_sd, lfp_m + lfp_sd, color="#1f4e79", alpha=0.15)
        ax1.errorbar(x, rate_m, yerr=rate_sd, fmt="s-", color="#2e7d32", capsize=5, label="E-rate reduction")
        ax1.fill_between(x, rate_m - rate_sd, rate_m + rate_sd, color="#2e7d32", alpha=0.15)
        ax1.set_title(f"Sensitivity to intervention latency (mean ± SD, n={n_seeds} seeds)")
    else:
        x = np.asarray([float(r["added_latency_ms"]) for r in rows])
        lfp_m = np.asarray([float(r["lfp_power_suppression_pct"]) for r in rows])
        rate_m = np.asarray([float(r["e_rate_suppression_pct"]) for r in rows])
        pulses_m = np.asarray([float(r["n_pulses"]) for r in rows])
        pulses_sd = np.zeros_like(pulses_m)
        ax1.plot(x, lfp_m, "o-", color="#1f4e79", label="LFP power reduction")
        ax1.plot(x, rate_m, "s-", color="#2e7d32", label="E-rate reduction")
        ax1.set_title("Sensitivity to intervention latency (representative seed)")

    ax1.axhline(0, color="#111827", lw=0.8)
    ax1.set_xlabel("Added intervention latency (ms)")
    ax1.set_ylabel("Suppression (%)")
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.errorbar(x, pulses_m, yerr=pulses_sd, fmt="^-", color="#8a5a00", capsize=4, alpha=0.8, label="Accepted pulses")
    ax2.set_ylabel("Accepted pulses (mean)")
    _save(fig, path)


def fig_long_record_false_alarms(path: Path, bundle: LongBaselineBundle):
    cfg = bundle.cfg
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=False)
    axes[0].plot(bundle.lfp.t_s, bundle.signal, color="#111827", lw=0.75)
    for idx, start in enumerate(bundle.artifact_times_s):
        axes[0].axvspan(
            start,
            start + cfg.long_artifact_duration_s,
            color="#d95f02",
            alpha=0.16,
            label="artifact burst" if idx == 0 else "_nolegend_",
        )
    for idx, t in enumerate(bundle.detection.detection_times_s):
        axes[0].axvline(t, color="#b3261e", ls=":", lw=1.0, label="false detection" if idx == 0 else "_nolegend_")
    axes[0].set_ylabel("LFP proxy (z)")
    axes[0].set_title(f"Long no-runaway validation, seed {bundle.seed}")
    if bundle.artifact_times_s or bundle.detection.detection_times_s:
        axes[0].legend(loc="upper right")

    axes[1].plot(bundle.detection.feature.t_s, bundle.detection.feature.values, color="#1f4e79", lw=1.0, label="line length")
    axes[1].axhline(bundle.detection.threshold, color="#b3261e", ls="--", lw=1.1, label=f"threshold k={bundle.detection.k:g}")
    for idx, start in enumerate(bundle.artifact_times_s):
        axes[1].axvspan(
            start,
            start + cfg.long_artifact_duration_s,
            color="#d95f02",
            alpha=0.16,
            label="artifact burst" if idx == 0 else "_nolegend_",
        )
    for idx, t in enumerate(bundle.detection.detection_times_s):
        axes[1].axvline(t, color="#b3261e", ls=":", lw=1.0, label="false detection" if idx == 0 else "_nolegend_")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Line length (z/sample)")
    axes[1].set_title(
        f"False alarms: {bundle.score.fp_events}, FA/h={bundle.score.false_alarms_per_hour:.1f}, "
        f"hypothetical pulses={bundle.schedule.n_pulses}"
    )
    axes[1].legend(loc="upper right")
    _save(fig, path)


def fig_long_fa_vs_k(path: Path, rows: list[dict]):
    rows = sorted(rows, key=lambda r: float(r["k"]))
    k = np.asarray([float(r["k"]) for r in rows], dtype=float)
    far = np.asarray([float(r["mean_false_alarms_per_hour"]) for r in rows], dtype=float)
    far_std = np.asarray([float(r.get("std_false_alarms_per_hour", 0.0)) for r in rows], dtype=float)
    pulses = np.asarray([float(r.get("mean_n_pulses", 0.0)) for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(8.8, 5.0))
    ax1.plot(k, far, "o-", color="#1f4e79", lw=1.8, label="False alarms/hour")
    ax1.fill_between(k, np.maximum(0.0, far - far_std), far + far_std, color="#1f4e79", alpha=0.16, label="Across-seed SD")
    ax1.axvline(4.0, color="#b3261e", ls="--", lw=1.2, label="default k=4")
    ax1.set_xlabel("Threshold multiplier k")
    ax1.set_ylabel("False alarms/hour")
    ax1.set_title("Long no-runaway false alarms decrease as threshold increases")
    ax1.set_xlim(k.min() - 0.15, k.max() + 0.15)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(k, pulses, "s-", color="#8a5a00", lw=1.3, alpha=0.8, label="Accepted pulses")
    ax2.set_ylabel("Mean accepted pulses per 60 s")
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    _save(fig, path)


def fig_feature_comparison(path: Path, feature_summary: list[dict]):
    rows = sorted(feature_summary, key=lambda r: (str(r["feature_kind"]), float(r["k"])))
    feature_order = ["line_length", "band_power", "variance"]
    colors = {
        "line_length": "#1f4e79",
        "band_power": "#2e7d32",
        "variance": "#8a5a00",
    }
    labels = {
        "line_length": "Line length",
        "band_power": "Band power 10-40 Hz",
        "variance": "Variance",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)

    for feature_kind in feature_order:
        subset = [r for r in rows if str(r["feature_kind"]) == feature_kind]
        if not subset:
            continue
        k = np.asarray([float(r["k"]) for r in subset])
        sens = np.asarray([float(r["sensitivity"]) for r in subset])
        fpr = np.asarray([float(r["false_positive_rate"]) for r in subset])
        precision = np.asarray([float(r["precision"]) for r in subset])
        far = np.asarray([float(r["false_alarms_per_hour"]) for r in subset])
        auc = float(subset[0]["auc"])
        color = colors.get(feature_kind, "#4b5563")
        label = f"{labels.get(feature_kind, feature_kind)} (AUC={auc:.2f})"

        axes[0].plot(fpr, sens, "o-", ms=3.5, lw=1.5, color=color, label=label)
        axes[1].plot(sens, precision, "o-", ms=3.5, lw=1.5, color=color, label=labels.get(feature_kind, feature_kind))
        axes[2].plot(k, far, "o-", ms=3.5, lw=1.5, color=color, label=labels.get(feature_kind, feature_kind))

    axes[0].plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1.0)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("ROC-style comparison")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].set_xlabel("Recall / sensitivity")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(-0.03, 1.03)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Precision-recall comparison")

    axes[2].set_xlabel("Threshold multiplier k")
    axes[2].set_ylabel("False alarms/hour")
    axes[2].set_title("False alarms vs threshold")
    axes[2].set_ylim(bottom=0)

    fig.suptitle("Detector feature comparison on the same mixed-record benchmark", fontsize=13, weight="bold")
    _save(fig, path)
