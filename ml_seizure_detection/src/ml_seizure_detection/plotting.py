"""Figures for the ML detector."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import pr_points, roc_points


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 220,
    }
)


def _save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def fig_roc(path: Path, y_true: np.ndarray, model_scores: dict[str, np.ndarray], selected_name: str):
    plt.figure(figsize=(5.4, 4.2))
    for name, scores in model_scores.items():
        fpr, tpr, auc_value = roc_points(y_true, scores)
        label = f"{name.replace('_', ' ')} AUC={auc_value:.3f}"
        lw = 2.5 if name == selected_name else 1.5
        plt.plot(fpr, tpr, linewidth=lw, label=label)
    plt.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("Sensitivity")
    plt.title("Holdout ROC Curve")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.25)
    _save(path)


def fig_precision_recall(path: Path, y_true: np.ndarray, model_scores: dict[str, np.ndarray], selected_name: str):
    plt.figure(figsize=(5.4, 4.2))
    for name, scores in model_scores.items():
        precision, recall, ap = pr_points(y_true, scores)
        label = f"{name.replace('_', ' ')} AP={ap:.3f}"
        lw = 2.5 if name == selected_name else 1.5
        plt.plot(recall, precision, linewidth=lw, label=label)
    prevalence = float(np.mean(y_true))
    plt.axhline(prevalence, color="0.5", linestyle="--", linewidth=1, label=f"prevalence={prevalence:.2f}")
    plt.xlabel("Sensitivity")
    plt.ylabel("Precision")
    plt.title("Holdout Precision-Recall Curve")
    plt.legend(loc="lower left", frameon=True)
    plt.grid(alpha=0.25)
    _save(path)


def fig_confusion_matrix(path: Path, tn: int, fp: int, fn: int, tp: int, title: str):
    matrix = np.asarray([[tn, fp], [fn, tp]], dtype=int)
    plt.figure(figsize=(4.6, 4.0))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Predicted no runaway", "Predicted runaway"], rotation=25, ha="right")
    plt.yticks([0, 1], ["True no runaway", "True runaway"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14, color="black")
    plt.colorbar(label="Window count")
    plt.title(title)
    _save(path)


def fig_feature_importance(path: Path, importance_rows: list[dict[str, float | str]], top_n: int = 12):
    top = importance_rows[:top_n]
    names = [str(row["feature"]).replace("_", " ") for row in top][::-1]
    means = [float(row["importance_mean_auc_drop"]) for row in top][::-1]
    stds = [float(row["importance_std"]) for row in top][::-1]

    plt.figure(figsize=(6.6, 4.8))
    plt.barh(names, means, xerr=stds, color="#2b6cb0", alpha=0.85)
    plt.xlabel("Permutation importance (ROC-AUC drop)")
    plt.title("Top ML Detector Features")
    plt.grid(axis="x", alpha=0.25)
    _save(path)


def fig_long_fa_vs_threshold(path: Path, summary_rows: list[dict[str, float | int]], selected_threshold: float):
    thresholds = np.asarray([float(row["threshold"]) for row in summary_rows], dtype=float)
    mean_far = np.asarray([float(row["mean_false_alarms_per_hour"]) for row in summary_rows], dtype=float)
    std_far = np.asarray([float(row["std_false_alarms_per_hour"]) for row in summary_rows], dtype=float)

    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    mean_far = mean_far[order]
    std_far = std_far[order]

    plt.figure(figsize=(6.0, 4.2))
    plt.plot(thresholds, mean_far, color="#1f77b4", linewidth=2.2)
    plt.fill_between(thresholds, np.maximum(0.0, mean_far - std_far), mean_far + std_far, color="#1f77b4", alpha=0.18)
    plt.axvline(float(selected_threshold), color="#c0392b", linestyle="--", linewidth=1.8, label="selected threshold")
    plt.xlabel("ML probability threshold")
    plt.ylabel("False alarms/hour")
    plt.title("Long No-Runaway False-Alarm Rate")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(alpha=0.25)
    _save(path)


def fig_long_record_scores(
    path: Path,
    t_s: np.ndarray,
    signal: np.ndarray,
    centers_s: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    artifact_times_s: list[float],
):
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True)

    axes[0].plot(t_s, signal, color="#1f2937", linewidth=0.8)
    for art_t in artifact_times_s:
        axes[0].axvspan(float(art_t), float(art_t) + 0.30, color="#f59e0b", alpha=0.20)
    axes[0].set_ylabel("LFP proxy (z)")
    axes[0].set_title("Representative Long No-Runaway Record")
    axes[0].grid(alpha=0.20)

    axes[1].plot(centers_s, scores, color="#2563eb", linewidth=1.1, label="ML score")
    axes[1].axhline(float(threshold), color="#c0392b", linestyle="--", linewidth=1.5, label="threshold")
    axes[1].fill_between(centers_s, 0.0, 1.0, where=scores >= threshold, color="#c0392b", alpha=0.12, step="mid")
    for art_t in artifact_times_s:
        axes[1].axvspan(float(art_t), float(art_t) + 0.30, color="#f59e0b", alpha=0.20)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Runaway probability")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(alpha=0.20)

    _save(path)


def fig_operating_tradeoff(path: Path, operating_rows: list[dict[str, float | int]], target_fah: float):
    thresholds = np.asarray([float(row["threshold"]) for row in operating_rows], dtype=float)
    sensitivity = np.asarray([float(row["test_sensitivity"]) for row in operating_rows], dtype=float)
    f1 = np.asarray([float(row["test_f1"]) for row in operating_rows], dtype=float)
    far = np.asarray([float(row["mean_false_alarms_per_hour"]) for row in operating_rows], dtype=float)
    order = np.argsort(thresholds)

    fig, ax1 = plt.subplots(figsize=(6.6, 4.4))
    ax1.plot(thresholds[order], sensitivity[order], label="test sensitivity", color="#2563eb", linewidth=2.0)
    ax1.plot(thresholds[order], f1[order], label="test F1", color="#16a34a", linewidth=2.0)
    ax1.set_xlabel("ML probability threshold")
    ax1.set_ylabel("Holdout metric")
    ax1.set_ylim(-0.03, 1.03)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(thresholds[order], far[order], label="long FA/h", color="#dc2626", linewidth=2.0)
    ax2.axhline(float(target_fah), color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.75)
    ax2.set_ylabel("False alarms/hour")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)
    ax1.set_title("ML Operating Threshold Tradeoff")
    _save(path)
