"""Metrics for binary seizure-like event detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class BinaryMetrics:
    threshold: float
    roc_auc: float
    pr_auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_row(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "f1": self.f1,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def choose_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Choose a validation threshold by maximizing TPR minus FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5
    idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        finite = thresholds[np.isfinite(thresholds)]
        threshold = float(finite[0]) if finite.size else 0.5
    return threshold


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> BinaryMetrics:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    return BinaryMetrics(
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        sensitivity=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(specificity),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, float(roc_auc_score(y_true, y_score))


def pr_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return precision, recall, float(average_precision_score(y_true, y_score))

