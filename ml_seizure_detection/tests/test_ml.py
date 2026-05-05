from __future__ import annotations

import sys
import unittest
from pathlib import Path
import os

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL_PROJECT_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
for path in (ROOT / "src", FINAL_PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ml_seizure_detection.features import FEATURE_NAMES, WindowedFeatures, extract_window_feature_vector, window_signal
from ml_seizure_detection.long_validation import (
    choose_fa_constrained_threshold,
    merge_positive_windows,
    score_false_alarms_per_hour,
)
from ml_seizure_detection.metrics import binary_metrics, choose_threshold_youden
from ml_seizure_detection.models import fit_best_model


class MlDetectorTests(unittest.TestCase):
    def test_feature_vector_is_finite_and_expected_length(self):
        fs_hz = 200.0
        t = np.arange(0.0, 1.0, 1.0 / fs_hz)
        signal = np.sin(2 * np.pi * 20.0 * t) + 0.2 * np.sin(2 * np.pi * 4.0 * t)
        features = extract_window_feature_vector(signal, fs_hz)
        self.assertEqual(features.shape[0], len(FEATURE_NAMES))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertGreater(features[FEATURE_NAMES.index("band_10_40_hz")], 0.0)

    def test_window_labels_follow_positive_interval(self):
        t = np.arange(0.0, 4.0, 0.01)
        signal = np.sin(2 * np.pi * 8.0 * t)
        dataset = window_signal(
            t_s=t,
            signal=signal,
            window_s=0.5,
            step_s=0.25,
            positive_interval_s=(1.5, 2.5),
            record_id="synthetic",
            record_type="synthetic",
            seed=1,
        )
        self.assertGreater(int(np.sum(dataset.y == 1)), 0)
        self.assertGreater(int(np.sum(dataset.y == 0)), 0)
        positive_centers = dataset.centers_s[dataset.y == 1]
        self.assertTrue(np.all((positive_centers >= 1.25) & (positive_centers <= 2.75)))

    def test_threshold_and_metrics_on_separable_scores(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
        threshold = choose_threshold_youden(y, scores)
        metrics = binary_metrics(y, scores, threshold)
        self.assertAlmostEqual(metrics.roc_auc, 1.0)
        self.assertAlmostEqual(metrics.pr_auc, 1.0)
        self.assertEqual(metrics.fp, 0)
        self.assertEqual(metrics.fn, 0)

    def test_long_false_alarm_scoring_merges_adjacent_windows(self):
        centers = np.asarray([1.0, 1.05, 1.10, 2.0, 3.0, 3.40])
        scores = np.asarray([0.9, 0.8, 0.7, 0.1, 0.95, 0.96])
        events = merge_positive_windows(centers, scores >= 0.5, gap_tolerance_s=0.30)
        self.assertEqual(len(events), 3)
        scored = score_false_alarms_per_hour(centers, scores, threshold=0.5, duration_s=60.0, gap_tolerance_s=0.30)
        self.assertEqual(scored["false_alarm_events"], 3)
        self.assertAlmostEqual(scored["false_alarms_per_hour"], 180.0)

    def test_fa_constrained_threshold_prefers_best_feasible_f1(self):
        rows = [
            {"threshold": 0.2, "mean_false_alarms_per_hour": 180.0, "test_f1": 0.99, "test_sensitivity": 1.0},
            {"threshold": 0.5, "mean_false_alarms_per_hour": 60.0, "test_f1": 0.94, "test_sensitivity": 0.95},
            {"threshold": 0.8, "mean_false_alarms_per_hour": 0.0, "test_f1": 0.90, "test_sensitivity": 0.86},
        ]
        selected = choose_fa_constrained_threshold(rows, target_fah=60.0)
        self.assertEqual(selected["threshold"], 0.5)

    def test_fit_best_model_on_synthetic_dataset(self):
        rng = np.random.default_rng(5)
        x_neg = rng.normal(loc=-1.0, scale=0.25, size=(120, len(FEATURE_NAMES)))
        x_pos = rng.normal(loc=1.0, scale=0.25, size=(120, len(FEATURE_NAMES)))
        x = np.vstack([x_neg, x_pos])
        y = np.asarray([0] * x_neg.shape[0] + [1] * x_pos.shape[0])
        groups = [f"neg_{i // 4}" for i in range(x_neg.shape[0])] + [f"pos_{i // 4}" for i in range(x_pos.shape[0])]
        n = x.shape[0]
        dataset = WindowedFeatures(
            x=x,
            y=y,
            centers_s=np.arange(n, dtype=float),
            starts_s=np.arange(n, dtype=float),
            ends_s=np.arange(n, dtype=float) + 0.5,
            record_ids=groups,
            record_types=["synthetic"] * n,
            seeds=[0] * n,
        )
        fitted = fit_best_model(dataset, random_state=11)
        self.assertGreaterEqual(fitted.metrics.roc_auc, 0.95)
        self.assertGreaterEqual(fitted.metrics.sensitivity, 0.90)
        self.assertGreaterEqual(fitted.metrics.specificity, 0.90)


if __name__ == "__main__":
    unittest.main()
