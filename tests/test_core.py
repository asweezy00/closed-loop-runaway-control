from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runaway_control.config import Config
from runaway_control.controller import filter_stim_times
from runaway_control.detector import compute_feature, detect_feature, detect_line_length
from runaway_control.experiments import feature_benchmark_summary, roc_auc, run_trial
from runaway_control.lfp import compute_lfp
from runaway_control.metrics import compute_suppression_metrics, score_detections
from runaway_control.network import run_network


class CoreUnitTests(unittest.TestCase):
    def test_network_run_produces_finite_lfp(self):
        cfg = Config().test_sized()
        run = run_network(cfg, seed=1)
        lfp = compute_lfp(run.monitors["state"], cfg)
        self.assertEqual(lfp.t_s.shape, lfp.lfp_z.shape)
        self.assertTrue(np.all(np.isfinite(lfp.lfp_z)))
        self.assertGreater(lfp.t_s[-1], cfg.t_total_s * 0.9)

    def test_detector_does_not_fire_on_flat_baseline(self):
        cfg = Config().test_sized()
        t = np.arange(0, cfg.t_total_s, cfg.record_dt_ms / 1000.0)
        y = np.zeros_like(t)
        result = detect_line_length(t, y, cfg, k=4)
        self.assertEqual(result.detection_times_s, [])

    def test_detector_fires_on_synthetic_runaway(self):
        cfg = Config().test_sized()
        t = np.arange(0, cfg.t_total_s, cfg.record_dt_ms / 1000.0)
        y = np.zeros_like(t)
        idx = t >= cfg.runaway_onset_s
        y[idx] = 5.0 * np.sin(2 * np.pi * 35.0 * t[idx])
        result = detect_line_length(t, y, cfg, k=3)
        self.assertTrue(any(tt >= cfg.runaway_onset_s for tt in result.detection_times_s))

    def test_feature_traces_are_finite_and_aligned(self):
        cfg = Config()
        t = np.arange(0, cfg.t_total_s, cfg.record_dt_ms / 1000.0)
        y = np.sin(2 * np.pi * 20.0 * t)
        for feature_kind in cfg.feature_kinds:
            with self.subTest(feature_kind=feature_kind):
                feature = compute_feature(t, y, cfg, feature_kind=feature_kind)
                self.assertEqual(feature.t_s.shape, feature.values.shape)
                self.assertGreater(feature.t_s.size, 0)
                self.assertTrue(np.all(np.isfinite(feature.values)))
                self.assertTrue(np.all(np.diff(feature.t_s) > 0))

    def test_unknown_feature_kind_raises(self):
        cfg = Config().test_sized()
        t = np.arange(0, cfg.t_total_s, cfg.record_dt_ms / 1000.0)
        y = np.zeros_like(t)
        with self.assertRaisesRegex(ValueError, "unknown feature_kind"):
            detect_feature(t, y, cfg, feature_kind="not_a_feature")

    def test_generic_line_length_matches_legacy_detector(self):
        cfg = Config().test_sized()
        t = np.arange(0, cfg.t_total_s, cfg.record_dt_ms / 1000.0)
        y = np.zeros_like(t)
        idx = t >= cfg.runaway_onset_s
        y[idx] = 4.0 * np.sin(2 * np.pi * 35.0 * t[idx])
        legacy = detect_line_length(t, y, cfg, k=3)
        generic = detect_feature(t, y, cfg, k=3, feature_kind="line_length")
        self.assertEqual(legacy.detection_times_s, generic.detection_times_s)
        self.assertAlmostEqual(legacy.threshold, generic.threshold)

    def test_band_power_and_variance_fire_on_synthetic_signal(self):
        cfg = Config()
        t = np.arange(0, cfg.t_total_s, cfg.record_dt_ms / 1000.0)
        y = np.zeros_like(t)
        idx = t >= cfg.runaway_onset_s
        y[idx] = 5.0 * np.sin(2 * np.pi * 20.0 * t[idx])
        for feature_kind in ["band_power", "variance"]:
            with self.subTest(feature_kind=feature_kind):
                result = detect_feature(t, y, cfg, k=3, feature_kind=feature_kind)
                self.assertTrue(any(tt >= cfg.runaway_onset_s for tt in result.detection_times_s))

    def test_controller_enforces_spacing_and_duty(self):
        cfg = Config().test_sized().with_updates(
            stim_duration_s=0.1,
            min_pulse_spacing_s=0.2,
            max_duty_cycle=0.34,
            max_pulses=5,
        )
        schedule = filter_stim_times([0.30, 0.35, 0.50, 0.70], cfg)
        self.assertLessEqual(schedule.duty_cycle, cfg.max_duty_cycle)
        self.assertEqual(schedule.safe_times_s, [0.30, 0.50])
        self.assertGreater(schedule.skipped_cooldown + schedule.skipped_duty, 0)

    def test_metrics_are_finite_on_valid_trials(self):
        cfg = Config().test_sized()
        open_run = run_network(cfg, seed=2)
        open_lfp = compute_lfp(open_run.monitors["state"], cfg)
        schedule = filter_stim_times([cfg.runaway_onset_s + 0.05], cfg)
        closed_run = run_network(cfg, seed=2, stim_times=schedule.safe_times_s)
        closed_lfp = compute_lfp(closed_run.monitors["state"], cfg)
        metrics = compute_suppression_metrics(open_run, closed_run, open_lfp, closed_lfp, schedule)
        for value in [
            metrics.lfp_power_suppression_pct,
            metrics.e_rate_suppression_pct,
            metrics.runaway_duration_reduction_pct,
            metrics.open_eval_rate_hz,
            metrics.closed_eval_rate_hz,
            metrics.burden_nS_s,
        ]:
            self.assertTrue(math.isfinite(value))

    def test_score_detections_separates_tp_and_fp(self):
        cfg = Config()
        score = score_detections(
            [
                cfg.runaway_onset_s - 0.5,
                cfg.runaway_onset_s + 0.05,
            ],
            cfg,
        )
        self.assertEqual(score.tp_events, 1)
        self.assertEqual(score.fp_events, 1)
        self.assertEqual(score.n_events, 2)
        self.assertEqual(score.sensitivity, 1.0)
        self.assertAlmostEqual(score.latency_ms, 50.0)
        self.assertGreater(score.false_alarms_per_hour, 0.0)

    def test_roc_auc_known_curves(self):
        perfect = [{"false_positive_rate": 0.0, "sensitivity": 1.0}]
        diagonal = [
            {"false_positive_rate": 0.0, "sensitivity": 0.0},
            {"false_positive_rate": 1.0, "sensitivity": 1.0},
        ]
        self.assertAlmostEqual(roc_auc(perfect), 1.0)
        self.assertAlmostEqual(roc_auc(diagonal), 0.5)

    def test_feature_benchmark_summary_groups_feature_and_threshold(self):
        rows = []
        for feature_kind in ["line_length", "variance"]:
            for k in [3.0, 4.0]:
                rows.extend(
                    [
                        {
                            "feature_kind": feature_kind,
                            "k": k,
                            "has_runaway": 1,
                            "tp_record": 1,
                            "fp_record": 0,
                            "false_event_count": 0,
                            "non_event_hours": 1.0,
                            "latency_ms": 50.0,
                        },
                        {
                            "feature_kind": feature_kind,
                            "k": k,
                            "has_runaway": 0,
                            "tp_record": 0,
                            "fp_record": int(k == 3.0),
                            "false_event_count": int(k == 3.0),
                            "non_event_hours": 1.0,
                            "latency_ms": float("nan"),
                        },
                    ]
                )
        summary = feature_benchmark_summary(rows)
        self.assertEqual(len(summary), 4)
        self.assertEqual({row["feature_kind"] for row in summary}, {"line_length", "variance"})
        self.assertTrue(all("auc" in row for row in summary))

    def test_run_trial_finishes_for_stress_scenarios(self):
        cfg = Config().test_sized()
        for scenario in ["baseline", "noisy_lfp", "parameter_shifted"]:
            with self.subTest(scenario=scenario):
                trial = run_trial(3, cfg, scenario=scenario)
                for value in [
                    trial.score.sensitivity,
                    trial.score.false_alarms_per_hour,
                    trial.metrics.lfp_power_suppression_pct,
                    trial.metrics.e_rate_suppression_pct,
                    trial.metrics.runaway_duration_reduction_pct,
                    trial.metrics.burden_nS_s,
                ]:
                    self.assertTrue(math.isfinite(value))
                if scenario == "baseline":
                    self.assertGreaterEqual(trial.metrics.lfp_power_suppression_pct, 0.0)

    def test_seed42_baseline_suppression_directional_regression(self):
        trial = run_trial(42, Config(), scenario="baseline")
        self.assertGreaterEqual(trial.metrics.lfp_power_suppression_pct, 20.0)
        self.assertGreaterEqual(trial.metrics.runaway_duration_reduction_pct, 15.0)


if __name__ == "__main__":
    unittest.main()
