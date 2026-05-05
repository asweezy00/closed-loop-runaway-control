# Clean Final Project: Model-Based Runaway Suppression

This folder contains the focused final-project implementation for Challenge E:
controlled signals suppressing runaway neural activity in a simulated
excitatory-inhibitory spiking network.

This is the BME 525 graduate final project codebase for a model-based
closed-loop detector/controller. The default detector uses an LFP-derived
line-length feature, then triggers safety-limited inhibitory stimulation in an
event-triggered replay simulation.

## Install

```bash
python -m pip install -e .
```

## Run

```bash
python scripts/run_core.py --quick --out results/core_run_quick
python scripts/run_core.py --out results/core_run --seeds 42-51
```

## Outputs

The runner writes all required code-side project artifacts:

- `figures/fig1_block_diagram.png`
- `figures/fig2_runaway_example.png`
- `figures/fig3_detector_trace.png`
- `figures/fig4_detector_performance.png`
- `figures/fig4b_feature_comparison.png`
- `figures/fig5_closed_loop_suppression.png`
- `figures/fig6_latency_sensitivity.png`
- `figures/fig7_long_record_fa.png`
- `figures/fig8_long_fa_vs_k.png`
- `tables/table1_stress_summary.csv`
- `tables/table1_stress_summary.md`
- `tables/feature_benchmark.csv`
- `tables/feature_benchmark_summary.csv`
- `tables/long_baseline_false_alarms.csv`
- `tables/long_baseline_fa_by_k.csv`

## Tests

```bash
python -m unittest discover tests
```

## ML Extension

A separate machine-learning seizure-like event detector is included in
`ml_seizure_detection/`. It trains supervised classifiers on simulated LFP
windows, reports ROC-AUC/PR-AUC metrics, and validates long-record
false alarms/hour across probability thresholds.

```bash
../.venv/bin/python -m unittest discover ml_seizure_detection/tests
../.venv/bin/python ml_seizure_detection/scripts/run_ml.py --out ml_seizure_detection/results/ml_run --seeds 42-51
```

The most conservative evaluated operating point was threshold `0.82`, which
gave `0.0` false alarms/hour on five 60 s no-runaway artifact-corrupted
simulation records while retaining `0.861` held-out window sensitivity.
