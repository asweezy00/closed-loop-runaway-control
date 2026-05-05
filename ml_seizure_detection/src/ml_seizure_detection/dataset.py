"""Build ML datasets from the existing simulated runaway-control model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from .features import FEATURE_NAMES, WindowedFeatures, concatenate_feature_sets, window_signal


FINAL_PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_PROJECT_SRC = FINAL_PROJECT_ROOT / "src"
if str(FINAL_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(FINAL_PROJECT_SRC))

from runaway_control.config import Config
from runaway_control.lfp import add_artifact_burst, add_gaussian_noise, compute_lfp
from runaway_control.network import run_network


@dataclass(frozen=True)
class RecordSpec:
    """One simulated record used to train or test the ML detector."""

    record_type: str
    has_runaway: bool
    cfg: Config
    signal_kind: str = "clean"


def ml_record_specs(base_cfg: Config) -> list[RecordSpec]:
    """Return a compact but diverse benchmark suite.

    Positive records include strong, weak, and noisy runaway examples.
    Negative records include clean no-runaway and artifact-corrupted no-runaway
    examples so the classifier has to learn seizure-like structure rather than
    simply flagging every high-amplitude transient.
    """
    return [
        RecordSpec("strong_runaway", True, base_cfg, "clean"),
        RecordSpec("weak_runaway", True, base_cfg.with_updates(inhibitory_weight_factor=0.50), "clean"),
        RecordSpec("noisy_runaway", True, base_cfg, "noise"),
        RecordSpec("no_runaway_clean", False, base_cfg.with_updates(inhibitory_weight_factor=1.00), "clean"),
        RecordSpec("no_runaway_artifact", False, base_cfg.with_updates(inhibitory_weight_factor=1.00), "artifact"),
    ]


def _signal_for_spec(t_s: np.ndarray, lfp_z: np.ndarray, spec: RecordSpec, seed: int) -> np.ndarray:
    if spec.signal_kind == "noise":
        return add_gaussian_noise(lfp_z, snr_db=10.0, seed=seed + 10_000)
    if spec.signal_kind == "artifact":
        return add_artifact_burst(t_s, lfp_z, seed=seed + 20_000, amplitude=1.0)
    if spec.signal_kind == "clean":
        return lfp_z.copy()
    raise ValueError(f"unknown signal kind: {spec.signal_kind}")


def build_simulated_window_dataset(
    seeds: list[int],
    cfg: Config,
    window_s: float = 0.500,
    step_s: float = 0.050,
) -> WindowedFeatures:
    """Generate simulated records and convert them into labeled windows."""
    parts: list[WindowedFeatures] = []
    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for spec in ml_record_specs(cfg):
        for seed in seeds:
            cache_key = (spec.record_type, int(seed))
            if cache_key in cache:
                t_s, lfp_z = cache[cache_key]
            else:
                run = run_network(spec.cfg, seed=seed)
                lfp = compute_lfp(run.monitors["state"], spec.cfg)
                t_s, lfp_z = lfp.t_s, lfp.lfp_z
                cache[cache_key] = (t_s, lfp_z)

            signal = _signal_for_spec(t_s, lfp_z, spec, seed)
            positive_interval = (spec.cfg.runaway_onset_s, spec.cfg.runaway_offset_s) if spec.has_runaway else None
            record_id = f"{spec.record_type}_seed_{seed}"
            parts.append(
                window_signal(
                    t_s=t_s,
                    signal=signal,
                    window_s=window_s,
                    step_s=step_s,
                    positive_interval_s=positive_interval,
                    record_id=record_id,
                    record_type=spec.record_type,
                    seed=seed,
                )
            )
    return concatenate_feature_sets(parts)


def dataset_rows(dataset: WindowedFeatures) -> list[dict[str, float | int | str]]:
    """Serialize the window dataset for auditability."""
    rows: list[dict[str, float | int | str]] = []
    for idx in range(dataset.x.shape[0]):
        row: dict[str, float | int | str] = {
            "row": idx,
            "label": int(dataset.y[idx]),
            "center_s": float(dataset.centers_s[idx]),
            "start_s": float(dataset.starts_s[idx]),
            "end_s": float(dataset.ends_s[idx]),
            "record_id": dataset.record_ids[idx],
            "record_type": dataset.record_types[idx],
            "seed": int(dataset.seeds[idx]),
        }
        row.update({name: float(dataset.x[idx, j]) for j, name in enumerate(FEATURE_NAMES)})
        rows.append(row)
    return rows


def quick_ml_config() -> Config:
    """Faster config for smoke tests and quick ML runs."""
    return Config().with_updates(
        n_e=240,
        n_i=60,
        n_record=30,
        epsilon=0.12,
        t_total_s=3.5,
        baseline_end_s=1.5,
        runaway_onset_s=1.5,
        runaway_duration_s=1.0,
    )
