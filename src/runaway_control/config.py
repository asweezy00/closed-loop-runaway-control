"""Configuration for the clean runaway-control project."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    """Single source of truth for the final project simulations.

    Values are stored as plain floats so analysis and tests stay unit-light.
    Brian2 units are attached only inside the network builder.
    """

    # Simulation
    dt_ms: float = 0.1
    record_dt_ms: float = 1.0
    t_total_s: float = 4.0
    baseline_end_s: float = 2.0
    seed_default: int = 42

    # Network
    n_e: int = 800
    n_i: int = 200
    epsilon: float = 0.1
    c_m_pF: float = 200.0
    tau_m_ms: float = 20.0
    e_l_mV: float = -60.0
    e_e_mV: float = 0.0
    e_i_mV: float = -80.0
    v_th_mV: float = -50.0
    v_reset_mV: float = -60.0
    tau_ref_ms: float = 2.0
    tau_e_ms: float = 5.0
    tau_i_ms: float = 10.0
    w_e_nS: float = 0.6
    w_i_nS: float = 6.7
    rate_ext_hz: float = 1400.0
    w_ext_nS: float = 0.6

    # Runaway induction
    runaway_onset_s: float = 2.0
    runaway_duration_s: float = 1.0
    inhibitory_weight_factor: float = 0.3

    # LFP proxy
    n_record: int = 50
    alpha_lfp: float = 1.0
    beta_lfp: float = -1.65
    lfp_lowpass_hz: float = 100.0
    lfp_filter_order: int = 4

    # Detector
    line_length_window_s: float = 0.250
    feature_step_s: float = 0.010
    detector_persistence_s: float = 0.050
    detector_cooldown_s: float = 0.200
    detector_lowpass_hz: float = 50.0
    detector_filter_order: int = 4
    feature_kind: str = "line_length"
    feature_kinds: tuple[str, ...] = ("line_length", "band_power", "variance")
    band_power_low_hz: float = 10.0
    band_power_high_hz: float = 40.0
    default_k: float = 4.0
    threshold_values: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 7.0)
    roc_threshold_values: tuple[float, ...] = (
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.25,
        4.5,
        4.75,
        5.0,
        5.25,
        5.5,
        5.75,
        6.0,
        6.25,
        6.5,
        6.75,
        7.0,
        7.25,
        7.5,
        7.75,
        8.0,
    )

    # Intervention
    stim_amplitude_cap_nS: float = 50.0
    stim_duration_s: float = 0.100
    stim_fraction: float = 0.50
    max_pulses: int = 5
    min_pulse_spacing_s: float = 0.200
    max_duty_cycle: float = 0.15
    latency_sweep_ms: tuple[float, ...] = (0.0, 100.0, 250.0, 500.0)

    # Long no-runaway false-alarm validation
    long_baseline_duration_s: float = 60.0
    quick_long_baseline_duration_s: float = 8.0
    long_baseline_seed_count: int = 5
    long_artifact_rate_per_min: float = 10.0
    long_artifact_duration_s: float = 0.30
    long_artifact_amplitude: float = 1.0

    @property
    def n_total(self) -> int:
        return self.n_e + self.n_i

    @property
    def runaway_offset_s(self) -> float:
        return self.runaway_onset_s + self.runaway_duration_s

    def with_updates(self, **kwargs) -> "Config":
        return replace(self, **kwargs)

    def parameter_shifted(self) -> "Config":
        """Scenario required by the rubric: w_E +10%, w_I -10%, tau_m +10%."""
        return self.with_updates(
            w_e_nS=self.w_e_nS * 1.10,
            w_i_nS=self.w_i_nS * 0.90,
            tau_m_ms=self.tau_m_ms * 1.10,
        )

    def test_sized(self) -> "Config":
        """Small deterministic config for unit tests only."""
        return self.with_updates(
            n_e=40,
            n_i=10,
            n_record=8,
            epsilon=0.2,
            t_total_s=0.6,
            baseline_end_s=0.25,
            runaway_onset_s=0.25,
            runaway_duration_s=0.2,
            record_dt_ms=1.0,
        )


def parse_seed_spec(spec: str) -> list[int]:
    """Parse seed specs such as ``42-51`` or ``42,43,50``."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            seeds.extend(range(lo, hi + step, step))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds
