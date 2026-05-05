"""Brian2 E/I network with optional safety-filtered event-triggered replay.

The main experiment uses an offline closed-loop replay workflow: first run the
network without stimulation, detect events from the simulated LFP proxy, filter
the requested intervention times through the safety controller, then rerun the
same seed with those stimulation times applied. This preserves the sense-decide-
intervene structure while avoiding a fully online detector inside Brian2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import brian2
import numpy as np
from brian2 import (
    Hz,
    Network,
    NeuronGroup,
    PoissonInput,
    PopulationRateMonitor,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    defaultclock,
    ms,
    mV,
    nS,
    network_operation,
    pF,
    second,
)

from .config import Config

brian2.prefs.codegen.target = "numpy"
brian2.prefs.logging.console_log_level = "ERROR"


@dataclass
class NetworkRun:
    config: Config
    seed: int
    monitors: dict[str, Any]
    objects: dict[str, Any]
    stim_log: list[dict[str, float]]


EQNS = """
dv/dt = (g_L*(E_L - v) + g_e*(E_e - v) + g_i*(E_i - v) + g_stim*(E_i - v)) / C_m : volt (unless refractory)
dg_e/dt = -g_e / tau_e : siemens
dg_i/dt = -g_i / tau_i : siemens
g_stim : siemens
"""


def _namespace(cfg: Config) -> dict[str, Any]:
    g_l = (cfg.c_m_pF * pF) / (cfg.tau_m_ms * ms)
    return {
        "g_L": g_l,
        "E_L": cfg.e_l_mV * mV,
        "E_e": cfg.e_e_mV * mV,
        "E_i": cfg.e_i_mV * mV,
        "C_m": cfg.c_m_pF * pF,
        "tau_e": cfg.tau_e_ms * ms,
        "tau_i": cfg.tau_i_ms * ms,
        "V_th": cfg.v_th_mV * mV,
        "V_r": cfg.v_reset_mV * mV,
        "tau_ref": cfg.tau_ref_ms * ms,
    }


def build_network(cfg: Config, seed: int | None = None):
    """Build an unstimulated network with runtime inhibitory-weight reduction."""
    seed = cfg.seed_default if seed is None else int(seed)
    brian2.seed(seed)
    np.random.seed(seed)
    defaultclock.dt = cfg.dt_ms * ms

    ns = _namespace(cfg)
    neurons = NeuronGroup(
        cfg.n_total,
        model=EQNS,
        threshold="v > V_th",
        reset="v = V_r",
        refractory="tau_ref",
        method="euler",
        namespace=ns,
    )
    neurons.v = cfg.e_l_mV * mV + (cfg.v_th_mV - cfg.e_l_mV) * mV * np.random.rand(cfg.n_total)
    neurons.g_e = 0 * nS
    neurons.g_i = 0 * nS
    neurons.g_stim = 0 * nS

    e_neurons = neurons[: cfg.n_e]
    i_neurons = neurons[cfg.n_e :]

    w_e = cfg.w_e_nS * nS
    w_i = cfg.w_i_nS * nS

    s_ee = Synapses(e_neurons, e_neurons, on_pre="g_e_post += w_E", namespace={"w_E": w_e})
    s_ee.connect(condition="i != j", p=cfg.epsilon)

    s_ei = Synapses(e_neurons, i_neurons, on_pre="g_e_post += w_E", namespace={"w_E": w_e})
    s_ei.connect(p=cfg.epsilon)

    s_ie = Synapses(i_neurons, e_neurons, model="w_I_eff : siemens", on_pre="g_i_post += w_I_eff")
    s_ie.connect(p=cfg.epsilon)
    s_ie.w_I_eff = w_i

    s_ii = Synapses(i_neurons, i_neurons, model="w_I_eff : siemens", on_pre="g_i_post += w_I_eff")
    s_ii.connect(condition="i != j", p=cfg.epsilon)
    s_ii.w_I_eff = w_i

    poisson_e = PoissonInput(e_neurons, target_var="g_e", N=1, rate=cfg.rate_ext_hz * Hz, weight=cfg.w_ext_nS * nS)
    poisson_i = PoissonInput(i_neurons, target_var="g_e", N=1, rate=cfg.rate_ext_hz * Hz, weight=cfg.w_ext_nS * nS)

    n_record = min(cfg.n_record, cfg.n_e)
    spike_e = SpikeMonitor(e_neurons)
    spike_i = SpikeMonitor(i_neurons)
    rate_e = PopulationRateMonitor(e_neurons)
    rate_i = PopulationRateMonitor(i_neurons)
    state = StateMonitor(e_neurons, ["g_e", "g_i"], record=list(range(n_record)), dt=cfg.record_dt_ms * ms)

    @network_operation(dt=defaultclock.dt, when="start")
    def runaway_controller(t):
        t_s = float(t / second)
        if cfg.runaway_onset_s <= t_s < cfg.runaway_offset_s:
            s_ie.w_I_eff = w_i * cfg.inhibitory_weight_factor
            s_ii.w_I_eff = w_i * cfg.inhibitory_weight_factor
        else:
            s_ie.w_I_eff = w_i
            s_ii.w_I_eff = w_i

    net = Network(
        neurons,
        s_ee,
        s_ei,
        s_ie,
        s_ii,
        poisson_e,
        poisson_i,
        spike_e,
        spike_i,
        rate_e,
        rate_i,
        state,
        runaway_controller,
    )
    monitors = {"spike_E": spike_e, "spike_I": spike_i, "rate_E": rate_e, "rate_I": rate_i, "state": state}
    objects = {"neurons": neurons, "E_neurons": e_neurons, "I_neurons": i_neurons}
    return net, monitors, objects


def _make_stim_op(cfg: Config, objects: dict[str, Any], stim_times: list[float], seed: int):
    e_neurons = objects["E_neurons"]
    stim_log: list[dict[str, float]] = []
    rng = np.random.default_rng(seed + 1000)
    n_stim = max(1, int(round(cfg.stim_fraction * cfg.n_e)))
    intervals = [(float(t), float(t) + cfg.stim_duration_s) for t in sorted(stim_times)]
    state = {"idx": 0, "active": False, "chosen": None}

    @network_operation(dt=defaultclock.dt, when="start")
    def stim_controller(t):
        t_s = float(t / second)
        idx = int(state["idx"])

        while idx < len(intervals) and t_s >= intervals[idx][1]:
            if state["active"]:
                e_neurons.g_stim[state["chosen"]] = 0 * nS
                state["active"] = False
            idx += 1
            state["idx"] = idx

        if idx >= len(intervals):
            if state["active"]:
                e_neurons.g_stim[state["chosen"]] = 0 * nS
                state["active"] = False
            return

        start_s, end_s = intervals[idx]
        if start_s <= t_s < end_s and not state["active"]:
            chosen = rng.choice(cfg.n_e, size=n_stim, replace=False).astype(int)
            e_neurons.g_stim[chosen] = cfg.stim_amplitude_cap_nS * nS
            state["active"] = True
            state["chosen"] = chosen
            stim_log.append(
                {
                    "t_s": t_s,
                    "duration_s": cfg.stim_duration_s,
                    "amplitude_nS": cfg.stim_amplitude_cap_nS,
                    "fraction": n_stim / cfg.n_e,
                    "n_stim": float(n_stim),
                }
            )

    return stim_controller, stim_log


def run_network(
    cfg: Config,
    seed: int | None = None,
    stim_times: list[float] | None = None,
    report: bool = False,
) -> NetworkRun:
    """Run one network realization.

    If ``stim_times`` are supplied, they are applied as event-triggered replay
    times from a prior detector pass, not generated by an online Brian2 detector.
    """
    seed = cfg.seed_default if seed is None else int(seed)
    brian2.start_scope()
    net, monitors, objects = build_network(cfg, seed=seed)

    stim_log: list[dict[str, float]] = []
    if stim_times:
        stim_op, stim_log = _make_stim_op(cfg, objects, stim_times, seed)
        net.add(stim_op)

    net.run(cfg.t_total_s * second, report="text" if report else None, report_period=10 * second)
    return NetworkRun(config=cfg, seed=seed, monitors=monitors, objects=objects, stim_log=stim_log)
