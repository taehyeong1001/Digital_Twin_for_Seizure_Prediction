#!/usr/bin/env python3
"""Shared Proix-2017 Epileptor dynamics for single- and multi-node GT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


# Rounded excitability boundary used by Proix et al. (Brain, 2017).
# The reduced two-dimensional equations place the corresponding Hopf point
# slightly higher, near -2.062; keep the paper value separate from empirical
# finite-horizon event thresholds.
X0_CRITICAL = -2.1


@dataclass(frozen=True)
class PaperEpileptorConfig:
    n_nodes: int = 8
    sampling_hz: int = 256
    duration_s: float = 180.0
    integration_dt: float = 0.04
    tau0: float = 2857.0
    tau2: float = 10.0
    i1: float = 3.1
    i2: float = 0.45
    gamma: float = 0.01
    global_coupling: float = 1.0
    noise_variance: float = 0.0025
    seed: int = 52


def paper_epileptor_rhs(
    states: np.ndarray,
    x0: np.ndarray,
    connectivity: np.ndarray,
    config: PaperEpileptorConfig,
) -> np.ndarray:
    """Evaluate Eq. (4) of Proix et al., Brain 2017."""
    x1, y1, z, x2, y2, lowpass = states.T
    f1 = np.where(
        x1 < 0.0,
        x1**3 - 3.0 * x1**2,
        (x2 - 0.6 * (z - 4.0) ** 2) * x1,
    )
    f2 = np.where(x2 < -0.25, 0.0, 6.0 * (x2 + 0.25))
    difference_coupling = (
        connectivity @ x1 - connectivity.sum(axis=1) * x1
    )

    derivative = np.empty_like(states)
    derivative[:, 0] = y1 - f1 - z + config.i1
    derivative[:, 1] = 1.0 - 5.0 * x1**2 - y1
    derivative[:, 2] = (
        4.0 * (x1 - x0)
        - z
        - config.global_coupling * difference_coupling
    ) / config.tau0
    derivative[:, 3] = (
        -y2
        + x2
        - x2**3
        + config.i2
        + 0.002 * lowpass
        - 0.3 * (z - 3.5)
    )
    derivative[:, 4] = (-y2 + f2) / config.tau2
    derivative[:, 5] = x1 - config.gamma * lowpass
    return derivative


def default_initial_state(
    n_nodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base_state = np.array([-1.55, -11.8, 3.05, -0.82, 0.0, 0.0])
    perturbation_scale = np.array([0.03, 0.20, 0.04, 0.02, 0.02, 0.03])
    states = np.tile(base_state, (n_nodes, 1))
    states += rng.normal(0.0, perturbation_scale, states.shape)
    return states


def stochastic_heun_step(
    states: np.ndarray,
    x0: np.ndarray,
    connectivity: np.ndarray,
    config: PaperEpileptorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance one additive-noise stochastic Heun step."""
    drift = paper_epileptor_rhs(states, x0, connectivity, config)
    noise = np.zeros_like(states)
    noise_scale = np.sqrt(config.noise_variance * config.integration_dt)
    noise[:, 3] = noise_scale * rng.standard_normal(config.n_nodes)
    noise[:, 4] = noise_scale * rng.standard_normal(config.n_nodes)
    predictor = states + config.integration_dt * drift + noise
    corrected_drift = paper_epileptor_rhs(
        predictor,
        x0,
        connectivity,
        config,
    )
    return (
        states
        + 0.5 * config.integration_dt * (drift + corrected_drift)
        + noise
    )


def simulate_fixed_x0(
    config: PaperEpileptorConfig,
    x0: np.ndarray,
    connectivity: np.ndarray,
    washout_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate fixed-excitability Epileptors with paper numerical settings."""
    x0 = np.asarray(x0, dtype=float)
    connectivity = np.asarray(connectivity, dtype=float)
    if x0.shape != (config.n_nodes,):
        raise ValueError(f"x0 must have shape ({config.n_nodes},).")
    if connectivity.shape != (config.n_nodes, config.n_nodes):
        raise ValueError(
            "connectivity must have shape "
            f"({config.n_nodes}, {config.n_nodes})."
        )

    rng = np.random.default_rng(config.seed)
    current = default_initial_state(config.n_nodes, rng)
    washout_steps = int(round(washout_s * config.sampling_hz))
    for _ in range(washout_steps):
        current = stochastic_heun_step(
            current,
            x0,
            connectivity,
            config,
            rng,
        )

    output_steps = int(round(config.duration_s * config.sampling_hz)) + 1
    states = np.empty((output_steps, config.n_nodes, 6), dtype=float)
    states[0] = current
    for index in range(1, output_steps):
        current = stochastic_heun_step(
            current,
            x0,
            connectivity,
            config,
            rng,
        )
        if not np.all(np.isfinite(current)):
            raise FloatingPointError("Paper Epileptor became non-finite.")
        states[index] = current

    time_s = np.arange(output_steps, dtype=float) / config.sampling_hz
    return time_s, states


def source_proxy(states: np.ndarray) -> np.ndarray:
    """Return the project visualization proxy x2-x1."""
    return states[..., 3] - states[..., 0]


def detect_local_minimum_onset(
    time_s: np.ndarray,
    proxy: np.ndarray,
    threshold: float = -2.0,
) -> float | None:
    """Return the first local proxy minimum at or below the project threshold."""
    signal = np.asarray(proxy, dtype=float)
    local_minimum = np.r_[
        False,
        (signal[1:-1] <= signal[:-2]) & (signal[1:-1] <= signal[2:]),
        False,
    ]
    hits = np.flatnonzero(local_minimum & (signal <= threshold))
    return None if hits.size == 0 else float(time_s[hits[0]])


def detect_sustained_onset(
    time_s: np.ndarray,
    proxy: np.ndarray,
    threshold: float = -2.0,
    prominence: float = 0.35,
    confirmation_s: float = 2.0,
    minimum_troughs: int = 3,
) -> float | None:
    """Return the first confirmed seizure-like oscillatory onset.

    A threshold crossing alone also labels isolated transients and negative
    fixed-point saturation. The operational event used by the current GT and
    RC analyses therefore requires repeated prominent troughs.
    """
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(proxy, dtype=float)
    if time_s.ndim != 1 or signal.ndim != 1 or len(time_s) != len(signal):
        raise ValueError("time_s and proxy must be one-dimensional and aligned")
    if len(time_s) < 3:
        return None
    sample_interval = float(np.median(np.diff(time_s)))
    if not np.isfinite(sample_interval) or sample_interval <= 0.0:
        raise ValueError("time_s must be strictly increasing")
    confirmation_samples = max(1, int(round(confirmation_s / sample_interval)))
    troughs, _ = find_peaks(
        -signal,
        height=-threshold,
        prominence=prominence,
    )
    for trough in troughs:
        confirmed = np.sum(
            (troughs >= trough) & (troughs <= trough + confirmation_samples)
        )
        if confirmed >= minimum_troughs:
            return float(time_s[trough])
    return None


def detect_sustained_onsets(
    signals: np.ndarray,
    sample_hz: float,
    threshold: float = -2.0,
    prominence: float = 0.35,
    confirmation_s: float = 2.0,
    minimum_troughs: int = 3,
) -> np.ndarray:
    """Vector-shaped wrapper for :func:`detect_sustained_onset`.

    The first axis is time; all remaining axes are independent signals. The
    returned array contains onset times in seconds and NaN for no event.
    """
    values = np.asarray(signals, dtype=float)
    if values.ndim < 2:
        raise ValueError("signals must have shape (time, ...)")
    if sample_hz <= 0.0:
        raise ValueError("sample_hz must be positive")
    time_s = np.arange(values.shape[0], dtype=float) / sample_hz
    events = np.full(values.shape[1:], np.nan, dtype=float)
    for index in np.ndindex(values.shape[1:]):
        onset = detect_sustained_onset(
            time_s,
            values[(slice(None),) + index],
            threshold=threshold,
            prominence=prominence,
            confirmation_s=confirmation_s,
            minimum_troughs=minimum_troughs,
        )
        if onset is not None:
            events[index] = onset
    return events
