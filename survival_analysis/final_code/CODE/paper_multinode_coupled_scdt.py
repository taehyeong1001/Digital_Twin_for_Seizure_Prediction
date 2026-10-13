#!/usr/bin/env python3
"""Static-conditioned Network Epileptor reservoir experiment.

The local dynamics, fixed regional excitability, stochastic integration, and
slow permittivity coupling follow the paper-style implementation in this
project. The controlled condition is the fixed excitability of node 0 (the EZ)
for an entire trial. The experiment supports either one monolithic random
reservoir or one graph-coupled reservoir partitioned into node blocks. Both are
trained only on precritical trajectories. BO can target either normal rollout
quality alone or the broader recruitment-survival objective.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.signal import coherence, correlate, find_peaks, hilbert
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import Ridge

from paper_epileptor_core import (
    PaperEpileptorConfig,
    X0_CRITICAL,
    detect_sustained_onsets,
    source_proxy,
)
from paper_network_epileptor_gt import (
    build_paper_x0_profile,
    build_synthetic_structural_proxy,
)
from paper_single_node_feature_discovery import permutation_entropy
from paper_single_node_scrc import (
    RCParams,
    SparseStaticRC,
    expected_improvement,
    fit_normalizer,
    normalize,
    static_from_x0,
)


# The current synthetic network has a finite-horizon node-0 P(event)=0.5
# boundary near -2.0278. The eight training anchors span a wider empirically
# event-free normal branch while stopping at -2.04, before the stochastic
# transition observed at -2.03 in the unfiltered 160 s audit.
TRAIN_EZ_X0 = np.round(np.linspace(-2.50, -2.04, 8), 2)
# Feature calibration retains dense resolution near the boundary and includes
# every wider normal-branch coordinate used for RC trajectory training.
FEATURE_EZ_X0 = np.unique(
    np.concatenate(
        [TRAIN_EZ_X0, np.round(np.linspace(-2.18, -2.03, 31), 4)]
    )
)
VALIDATION_EZ_X0 = np.asarray(
    [-2.165, -2.125, -2.085, -2.055, -2.035, -2.0275, -2.0175, -1.99]
)
NORMAL_VALIDATION_EZ_X0 = np.asarray(
    [-2.47, -2.40, -2.34, -2.27, -2.20, -2.14, -2.08, -2.04]
)
ROLE_EZ_X0 = {"stable": -2.18, "recruiting": -1.98}
NETWORK_TRAIN_X0_CEILING = -2.035
NETWORK_OPERATIONAL_X0 = -2.0278

# Primary SCDT coordinate. The wide precritical audit selected EZ standard
# deviation as one interpretable scalar; correlated linear feature fits are not
# duplicated as nominally separate reservoir channels.
SCDT_STATIC_FEATURES = ("ez_std",)
SCDT_STATIC_ABLATION = "mean_spatial_dispersion"

STATIC_CANDIDATES = [
    "ez_std",
    "ez_permutation_entropy",
    "mean_node_std",
    "pz_mean_std",
    "ez_ac1",
    "mean_node_ac1",
    "ez_acf_width_s",
    "mean_node_acf_width_s",
    "spatial_mean_std",
    "mean_spatial_dispersion",
    "mean_abs_covariance",
    "max_cov_eigen_fraction",
    "covariance_effective_rank",
    "covariance_participation_ratio",
    "max_node_variance_fraction",
    "node_variance_entropy",
    "ez_pz_variance_log_ratio",
    "mean_abs_correlation",
    "mean_phase_locking_value",
    "ez_pz_phase_locking_value",
    "kuramoto_order_mean",
    "kuramoto_metastability",
    "mean_max_cross_correlation",
    "ez_pz_max_cross_correlation",
    "ez_pz_peak_lag_s",
    "mean_broadband_coherence",
    "ez_pz_broadband_coherence",
    "fc_strength_std",
    "ez_fc_strength",
    "pz_mean_fc_strength",
    "fc_algebraic_connectivity",
    "fc_global_efficiency",
    "fc_weighted_clustering",
    "dynamic_fc_variability",
    "dynamic_fc_persistence",
    "edge_cofluctuation_variability",
    "structural_fc_coupling",
    "structural_graph_total_variation",
    "low_graph_mode_energy_fraction",
    "var1_spectral_radius",
    "lag1_canonical_correlation",
]

FEATURE_GROUPS = {
    "critical_slowing": [
        "ez_std",
        "mean_node_std",
        "pz_mean_std",
        "ez_ac1",
        "mean_node_ac1",
        "ez_acf_width_s",
        "mean_node_acf_width_s",
        "spatial_mean_std",
        "mean_spatial_dispersion",
        "ez_permutation_entropy",
    ],
    "synchrony_and_propagation": [
        "mean_abs_correlation",
        "mean_phase_locking_value",
        "ez_pz_phase_locking_value",
        "kuramoto_order_mean",
        "kuramoto_metastability",
        "mean_max_cross_correlation",
        "ez_pz_max_cross_correlation",
        "ez_pz_peak_lag_s",
        "mean_broadband_coherence",
        "ez_pz_broadband_coherence",
    ],
    "network_geometry_and_stability": [
        "mean_abs_covariance",
        "max_cov_eigen_fraction",
        "covariance_effective_rank",
        "covariance_participation_ratio",
        "max_node_variance_fraction",
        "node_variance_entropy",
        "ez_pz_variance_log_ratio",
        "fc_strength_std",
        "ez_fc_strength",
        "pz_mean_fc_strength",
        "fc_algebraic_connectivity",
        "fc_global_efficiency",
        "fc_weighted_clustering",
        "dynamic_fc_variability",
        "dynamic_fc_persistence",
        "edge_cofluctuation_variability",
        "structural_fc_coupling",
        "structural_graph_total_variation",
        "low_graph_mode_energy_fraction",
        "var1_spectral_radius",
        "lag1_canonical_correlation",
    ],
}

NODE_COLORS = plt.get_cmap("tab10").colors
FEATURE_LABELS = {
    "ez_std": "EZ standard deviation",
    "ez_permutation_entropy": "EZ permutation entropy",
    "mean_node_std": "Mean node standard deviation",
    "pz_mean_std": "PZ mean standard deviation",
    "ez_ac1": "EZ lag-1 autocorrelation",
    "mean_node_ac1": "Mean node lag-1 autocorrelation",
    "ez_acf_width_s": "EZ autocorrelation width",
    "mean_node_acf_width_s": "Mean node autocorrelation width",
    "spatial_mean_std": "Std. of spatially averaged activity",
    "mean_spatial_dispersion": "Mean instantaneous spatial dispersion",
    "mean_abs_covariance": "Mean absolute covariance",
    "max_cov_eigen_fraction": "Dominant covariance fraction",
    "covariance_effective_rank": "Covariance effective rank",
    "covariance_participation_ratio": "Covariance participation ratio",
    "max_node_variance_fraction": "Maximum node variance fraction",
    "node_variance_entropy": "Node variance entropy",
    "ez_pz_variance_log_ratio": "EZ-to-PZ log variance ratio",
    "mean_abs_correlation": "Mean absolute correlation",
    "mean_phase_locking_value": "Mean phase-locking value",
    "ez_pz_phase_locking_value": "EZ-PZ phase-locking value",
    "kuramoto_order_mean": "Mean Kuramoto order",
    "kuramoto_metastability": "Kuramoto metastability",
    "mean_max_cross_correlation": "Mean maximum lagged correlation",
    "ez_pz_max_cross_correlation": "EZ-PZ maximum lagged correlation",
    "ez_pz_peak_lag_s": "EZ-PZ peak-correlation lag",
    "mean_broadband_coherence": "Mean broadband coherence",
    "ez_pz_broadband_coherence": "EZ-PZ broadband coherence",
    "fc_strength_std": "FC node-strength heterogeneity",
    "ez_fc_strength": "EZ functional-connectivity strength",
    "pz_mean_fc_strength": "PZ mean functional-connectivity strength",
    "fc_algebraic_connectivity": "FC algebraic connectivity",
    "fc_global_efficiency": "FC global efficiency",
    "fc_weighted_clustering": "FC weighted clustering",
    "dynamic_fc_variability": "Dynamic FC variability",
    "dynamic_fc_persistence": "Dynamic FC persistence",
    "edge_cofluctuation_variability": "Edge cofluctuation variability",
    "structural_fc_coupling": "Structural-functional coupling",
    "structural_graph_total_variation": "Structural graph total variation",
    "low_graph_mode_energy_fraction": "Low graph-mode energy fraction",
    "var1_spectral_radius": "Observed VAR(1) spectral radius",
    "lag1_canonical_correlation": "Lag-1 canonical correlation",
}


@dataclass
class NetworkSwitchCohort:
    ez_x0: np.ndarray
    warmups_raw: np.ndarray
    futures_raw: np.ndarray
    gt_events: np.ndarray


@dataclass(frozen=True)
class CoupledRCParams(RCParams):
    """Hyperparameters for one graph-coupled block reservoir."""

    block_size: int = 200
    block_coupling: float = 0.20
    cross_density_ratio: float = 0.25


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_x0_profiles(ez_x0: np.ndarray, n_nodes: int = 8) -> np.ndarray:
    """Return fixed network profiles with only node-0 excitability varied."""
    template, _ = build_paper_x0_profile(n_nodes)
    profiles = np.repeat(template[None, :], len(ez_x0), axis=0)
    profiles[:, 0] = np.asarray(ez_x0, dtype=float)
    return profiles


def initial_state_batch(
    batch: int,
    n_nodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base = np.asarray([-1.55, -11.8, 3.05, -0.82, 0.0, 0.0])
    scale = np.asarray([0.03, 0.20, 0.04, 0.02, 0.02, 0.03])
    states = np.broadcast_to(base, (batch, n_nodes, 6)).copy()
    states += rng.normal(0.0, scale, size=states.shape)
    return states


def network_rhs_batch(
    states: np.ndarray,
    x0: np.ndarray,
    connectivity: np.ndarray,
    config: PaperEpileptorConfig,
) -> np.ndarray:
    """Vectorized Eq. (4) for independent batches of the same network."""
    x1, y1, z, x2, y2, lowpass = np.moveaxis(states, -1, 0)
    f1 = np.where(
        x1 < 0.0,
        x1**3 - 3.0 * x1**2,
        (x2 - 0.6 * (z - 4.0) ** 2) * x1,
    )
    f2 = np.where(x2 < -0.25, 0.0, 6.0 * (x2 + 0.25))
    difference = x1 @ connectivity.T - x1 * connectivity.sum(axis=1)

    derivative = np.empty_like(states)
    derivative[..., 0] = y1 - f1 - z + config.i1
    derivative[..., 1] = 1.0 - 5.0 * x1**2 - y1
    derivative[..., 2] = (
        4.0 * (x1 - x0)
        - z
        - config.global_coupling * difference
    ) / config.tau0
    derivative[..., 3] = (
        -y2
        + x2
        - x2**3
        + config.i2
        + 0.002 * lowpass
        - 0.3 * (z - 3.5)
    )
    derivative[..., 4] = (-y2 + f2) / config.tau2
    derivative[..., 5] = x1 - config.gamma * lowpass
    return derivative


def network_heun_step_batch(
    states: np.ndarray,
    x0: np.ndarray,
    connectivity: np.ndarray,
    config: PaperEpileptorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    drift = network_rhs_batch(states, x0, connectivity, config)
    noise = np.zeros_like(states)
    scale = np.sqrt(config.noise_variance * config.integration_dt)
    noise[..., 3] = scale * rng.standard_normal(states.shape[:2])
    noise[..., 4] = scale * rng.standard_normal(states.shape[:2])
    predictor = states + config.integration_dt * drift + noise
    corrected = network_rhs_batch(predictor, x0, connectivity, config)
    return states + 0.5 * config.integration_dt * (drift + corrected) + noise


def simulate_fixed_network_batch(
    ez_x0: np.ndarray,
    duration_s: float,
    rc_stride: int,
    seed: int,
    global_coupling: float,
    noise_variance: float,
) -> np.ndarray:
    values = np.asarray(ez_x0, dtype=float)
    profiles = make_x0_profiles(values)
    n_nodes = profiles.shape[1]
    config = PaperEpileptorConfig(
        n_nodes=n_nodes,
        duration_s=duration_s,
        global_coupling=global_coupling,
        noise_variance=noise_variance,
        seed=seed,
    )
    connectivity = build_synthetic_structural_proxy(n_nodes)
    rng = np.random.default_rng(seed)
    states = initial_state_batch(len(values), n_nodes, rng)
    steps = int(round(duration_s * config.sampling_hz))
    records = np.empty(((steps + rc_stride - 1) // rc_stride, len(values), n_nodes))
    output = 0
    for index in range(steps):
        states = network_heun_step_batch(states, profiles, connectivity, config, rng)
        if index % rc_stride == 0:
            records[output] = source_proxy(states)
            output += 1
    return records[:output]


def simulate_network_switch_batch(
    target_ez_x0: np.ndarray,
    base_ez_x0: float,
    warmup_s: float,
    rollout_s: float,
    rc_stride: int,
    seed: int,
    global_coupling: float,
    noise_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(target_ez_x0, dtype=float)
    target_profiles = make_x0_profiles(targets)
    base_profiles = make_x0_profiles(np.full(len(targets), base_ez_x0))
    n_nodes = target_profiles.shape[1]
    config = PaperEpileptorConfig(
        n_nodes=n_nodes,
        duration_s=warmup_s + rollout_s,
        global_coupling=global_coupling,
        noise_variance=noise_variance,
        seed=seed,
    )
    connectivity = build_synthetic_structural_proxy(n_nodes)
    rng = np.random.default_rng(seed)
    states = initial_state_batch(len(targets), n_nodes, rng)

    def run_segment(profiles: np.ndarray, duration_s: float) -> np.ndarray:
        nonlocal states
        steps = int(round(duration_s * config.sampling_hz))
        records = np.empty(
            ((steps + rc_stride - 1) // rc_stride, len(targets), n_nodes),
            dtype=float,
        )
        output = 0
        for index in range(steps):
            states = network_heun_step_batch(
                states,
                profiles,
                connectivity,
                config,
                rng,
            )
            if index % rc_stride == 0:
                records[output] = source_proxy(states)
                output += 1
        return records[:output]

    return run_segment(base_profiles, warmup_s), run_segment(
        target_profiles,
        rollout_s,
    )


def event_times(signals: np.ndarray, sample_hz: float) -> np.ndarray:
    """Detect the shared sustained seizure-like event definition."""
    return detect_sustained_onsets(signals, sample_hz)


def lag1_autocorrelation(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=float) - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(centered[:-1], centered[1:]) / denominator)


def autocorrelation_width(
    values: np.ndarray,
    sample_hz: float,
    max_lag_s: float = 4.0,
) -> float:
    """Return the first 1/e ACF crossing, capped at max_lag_s."""
    centered = np.asarray(values, dtype=float) - np.mean(values)
    maximum_lag = min(len(centered) - 1, int(round(max_lag_s * sample_hz)))
    if maximum_lag < 1 or np.dot(centered, centered) <= 1e-15:
        return 0.0
    acf = correlate(centered, centered, mode="full", method="fft")
    acf = acf[len(centered) - 1 : len(centered) + maximum_lag]
    acf /= max(float(acf[0]), 1e-15)
    crossings = np.flatnonzero(acf <= np.exp(-1.0))
    lag = int(crossings[0]) if len(crossings) else maximum_lag
    return float(lag / sample_hz)


def maximum_lagged_correlation(
    first: np.ndarray,
    second: np.ndarray,
    sample_hz: float,
    max_lag_s: float = 2.0,
) -> tuple[float, float]:
    """Return maximum absolute cross-correlation and its absolute lag."""
    first = np.asarray(first, dtype=float) - np.mean(first)
    second = np.asarray(second, dtype=float) - np.mean(second)
    scale = float(np.std(first) * np.std(second))
    if scale <= 1e-15:
        return 0.0, 0.0
    full = correlate(first, second, mode="full", method="fft")
    lags = np.arange(-len(second) + 1, len(first))
    maximum_lag = int(round(max_lag_s * sample_hz))
    keep = np.abs(lags) <= maximum_lag
    selected_lags = lags[keep]
    overlap = np.minimum(len(first), len(second)) - np.abs(selected_lags)
    normalized = full[keep] / np.maximum(overlap * scale, 1e-15)
    peak = int(np.argmax(np.abs(normalized)))
    return float(abs(normalized[peak])), float(abs(selected_lags[peak]) / sample_hz)


def functional_graph_features(adjacency: np.ndarray) -> dict[str, float]:
    """Weighted graph summaries of the absolute correlation network."""
    adjacency = np.asarray(adjacency, dtype=float).copy()
    np.fill_diagonal(adjacency, 0.0)
    n_nodes = adjacency.shape[0]
    strengths = np.sum(adjacency, axis=1) / max(n_nodes - 1, 1)
    laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
    laplacian_eigenvalues = np.linalg.eigvalsh(laplacian)
    algebraic = float(laplacian_eigenvalues[1] / max(n_nodes, 1))

    distances = np.full_like(adjacency, np.inf)
    connected = adjacency > 1e-9
    distances[connected] = 1.0 / adjacency[connected]
    np.fill_diagonal(distances, 0.0)
    geodesic = shortest_path(distances, directed=False)
    mask = ~np.eye(n_nodes, dtype=bool)
    efficiency = np.zeros_like(geodesic)
    valid = np.isfinite(geodesic) & (geodesic > 0.0)
    efficiency[valid] = 1.0 / geodesic[valid]

    maximum = float(np.max(adjacency))
    normalized = adjacency / maximum if maximum > 1e-12 else adjacency
    root = np.cbrt(normalized)
    triangles = np.diag(root @ root @ root)
    degrees = np.sum(adjacency > 1e-9, axis=1)
    denominator = degrees * np.maximum(degrees - 1, 1)
    clustering = np.divide(
        triangles,
        denominator,
        out=np.zeros_like(triangles),
        where=denominator > 0,
    )
    return {
        "fc_strength_std": float(np.std(strengths)),
        "ez_fc_strength": float(strengths[0]),
        "pz_mean_fc_strength": float(np.mean(strengths[1:4])),
        "fc_algebraic_connectivity": algebraic,
        "fc_global_efficiency": float(np.mean(efficiency[mask])),
        "fc_weighted_clustering": float(np.mean(clustering)),
    }


def dynamic_connectivity_features(
    sequence: np.ndarray,
    sample_hz: float,
    window_s: float = 8.0,
) -> tuple[float, float]:
    """Variability and persistence of sliding-window functional connectivity."""
    window = min(len(sequence), max(16, int(round(window_s * sample_hz))))
    step = max(1, window // 2)
    mask = np.triu(np.ones((sequence.shape[1], sequence.shape[1]), dtype=bool), 1)
    edge_vectors: list[np.ndarray] = []
    mean_connections: list[float] = []
    for start in range(0, len(sequence) - window + 1, step):
        correlation = np.corrcoef(sequence[start : start + window], rowvar=False)
        edges = np.nan_to_num(correlation[mask], nan=0.0)
        edge_vectors.append(edges)
        mean_connections.append(float(np.mean(np.abs(edges))))
    if len(edge_vectors) < 2:
        return 0.0, 0.0
    persistence = []
    for previous, current in zip(edge_vectors[:-1], edge_vectors[1:]):
        if np.std(previous) <= 1e-12 or np.std(current) <= 1e-12:
            persistence.append(0.0)
        else:
            persistence.append(float(np.corrcoef(previous, current)[0, 1]))
    return float(np.std(mean_connections)), float(np.mean(persistence))


def network_features(sequence: np.ndarray, sample_hz: float) -> dict[str, float]:
    """Observable source-level trajectory statistics for one network."""
    sequence = np.asarray(sequence, dtype=float)
    n_nodes = sequence.shape[1]
    node_std = np.std(sequence, axis=0)
    covariance = np.cov(sequence, rowvar=False)
    correlation = np.corrcoef(sequence, rowvar=False)
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~np.eye(n_nodes, dtype=bool)
    upper_mask = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    trace = float(np.trace(covariance))
    normalized_eigenvalues = eigenvalues / max(float(np.sum(eigenvalues)), 1e-12)
    positive = normalized_eigenvalues > 1e-15
    covariance_entropy = -float(
        np.sum(normalized_eigenvalues[positive] * np.log(normalized_eigenvalues[positive]))
    )
    node_variances = np.maximum(node_std**2, 1e-15)
    node_variance_fraction = node_variances / np.sum(node_variances)
    variance_entropy = -float(
        np.sum(node_variance_fraction * np.log(node_variance_fraction))
        / np.log(n_nodes)
    )

    structural = build_synthetic_structural_proxy(n_nodes)
    structural /= max(float(np.max(structural)), 1e-12)
    structural_laplacian = np.diag(np.sum(structural, axis=1)) - structural
    _, graph_modes = np.linalg.eigh(structural_laplacian)
    graph_signal = sequence - np.mean(sequence, axis=0)
    graph_numerator = np.einsum(
        "ti,ij,tj->t",
        graph_signal,
        structural_laplacian,
        graph_signal,
    )
    graph_denominator = np.sum(graph_signal**2, axis=1)
    graph_total_variation = float(
        np.mean(graph_numerator / np.maximum(graph_denominator, 1e-12))
    )
    graph_mode_energy = np.sum((graph_signal @ graph_modes) ** 2, axis=0)
    low_graph_mode_fraction = float(
        np.sum(graph_mode_energy[:2]) / max(float(np.sum(graph_mode_energy)), 1e-12)
    )
    structural_edges = structural[upper_mask]
    functional_edges = np.abs(correlation[upper_mask])
    if np.std(structural_edges) <= 1e-12 or np.std(functional_edges) <= 1e-12:
        structural_fc_coupling = 0.0
    else:
        structural_fc_coupling = float(
            np.corrcoef(structural_edges, functional_edges)[0, 1]
        )

    ac1 = np.asarray([lag1_autocorrelation(sequence[:, node]) for node in range(n_nodes)])
    acf_width = np.asarray(
        [autocorrelation_width(sequence[:, node], sample_hz) for node in range(n_nodes)]
    )

    analytic = hilbert(sequence - np.mean(sequence, axis=0), axis=0)
    phase = np.angle(analytic)
    phase_difference = phase[:, :, None] - phase[:, None, :]
    phase_locking = np.abs(np.mean(np.exp(1j * phase_difference), axis=0))
    kuramoto = np.abs(np.mean(np.exp(1j * phase), axis=1))

    pair_correlations = []
    pair_lags = []
    pair_coherences = []
    ez_pz_correlations = []
    ez_pz_lags = []
    ez_pz_coherences = []
    nperseg = min(256, len(sequence))
    for first in range(n_nodes):
        for second in range(first + 1, n_nodes):
            peak_correlation, peak_lag = maximum_lagged_correlation(
                sequence[:, first],
                sequence[:, second],
                sample_hz,
            )
            frequencies, coherence_values = coherence(
                sequence[:, first],
                sequence[:, second],
                fs=sample_hz,
                nperseg=nperseg,
                noverlap=nperseg // 2,
            )
            frequency_mask = (frequencies >= 0.25) & (frequencies <= 0.95 * sample_hz / 2.0)
            mean_coherence = float(np.mean(coherence_values[frequency_mask]))
            pair_correlations.append(peak_correlation)
            pair_lags.append(peak_lag)
            pair_coherences.append(mean_coherence)
            if first == 0 and 1 <= second <= 3:
                ez_pz_correlations.append(peak_correlation)
                ez_pz_lags.append(peak_lag)
                ez_pz_coherences.append(mean_coherence)

    standardized = (sequence - np.mean(sequence, axis=0)) / np.maximum(node_std, 1e-9)
    edge_products = standardized[:, :, None] * standardized[:, None, :]
    cofluctuation_amplitude = np.sqrt(
        np.mean(edge_products[:, upper_mask] ** 2, axis=1)
    )
    dynamic_variability, dynamic_persistence = dynamic_connectivity_features(
        sequence,
        sample_hz,
    )

    predictors = standardized[:-1]
    targets = standardized[1:]
    gram = predictors.T @ predictors
    ridge = 1e-3 * max(float(np.trace(gram)) / n_nodes, 1.0)
    var_coefficients = np.linalg.solve(
        gram + ridge * np.eye(n_nodes),
        predictors.T @ targets,
    )
    var_radius = float(np.max(np.abs(np.linalg.eigvals(var_coefficients))))

    c00 = predictors.T @ predictors / len(predictors) + 1e-5 * np.eye(n_nodes)
    c11 = targets.T @ targets / len(targets) + 1e-5 * np.eye(n_nodes)
    c01 = predictors.T @ targets / len(predictors)
    eval0, evec0 = np.linalg.eigh(c00)
    eval1, evec1 = np.linalg.eigh(c11)
    invsqrt0 = (evec0 * (1.0 / np.sqrt(np.maximum(eval0, 1e-9)))) @ evec0.T
    invsqrt1 = (evec1 * (1.0 / np.sqrt(np.maximum(eval1, 1e-9)))) @ evec1.T
    canonical = float(np.linalg.svd(invsqrt0 @ c01 @ invsqrt1, compute_uv=False)[0])

    downsample = max(1, int(round(sample_hz / 16.0)))
    features = {
        "ez_std": float(node_std[0]),
        "ez_permutation_entropy": permutation_entropy(sequence[::downsample, 0]),
        "mean_node_std": float(np.mean(node_std)),
        "pz_mean_std": float(np.mean(node_std[1:4])),
        "ez_ac1": float(ac1[0]),
        "mean_node_ac1": float(np.mean(ac1)),
        "ez_acf_width_s": float(acf_width[0]),
        "mean_node_acf_width_s": float(np.mean(acf_width)),
        "spatial_mean_std": float(np.std(np.mean(sequence, axis=1))),
        "mean_spatial_dispersion": float(np.mean(np.std(sequence, axis=1))),
        "mean_abs_covariance": float(np.mean(np.abs(covariance[mask]))),
        "max_cov_eigen_fraction": float(np.max(eigenvalues) / max(trace, 1e-12)),
        "covariance_effective_rank": float(np.exp(covariance_entropy)),
        "covariance_participation_ratio": float(
            np.sum(eigenvalues) ** 2 / max(float(np.sum(eigenvalues**2)), 1e-12)
        ),
        "max_node_variance_fraction": float(np.max(node_variance_fraction)),
        "node_variance_entropy": variance_entropy,
        "ez_pz_variance_log_ratio": float(
            np.log(node_variances[0] / np.mean(node_variances[1:4]))
        ),
        "mean_abs_correlation": float(np.mean(np.abs(correlation[mask]))),
        "mean_phase_locking_value": float(np.mean(phase_locking[upper_mask])),
        "ez_pz_phase_locking_value": float(np.mean(phase_locking[0, 1:4])),
        "kuramoto_order_mean": float(np.mean(kuramoto)),
        "kuramoto_metastability": float(np.std(kuramoto)),
        "mean_max_cross_correlation": float(np.mean(pair_correlations)),
        "ez_pz_max_cross_correlation": float(np.mean(ez_pz_correlations)),
        "ez_pz_peak_lag_s": float(np.mean(ez_pz_lags)),
        "mean_broadband_coherence": float(np.mean(pair_coherences)),
        "ez_pz_broadband_coherence": float(np.mean(ez_pz_coherences)),
        "dynamic_fc_variability": dynamic_variability,
        "dynamic_fc_persistence": dynamic_persistence,
        "edge_cofluctuation_variability": float(np.std(cofluctuation_amplitude)),
        "structural_fc_coupling": structural_fc_coupling,
        "structural_graph_total_variation": graph_total_variation,
        "low_graph_mode_energy_fraction": low_graph_mode_fraction,
        "var1_spectral_radius": var_radius,
        "lag1_canonical_correlation": canonical,
    }
    features.update(functional_graph_features(np.abs(correlation)))
    return features


def stable_long_networks(
    ez_x0: float,
    duration_s: float,
    requested: int,
    seed: int,
    rc_stride: int,
    global_coupling: float,
    noise_variance: float,
) -> tuple[list[np.ndarray], list[int]]:
    batch = max(requested * 3, requested + 2)
    trajectories = simulate_fixed_network_batch(
        np.full(batch, ez_x0),
        duration_s,
        rc_stride,
        seed,
        global_coupling,
        noise_variance,
    )
    sample_hz = 256.0 / rc_stride
    events = event_times(trajectories, sample_hz)
    selected: list[np.ndarray] = []
    indices: list[int] = []
    for index in range(batch):
        if not np.any(np.isfinite(events[index])):
            selected.append(trajectories[:, index].copy())
            indices.append(index)
        if len(selected) == requested:
            break
    if len(selected) < requested:
        raise RuntimeError(
            f"Only {len(selected)} stable networks at EZ x0={ez_x0}; "
            f"requested {requested}."
        )
    return selected, indices


def build_feature_table(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    sample_hz = 256.0 / args.rc_stride
    for condition, ez_x0 in enumerate(FEATURE_EZ_X0):
        trajectories, source_indices = stable_long_networks(
            float(ez_x0),
            args.long_trajectory_s,
            args.feature_seeds,
            args.feature_seed_base + 1000 * condition,
            args.rc_stride,
            args.global_coupling,
            args.noise_variance,
        )
        for replicate, (sequence, source_index) in enumerate(
            zip(trajectories, source_indices)
        ):
            rows.append(
                {
                    "ez_x0": float(ez_x0),
                    "replicate": replicate,
                    "source_index": source_index,
                    **network_features(sequence, sample_hz),
                }
            )
        print(
            f"feature calibration EZ x0={ez_x0:.3f}: "
            f"{len(trajectories)} stable networks",
            flush=True,
        )
    return pd.DataFrame(rows)


def rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = frame.groupby("ez_x0")
    coordinates = np.asarray(sorted(frame["ez_x0"].unique()))
    for name in STATIC_CANDIDATES:
        means = grouped[name].mean().reindex(coordinates).to_numpy(dtype=float)
        within = grouped[name].std().reindex(coordinates).fillna(0.0).to_numpy(dtype=float)
        if not np.all(np.isfinite(means)) or np.ptp(means) <= 1e-12:
            continue
        slope, intercept = np.polyfit(coordinates, means, 1)
        fitted = slope * coordinates + intercept
        total = np.sum((means - np.mean(means)) ** 2)
        r2 = 1.0 - np.sum((means - fitted) ** 2) / max(total, 1e-15)
        association = spearmanr(coordinates, means)
        rho = float(association.statistic)
        p_value = float(association.pvalue)
        direction = np.sign(slope) if slope != 0.0 else 1.0
        monotonicity = float(np.mean(direction * np.diff(means) >= 0.0))
        signal_to_noise = float(
            np.ptp(means) / max(np.sqrt(np.mean(within**2)), 1e-12)
        )
        score = (
            0.32 * abs(rho)
            + 0.28 * np.clip(r2, 0.0, 1.0)
            + 0.22 * monotonicity
            + 0.18 * np.tanh(signal_to_noise / 4.0)
        )
        rows.append(
            {
                "feature": name,
                "score": score,
                "spearman_rho": rho,
                "spearman_p": p_value,
                "linear_r2": r2,
                "monotonic_fraction": monotonicity,
                "range_to_within_std": signal_to_noise,
                "slope": slope,
            }
        )
    ranking = pd.DataFrame(rows)
    p_values = ranking["spearman_p"].to_numpy(dtype=float)
    order = np.argsort(p_values)
    ordered_q = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    q_values = np.empty_like(ordered_q)
    q_values[order] = np.clip(ordered_q, 0.0, 1.0)
    ranking["spearman_fdr_q"] = q_values
    return ranking.sort_values("score", ascending=False).reset_index(drop=True)


def select_features(
    ranking: pd.DataFrame,
    frame: pd.DataFrame,
    count: int,
    correlation_limit: float,
) -> list[str]:
    correlation = frame[STATIC_CANDIDATES].corr(method="spearman").abs()
    selected: list[str] = []
    for name in ranking["feature"]:
        if any(correlation.loc[name, other] > correlation_limit for other in selected):
            continue
        selected.append(name)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError("Could not select enough nonredundant network features")
    return selected


def fit_static_mapping(
    frame: pd.DataFrame,
    selected: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = (
        frame.groupby("ez_x0")[selected]
        .mean()
        .reindex(TRAIN_EZ_X0)
        .to_numpy(dtype=float)
    )
    coefficients = np.vstack(
        [
            np.polyfit(TRAIN_EZ_X0, observed[:, feature], 1)
            for feature in range(len(selected))
        ]
    )
    slopes = coefficients[:, 0]
    if np.any(slopes <= 0.0):
        invalid = [name for name, slope in zip(selected, slopes) if slope <= 0.0]
        raise RuntimeError(
            "The primary SCDT coordinate must increase toward the boundary; "
            f"non-increasing features: {invalid}"
        )
    fitted = static_from_x0(TRAIN_EZ_X0, coefficients)
    return observed, fitted, coefficients


def build_training_sequences(
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[float], list[dict]]:
    sequences: list[np.ndarray] = []
    coordinates: list[float] = []
    rows: list[dict] = []
    for condition, ez_x0 in enumerate(TRAIN_EZ_X0):
        trajectories, indices = stable_long_networks(
            float(ez_x0),
            args.long_trajectory_s,
            args.train_seeds,
            args.train_seed_base + 1000 * condition,
            args.rc_stride,
            args.global_coupling,
            args.noise_variance,
        )
        for replicate, (sequence, source_index) in enumerate(zip(trajectories, indices)):
            sequences.append(sequence)
            coordinates.append(float(ez_x0))
            rows.append(
                {
                    "ez_x0": float(ez_x0),
                    "replicate": replicate,
                    "source_index": source_index,
                    "samples": len(sequence),
                }
            )
    return sequences, coordinates, rows


def build_switch_cohort(
    ez_x0_values: np.ndarray,
    trajectories_per_x0: int,
    args: argparse.Namespace,
    seed_base: int,
) -> NetworkSwitchCohort:
    coordinates = np.repeat(np.asarray(ez_x0_values, dtype=float), trajectories_per_x0)
    warmups, futures = simulate_network_switch_batch(
        coordinates,
        args.base_ez_x0,
        args.warmup_s,
        args.rollout_s,
        args.rc_stride,
        seed_base,
        args.global_coupling,
        args.noise_variance,
    )
    events = event_times(futures, 256.0 / args.rc_stride)
    for value in ez_x0_values:
        selection = np.isclose(coordinates, value)
        local_survival = np.mean(~np.isfinite(events[selection, 0]))
        network_survival = np.mean(
            np.sum(np.isfinite(events[selection, :4]), axis=1)
            < args.network_event_nodes
        )
        print(
            f"switch EZ x0={value:.4f}: GT node0/network survival="
            f"{local_survival:.2f}/{network_survival:.2f}",
            flush=True,
        )
    return NetworkSwitchCohort(
        ez_x0=coordinates,
        warmups_raw=warmups,
        futures_raw=futures,
        gt_events=events,
    )


class MultiOutputStaticRC:
    """One monolithic sparse reservoir with a joint eight-channel readout."""

    def __init__(
        self,
        params: RCParams,
        dynamic_dim: int,
        static_dim: int,
        readout_stride: int = 4,
        readout_size: int = 800,
    ):
        self.params = params
        self.dynamic_dim = int(dynamic_dim)
        self.static_dim = int(static_dim)
        self.readout_stride = max(1, int(readout_stride))
        rng = np.random.default_rng(params.seed)
        self.readout_size = min(params.n_reservoir, max(1, int(readout_size)))
        self.readout_indices = np.sort(
            rng.choice(params.n_reservoir, size=self.readout_size, replace=False)
        )
        self.w_in = (
            rng.uniform(-1.0, 1.0, (params.n_reservoir, self.dynamic_dim))
            * params.input_scaling
        )
        self.w_static = (
            rng.uniform(-1.0, 1.0, (params.n_reservoir, self.static_dim))
            * params.static_scaling
        )
        nnz = max(1, int(params.density * params.n_reservoir**2))
        rows = rng.integers(0, params.n_reservoir, size=nnz)
        columns = rng.integers(0, params.n_reservoir, size=nnz)
        values = rng.normal(0.0, 1.0, size=nnz)
        matrix = sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(params.n_reservoir, params.n_reservoir),
        ).tocsr()
        try:
            radius = float(
                np.max(
                    np.abs(
                        eigs(
                            matrix,
                            k=1,
                            which="LM",
                            return_eigenvectors=False,
                            v0=np.ones(params.n_reservoir),
                        )
                    )
                )
            )
        except Exception:
            radius = float(np.max(np.abs(np.linalg.eigvals(matrix.toarray()))))
        self.a = matrix * (params.spectral_radius / max(radius, 1e-9))
        self.w_out: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.clip_low = np.full(self.dynamic_dim, -5.0)
        self.clip_high = np.full(self.dynamic_dim, 5.0)

    def update(
        self,
        reservoir: np.ndarray,
        dynamic: np.ndarray,
        static: np.ndarray,
    ) -> np.ndarray:
        preactivation = (
            self.a @ reservoir
            + self.w_in @ dynamic
            + self.w_static @ static
        )
        return (
            (1.0 - self.params.leakage_rate) * reservoir
            + self.params.leakage_rate * np.tanh(preactivation)
        )

    def augment(self, reservoir: np.ndarray, dynamic: np.ndarray) -> np.ndarray:
        transformed = reservoir[self.readout_indices].copy()
        transformed[1::2] = transformed[1::2] ** 2
        if transformed.ndim == 1:
            return np.r_[transformed, dynamic, 1.0]
        return np.vstack(
            [transformed, dynamic, np.ones((1, transformed.shape[1]))]
        )

    def fit(
        self,
        sequences: list[np.ndarray],
        statics: list[np.ndarray],
        warmup: int,
    ) -> None:
        all_values = np.concatenate(sequences, axis=0)
        span = np.ptp(all_values, axis=0)
        self.clip_low = np.min(all_values, axis=0) - 2.5 * span
        self.clip_high = np.max(all_values, axis=0) + 1.2 * span
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) != 1:
            raise ValueError("global reservoir batching requires equal sequence lengths")
        sequence_batch = np.stack(sequences, axis=1)
        static_batch = np.stack(statics, axis=0).T
        batch = len(sequences)
        sampled_steps = len(
            range(warmup, len(sequence_batch) - 1, self.readout_stride)
        )
        columns = sampled_steps * batch
        features = np.empty(
            (columns, self.readout_size + self.dynamic_dim + 1),
            dtype=np.float32,
        )
        targets = np.empty((columns, self.dynamic_dim), dtype=np.float32)
        column = 0
        reservoir = np.zeros((self.params.n_reservoir, batch))
        for index in range(len(sequence_batch) - 1):
            reservoir = self.update(
                reservoir,
                sequence_batch[index].T,
                static_batch,
            )
            if index >= warmup and (index - warmup) % self.readout_stride == 0:
                next_column = column + batch
                features[column:next_column] = self.augment(
                    reservoir,
                    sequence_batch[index].T,
                ).T
                targets[column:next_column] = sequence_batch[index + 1]
                column = next_column
        model = Ridge(
            alpha=self.params.ridge,
            fit_intercept=False,
            solver="cholesky",
        )
        model.fit(features, targets)
        self.w_out = np.asarray(model.coef_, dtype=float)
        residuals = targets - model.predict(features)
        lower = np.quantile(residuals, 0.0025, axis=0)
        upper = np.quantile(residuals, 0.9975, axis=0)
        self.residuals = np.clip(residuals, lower, upper)

    def rollout_switch_batch(
        self,
        warmup: np.ndarray,
        warmup_statics: np.ndarray,
        target_statics: np.ndarray,
        n_steps: int,
        seed: int,
        ramp_steps: int = 0,
    ) -> np.ndarray:
        if self.w_out is None or self.residuals is None:
            raise RuntimeError("RC must be fitted before rollout")
        if warmup.ndim != 3:
            raise ValueError("warmup must have shape (time, batch, dynamic_dim)")
        batch = warmup.shape[1]
        reservoir = np.zeros((self.params.n_reservoir, batch))
        warmup_static = warmup_statics.T
        target_static = target_statics.T
        for index in range(len(warmup)):
            reservoir = self.update(reservoir, warmup[index].T, warmup_static)
        current = warmup[-1].T
        prediction = np.empty((n_steps, batch, self.dynamic_dim), dtype=float)
        rng = np.random.default_rng(seed)
        for index in range(n_steps):
            if ramp_steps > 0 and index < ramp_steps:
                fraction = (index + 1) / ramp_steps
                static = (1.0 - fraction) * warmup_static + fraction * target_static
            else:
                static = target_static
            reservoir = self.update(reservoir, current, static)
            current = self.w_out @ self.augment(reservoir, current)
            residual_index = rng.integers(0, len(self.residuals), size=batch)
            innovations = self.residuals[residual_index].T
            current += self.params.innovation_scaling * innovations
            current = np.clip(
                current,
                self.clip_low[:, None],
                self.clip_high[:, None],
            )
            prediction[index] = current.T
        return prediction


class IndependentNodeStaticRC:
    """Eight scalar autonomous readouts sharing one network static context.

    A joint eight-output autonomous readout readily converges to a clipped
    fixed point because small errors from every channel are fed back together.
    Node-wise reservoirs preserve the proven scalar SCDT rollout while the
    shared static vector still carries EZ and network-level information.
    """

    def __init__(self, params: RCParams, dynamic_dim: int, static_dim: int):
        self.dynamic_dim = int(dynamic_dim)
        self.models = [
            SparseStaticRC(
                replace(params, seed=params.seed + 1009 * node),
                static_dim=static_dim,
            )
            for node in range(self.dynamic_dim)
        ]

    def fit(
        self,
        sequences: list[np.ndarray],
        statics: list[np.ndarray],
        warmup: int,
    ) -> None:
        for node, model in enumerate(self.models):
            model.fit(
                [sequence[:, node] for sequence in sequences],
                statics,
                warmup,
            )

    def rollout_switch_batch(
        self,
        warmup: np.ndarray,
        warmup_statics: np.ndarray,
        target_statics: np.ndarray,
        n_steps: int,
        seed: int,
        ramp_steps: int = 0,
    ) -> np.ndarray:
        prediction = np.empty(
            (n_steps, warmup.shape[1], self.dynamic_dim),
            dtype=float,
        )
        for node, model in enumerate(self.models):
            prediction[:, :, node] = model.rollout_switch_batch(
                warmup[:, :, node],
                warmup_statics,
                target_statics,
                n_steps,
                seed + 1009 * node,
                ramp_steps,
            )
        return prediction


class GraphCoupledBlockRC:
    """One sparse reservoir with one recurrent block per Epileptor node.

    Local recurrent edges model node-internal memory. Inter-block edges exist
    only where the structural proxy has a nonzero connection. Each output uses
    its local block state plus the current eight-channel signal; information
    from other nodes reaches that block through the recurrent graph coupling.
    """

    def __init__(
        self,
        params: CoupledRCParams,
        dynamic_dim: int,
        static_dim: int,
    ):
        self.params = params
        self.dynamic_dim = int(dynamic_dim)
        self.static_dim = int(static_dim)
        self.block_size = int(params.block_size)
        if self.dynamic_dim != 8:
            raise ValueError("The current graph-coupled reservoir expects 8 nodes")
        expected = self.block_size * self.dynamic_dim
        if params.n_reservoir != expected:
            raise ValueError(
                f"n_reservoir must equal block_size * dynamic_dim ({expected})"
            )

        rng = np.random.default_rng(params.seed)
        self.w_in = np.zeros((params.n_reservoir, self.dynamic_dim))
        for node in range(self.dynamic_dim):
            block = self.block_slice(node)
            self.w_in[block, node] = (
                rng.uniform(-1.0, 1.0, self.block_size) * params.input_scaling
            )
        self.w_static = (
            rng.uniform(-1.0, 1.0, (params.n_reservoir, self.static_dim))
            * params.static_scaling
        )
        self.a = self.build_recurrent_matrix(rng)
        feature_dim = self.block_size + self.dynamic_dim + 1
        self.w_out = np.zeros((self.dynamic_dim, feature_dim))
        self.residuals: np.ndarray | None = None
        self.clip_low = np.full(self.dynamic_dim, -5.0)
        self.clip_high = np.full(self.dynamic_dim, 5.0)

    def block_slice(self, node: int) -> slice:
        start = node * self.block_size
        return slice(start, start + self.block_size)

    def build_recurrent_matrix(
        self,
        rng: np.random.Generator,
    ) -> sparse.csr_matrix:
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        local_nnz = max(1, int(self.params.density * self.block_size**2))
        for node in range(self.dynamic_dim):
            offset = node * self.block_size
            rows.append(offset + rng.integers(0, self.block_size, local_nnz))
            columns.append(offset + rng.integers(0, self.block_size, local_nnz))
            values.append(rng.normal(0.0, 1.0, local_nnz))

        connectivity = build_synthetic_structural_proxy(self.dynamic_dim)
        connectivity /= max(float(np.max(connectivity)), 1e-12)
        cross_density = self.params.density * self.params.cross_density_ratio
        cross_nnz = max(1, int(cross_density * self.block_size**2))
        for target in range(self.dynamic_dim):
            for source in range(self.dynamic_dim):
                weight = float(connectivity[target, source])
                if target == source or weight <= 0.0:
                    continue
                rows.append(
                    target * self.block_size
                    + rng.integers(0, self.block_size, cross_nnz)
                )
                columns.append(
                    source * self.block_size
                    + rng.integers(0, self.block_size, cross_nnz)
                )
                values.append(
                    rng.normal(0.0, 1.0, cross_nnz)
                    * self.params.block_coupling
                    * weight
                )

        matrix = sparse.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(self.params.n_reservoir, self.params.n_reservoir),
        ).tocsr()
        radius = float(
            np.max(
                np.abs(
                    eigs(matrix, k=1, which="LM", return_eigenvectors=False)
                )
            )
        )
        return matrix * (self.params.spectral_radius / max(radius, 1e-9))

    def update(
        self,
        reservoir: np.ndarray,
        dynamic: np.ndarray,
        static: np.ndarray,
    ) -> np.ndarray:
        preactivation = (
            self.a @ reservoir
            + self.w_in @ dynamic
            + self.w_static @ static
        )
        return (
            (1.0 - self.params.leakage_rate) * reservoir
            + self.params.leakage_rate * np.tanh(preactivation)
        )

    @staticmethod
    def augment_block(
        block_state: np.ndarray,
        dynamic: np.ndarray,
    ) -> np.ndarray:
        transformed = block_state.copy()
        transformed[1::2] = transformed[1::2] ** 2
        if transformed.ndim == 1:
            return np.r_[transformed, dynamic, 1.0]
        return np.vstack(
            [transformed, dynamic, np.ones((1, transformed.shape[1]))]
        )

    def collect_sequence_features(
        self,
        sequence: np.ndarray,
        static: np.ndarray,
        warmup: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = max(0, len(sequence) - 1 - warmup)
        states = np.empty((count, self.params.n_reservoir))
        dynamics = np.empty((count, self.dynamic_dim))
        targets = np.empty((count, self.dynamic_dim))
        reservoir = np.zeros(self.params.n_reservoir)
        output = 0
        for index in range(len(sequence) - 1):
            reservoir = self.update(reservoir, sequence[index], static)
            if index >= warmup:
                states[output] = reservoir
                dynamics[output] = sequence[index]
                targets[output] = sequence[index + 1]
                output += 1
        return states[:output], dynamics[:output], targets[:output]

    def fit(
        self,
        sequences: list[np.ndarray],
        statics: list[np.ndarray],
        warmup: int,
    ) -> None:
        all_values = np.concatenate(sequences, axis=0)
        span = np.ptp(all_values, axis=0)
        self.clip_low = np.min(all_values, axis=0) - 2.5 * span
        self.clip_high = np.max(all_values, axis=0) + 1.2 * span
        feature_dim = self.block_size + self.dynamic_dim + 1
        grams = np.zeros((self.dynamic_dim, feature_dim, feature_dim))
        target_features = np.zeros((self.dynamic_dim, feature_dim))

        for sequence, static in zip(sequences, statics):
            states, dynamics, targets = self.collect_sequence_features(
                sequence,
                static,
                warmup,
            )
            for node in range(self.dynamic_dim):
                features = self.augment_block(
                    states[:, self.block_slice(node)].T,
                    dynamics.T,
                )
                grams[node] += features @ features.T
                target_features[node] += targets[:, node] @ features.T

        identity = np.eye(feature_dim)
        for node in range(self.dynamic_dim):
            self.w_out[node] = np.linalg.solve(
                grams[node] + self.params.ridge * identity,
                target_features[node],
            )

        residual_parts = []
        for sequence, static in zip(sequences, statics):
            states, dynamics, targets = self.collect_sequence_features(
                sequence,
                static,
                warmup,
            )
            predictions = np.empty_like(targets)
            for node in range(self.dynamic_dim):
                features = self.augment_block(
                    states[:, self.block_slice(node)].T,
                    dynamics.T,
                )
                predictions[:, node] = self.w_out[node] @ features
            residual_parts.append(targets - predictions)
        residuals = np.concatenate(residual_parts, axis=0)
        lower = np.quantile(residuals, 0.0025, axis=0)
        upper = np.quantile(residuals, 0.9975, axis=0)
        self.residuals = np.clip(residuals, lower, upper)

    def rollout_switch_batch(
        self,
        warmup: np.ndarray,
        warmup_statics: np.ndarray,
        target_statics: np.ndarray,
        n_steps: int,
        seed: int,
        ramp_steps: int = 0,
    ) -> np.ndarray:
        if self.residuals is None:
            raise RuntimeError("RC must be fitted before rollout")
        if warmup.ndim != 3:
            raise ValueError("warmup must have shape (time, batch, dynamic_dim)")
        batch = warmup.shape[1]
        reservoir = np.zeros((self.params.n_reservoir, batch))
        warmup_static = warmup_statics.T
        target_static = target_statics.T
        for index in range(len(warmup)):
            reservoir = self.update(reservoir, warmup[index].T, warmup_static)
        current = warmup[-1].T
        prediction = np.empty((n_steps, batch, self.dynamic_dim), dtype=float)
        rng = np.random.default_rng(seed)
        for index in range(n_steps):
            if ramp_steps > 0 and index < ramp_steps:
                fraction = (index + 1) / ramp_steps
                static = (1.0 - fraction) * warmup_static + fraction * target_static
            else:
                static = target_static
            reservoir = self.update(reservoir, current, static)
            next_dynamic = np.empty_like(current)
            for node in range(self.dynamic_dim):
                features = self.augment_block(
                    reservoir[self.block_slice(node)],
                    current,
                )
                next_dynamic[node] = self.w_out[node] @ features
            residual_index = rng.integers(0, len(self.residuals), size=batch)
            innovations = self.residuals[residual_index].T
            next_dynamic += self.params.innovation_scaling * innovations
            current = np.clip(
                next_dynamic,
                self.clip_low[:, None],
                self.clip_high[:, None],
            )
            prediction[index] = current.T
        return prediction


class GraphLocalStaticRC:
    """Topology-aware local reservoir blocks with no global readout bypass.

    Block ``i`` receives the observable of node ``i`` and its direct
    structural neighbours. Recurrent communication between blocks follows the
    same structural graph, while output ``i`` is decoded only from block ``i``.
    The default readout omits a direct input-to-output skip. The opt-in local
    skip supports paired tests against the single-node readout mechanism.
    """

    def __init__(
        self,
        params: CoupledRCParams,
        dynamic_dim: int,
        static_dim: int,
        readout_stride: int = 4,
        *,
        direct_input: bool = False,
    ):
        self.params = params
        self.dynamic_dim = int(dynamic_dim)
        self.static_dim = int(static_dim)
        self.block_size = int(params.block_size)
        self.readout_stride = max(1, int(readout_stride))
        self.direct_input = bool(direct_input)
        if self.dynamic_dim != 8:
            raise ValueError("The graph-local reservoir expects 8 nodes")
        expected = self.block_size * self.dynamic_dim
        if params.n_reservoir != expected:
            raise ValueError(
                f"n_reservoir must equal block_size * dynamic_dim ({expected})"
            )

        rng = np.random.default_rng(params.seed)
        connectivity = build_synthetic_structural_proxy(self.dynamic_dim)
        connectivity = connectivity / max(float(np.max(connectivity)), 1e-12)
        row_sum = np.sum(connectivity, axis=1, keepdims=True)
        self.graph_mixing = np.divide(
            connectivity,
            row_sum,
            out=np.zeros_like(connectivity),
            where=row_sum > 0.0,
        )
        self.local_channels: list[np.ndarray] = []
        self.w_in = np.zeros((params.n_reservoir, self.dynamic_dim))
        for node in range(self.dynamic_dim):
            channels = np.flatnonzero(
                (connectivity[node] > 0.0)
                | (connectivity[:, node] > 0.0)
                | (np.arange(self.dynamic_dim) == node)
            )
            self.local_channels.append(channels)
            weights = connectivity[node, channels] + connectivity[channels, node]
            weights[channels == node] = 1.0
            weights /= max(float(np.linalg.norm(weights)), 1e-12)
            block = self.block_slice(node)
            self.w_in[block, channels] = (
                rng.uniform(-1.0, 1.0, (self.block_size, len(channels)))
                * params.input_scaling
                * weights[None, :]
            )

        # The selected SCDT statistic is a network-level risk coordinate and
        # is therefore broadcast to all local blocks.
        self.w_static = (
            rng.uniform(-1.0, 1.0, (params.n_reservoir, self.static_dim))
            * params.static_scaling
        )
        self.a_local = self.build_local_recurrent_matrix(rng)
        self.w_out: list[np.ndarray] = []
        self.residuals: np.ndarray | None = None
        self.clip_low = np.full(self.dynamic_dim, -5.0)
        self.clip_high = np.full(self.dynamic_dim, 5.0)

    def block_slice(self, node: int) -> slice:
        start = node * self.block_size
        return slice(start, start + self.block_size)

    def build_local_recurrent_matrix(
        self,
        rng: np.random.Generator,
    ) -> sparse.csr_matrix:
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        local_nnz = max(1, int(self.params.density * self.block_size**2))
        for node in range(self.dynamic_dim):
            offset = node * self.block_size
            rows.append(offset + rng.integers(0, self.block_size, local_nnz))
            columns.append(offset + rng.integers(0, self.block_size, local_nnz))
            values.append(rng.normal(0.0, 1.0, local_nnz))
        matrix = sparse.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(self.params.n_reservoir, self.params.n_reservoir),
        ).tocsr()
        radius = float(
            np.max(
                np.abs(
                    eigs(
                        matrix,
                        k=1,
                        which="LM",
                        return_eigenvectors=False,
                        v0=np.ones(self.params.n_reservoir),
                    )
                )
            )
        )
        return matrix * (self.params.spectral_radius / max(radius, 1e-9))

    def graph_message(self, reservoir: np.ndarray) -> np.ndarray:
        if reservoir.ndim == 1:
            blocks = reservoir.reshape(self.dynamic_dim, self.block_size)
            mixed = np.einsum("ij,jk->ik", self.graph_mixing, blocks)
        else:
            blocks = reservoir.reshape(
                self.dynamic_dim,
                self.block_size,
                reservoir.shape[1],
            )
            mixed = np.einsum("ij,jkb->ikb", self.graph_mixing, blocks)
        return mixed.reshape(reservoir.shape)

    def update(
        self,
        reservoir: np.ndarray,
        dynamic: np.ndarray,
        static: np.ndarray,
    ) -> np.ndarray:
        preactivation = (
            self.a_local @ reservoir
            + self.params.block_coupling * self.graph_message(reservoir)
            + self.w_in @ dynamic
            + self.w_static @ static
        )
        return (
            (1.0 - self.params.leakage_rate) * reservoir
            + self.params.leakage_rate * np.tanh(preactivation)
        )

    @staticmethod
    def augment_local(
        block_state: np.ndarray,
    ) -> np.ndarray:
        transformed = block_state.copy()
        transformed[1::2] = transformed[1::2] ** 2
        if transformed.ndim == 1:
            return np.r_[transformed, 1.0]
        return np.vstack(
            [transformed, np.ones((1, transformed.shape[1]))]
        )

    def readout_features(
        self,
        block_state: np.ndarray,
        own_dynamic: np.ndarray,
    ) -> np.ndarray:
        features = self.augment_local(block_state)
        if not self.direct_input:
            return features
        # Match scalar SCDT's [transformed state, current input, bias] order.
        if features.ndim == 1:
            return np.r_[features[:-1], own_dynamic, features[-1]]
        return np.vstack([features[:-1], own_dynamic, features[-1:]])

    def fit(
        self,
        sequences: list[np.ndarray],
        statics: list[np.ndarray],
        warmup: int,
    ) -> None:
        all_values = np.concatenate(sequences, axis=0)
        span = np.ptp(all_values, axis=0)
        self.clip_low = np.min(all_values, axis=0) - 2.5 * span
        self.clip_high = np.max(all_values, axis=0) + 1.2 * span
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) != 1:
            raise ValueError("graph-local batching requires equal sequence lengths")

        sequence_batch = np.stack(sequences, axis=1)
        static_batch = np.stack(statics, axis=0).T
        batch = len(sequences)
        sampled_steps = len(
            range(warmup, len(sequence_batch) - 1, self.readout_stride)
        )
        columns = sampled_steps * batch
        sampled_states = np.empty(
            (columns, self.params.n_reservoir),
            dtype=np.float32,
        )
        targets = np.empty((columns, self.dynamic_dim), dtype=np.float32)
        sampled_inputs = np.empty_like(targets)
        reservoir = np.zeros((self.params.n_reservoir, batch))
        column = 0
        for index in range(len(sequence_batch) - 1):
            reservoir = self.update(
                reservoir,
                sequence_batch[index].T,
                static_batch,
            )
            if index >= warmup and (index - warmup) % self.readout_stride == 0:
                next_column = column + batch
                sampled_states[column:next_column] = reservoir.T
                targets[column:next_column] = sequence_batch[index + 1]
                sampled_inputs[column:next_column] = sequence_batch[index]
                column = next_column

        residuals = np.empty_like(targets)
        self.w_out = []
        for node in range(self.dynamic_dim):
            features = self.readout_features(
                sampled_states[:, self.block_slice(node)].T,
                sampled_inputs[:, node][None, :],
            ).T
            model = Ridge(
                alpha=self.params.ridge,
                fit_intercept=False,
                solver="cholesky",
            )
            model.fit(features, targets[:, node])
            self.w_out.append(np.asarray(model.coef_, dtype=float))
            residuals[:, node] = targets[:, node] - model.predict(features)
        lower = np.quantile(residuals, 0.0025, axis=0)
        upper = np.quantile(residuals, 0.9975, axis=0)
        self.residuals = np.clip(residuals, lower, upper)

    def rollout_switch_batch(
        self,
        warmup: np.ndarray,
        warmup_statics: np.ndarray,
        target_statics: np.ndarray,
        n_steps: int,
        seed: int,
        ramp_steps: int = 0,
    ) -> np.ndarray:
        if not self.w_out or self.residuals is None:
            raise RuntimeError("RC must be fitted before rollout")
        if warmup.ndim != 3:
            raise ValueError("warmup must have shape (time, batch, dynamic_dim)")
        if len(warmup) == 0:
            raise ValueError("warmup must contain at least one observation")
        batch = warmup.shape[1]
        reservoir = np.zeros((self.params.n_reservoir, batch))
        warmup_static = warmup_statics.T
        target_static = target_statics.T
        # The rollout loop consumes the final observation before decoding its
        # successor, exactly as fit() does. Do not consume that sample twice.
        for index in range(len(warmup) - 1):
            reservoir = self.update(reservoir, warmup[index].T, warmup_static)
        current = warmup[-1].T
        prediction = np.empty((n_steps, batch, self.dynamic_dim), dtype=float)
        rng = np.random.default_rng(seed)
        for index in range(n_steps):
            if ramp_steps > 0 and index < ramp_steps:
                fraction = (index + 1) / ramp_steps
                static = (1.0 - fraction) * warmup_static + fraction * target_static
            else:
                static = target_static
            reservoir = self.update(reservoir, current, static)
            next_dynamic = np.empty_like(current)
            for node, channels in enumerate(self.local_channels):
                features = self.readout_features(
                    reservoir[self.block_slice(node)],
                    current[node:node + 1],
                )
                next_dynamic[node] = self.w_out[node] @ features
            if self.params.innovation_scaling > 0.0:
                residual_index = rng.integers(0, len(self.residuals), size=batch)
                next_dynamic += (
                    self.params.innovation_scaling
                    * self.residuals[residual_index].T
                )
            current = np.clip(
                next_dynamic,
                self.clip_low[:, None],
                self.clip_high[:, None],
            )
            prediction[index] = current.T
        return prediction


def grouped_recruitment(
    coordinates: np.ndarray,
    events: np.ndarray,
    network_event_nodes: int,
) -> dict[str, np.ndarray]:
    grid = np.asarray(sorted(np.unique(coordinates)))
    node_probability = []
    network_survival = []
    median_onset = []
    for value in grid:
        selected = events[np.isclose(coordinates, value)]
        finite = np.isfinite(selected)
        node_probability.append(np.mean(finite, axis=0))
        # A network recruitment event must occur inside the pre-registered
        # EZ/PZ set. Healthy-zone threshold crossings cannot substitute for
        # failed propagation into nodes 1-3.
        network_survival.append(
            np.mean(np.sum(finite[:, :4], axis=1) < network_event_nodes)
        )
        medians = []
        for node in range(events.shape[1]):
            values = selected[:, node]
            medians.append(
                np.nan if not np.any(np.isfinite(values)) else np.nanmedian(values)
            )
        median_onset.append(medians)
    node_probability_array = np.asarray(node_probability)
    return {
        "x0": grid,
        "node_probability": node_probability_array,
        "node_survival": 1.0 - node_probability_array,
        "network_survival": np.asarray(network_survival),
        "median_onset": np.asarray(median_onset),
    }


def prominent_trough_rate(signal: np.ndarray, sample_hz: float) -> float:
    """Return sustained-event-compatible troughs per second."""
    values = np.asarray(signal, dtype=float)
    if len(values) < 3:
        return 0.0
    troughs, _ = find_peaks(-values, height=2.0, prominence=0.35)
    return float(len(troughs) / (len(values) / sample_hz))


def ictal_regime_metrics(
    cohort: NetworkSwitchCohort,
    predictions: np.ndarray,
    rc_events: np.ndarray,
    gt: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
    sample_hz: float,
) -> dict[str, float]:
    """Measure whether risky static context produces an ictal attractor.

    Survival curves only test whether an event detector fired. These metrics
    additionally reject non-oscillatory fixed-point saturation and compare the
    amount, timing, and morphology of EZ/PZ recruitment at each x0 condition.
    """
    horizon_s = cohort.futures_raw.shape[0] / sample_hz
    recruitment_count_errors: list[float] = []
    onset_errors: list[float] = []
    variability_deficits: list[float] = []
    derivative_deficits: list[float] = []
    trough_rate_deficits: list[float] = []
    variability_errors: list[float] = []
    derivative_errors: list[float] = []
    trough_rate_errors: list[float] = []
    spatial_leakage: list[float] = []
    saturation_fractions: list[float] = []

    for condition, value in enumerate(gt["x0"]):
        selection = np.flatnonzero(np.isclose(cohort.ez_x0, value))
        gt_finite = np.isfinite(cohort.gt_events[selection, :4])
        rc_finite = np.isfinite(rc_events[selection, :4])
        gt_count = float(np.mean(np.sum(gt_finite, axis=1)))
        rc_count = float(np.mean(np.sum(rc_finite, axis=1)))
        recruitment_count_errors.append(abs(gt_count - rc_count) / 4.0)

        for node in range(4):
            # Compare onset only where GT establishes a non-negligible event
            # probability; a missing RC event receives the maximum penalty.
            if gt["node_probability"][condition, node] >= 0.25:
                gt_onset = gt["median_onset"][condition, node]
                rc_onset = predicted["median_onset"][condition, node]
                if np.isfinite(rc_onset):
                    onset_errors.append(
                        min(abs(float(rc_onset - gt_onset)) / horizon_s, 1.0)
                    )
                else:
                    onset_errors.append(1.0)

        # Morphology is assessed only in conditions where GT commonly enters
        # the network-ictal regime. Skip the first 8 s of switching transient.
        if gt["network_survival"][condition] > 0.50:
            continue
        start = min(
            cohort.futures_raw.shape[0] - 3,
            int(round(8.0 * sample_hz)),
        )
        gt_all = cohort.futures_raw[start:, selection]
        rc_all = predictions[start:, selection]
        gt_chunk = gt_all[:, :, :4]
        rc_chunk = rc_all[:, :, :4]
        gt_std = np.std(gt_chunk, axis=0)
        rc_std = np.std(rc_chunk, axis=0)
        gt_diff = np.sqrt(np.mean(np.diff(gt_chunk, axis=0) ** 2, axis=0))
        rc_diff = np.sqrt(np.mean(np.diff(rc_chunk, axis=0) ** 2, axis=0))

        gt_std_level = float(np.median(gt_std)) + 1e-8
        rc_std_level = float(np.median(rc_std))
        gt_diff_level = float(np.median(gt_diff)) + 1e-8
        rc_diff_level = float(np.median(rc_diff))
        variability_deficits.append(
            float(np.clip(1.0 - rc_std_level / gt_std_level, 0.0, 1.0))
        )
        derivative_deficits.append(
            float(np.clip(1.0 - rc_diff_level / gt_diff_level, 0.0, 1.0))
        )
        variability_errors.append(
            float(
                np.clip(
                    abs(np.log((rc_std_level + 1e-8) / gt_std_level)),
                    0.0,
                    3.0,
                )
            )
        )
        derivative_errors.append(
            float(
                np.clip(
                    abs(np.log((rc_diff_level + 1e-8) / gt_diff_level)),
                    0.0,
                    3.0,
                )
            )
        )

        gt_rates = []
        rc_rates = []
        for trial in range(gt_chunk.shape[1]):
            for node in range(gt_chunk.shape[2]):
                gt_rates.append(
                    prominent_trough_rate(gt_chunk[:, trial, node], sample_hz)
                )
                rc_rates.append(
                    prominent_trough_rate(rc_chunk[:, trial, node], sample_hz)
                )
        gt_rate = float(np.mean(gt_rates)) + 1e-8
        rc_rate = float(np.mean(rc_rates))
        trough_rate_deficits.append(
            float(np.clip(1.0 - rc_rate / gt_rate, 0.0, 1.0))
        )
        trough_rate_errors.append(
            float(
                np.clip(
                    abs(np.log((rc_rate + 1e-8) / gt_rate)),
                    0.0,
                    3.0,
                )
            )
        )

        gt_hz_rates = []
        rc_hz_rates = []
        for trial in range(gt_all.shape[1]):
            for node in range(4, gt_all.shape[2]):
                gt_hz_rates.append(
                    prominent_trough_rate(gt_all[:, trial, node], sample_hz)
                )
                rc_hz_rates.append(
                    prominent_trough_rate(rc_all[:, trial, node], sample_hz)
                )
        gt_hz_rate = float(np.mean(gt_hz_rates))
        rc_hz_rate = float(np.mean(rc_hz_rates))
        spatial_leakage.append(
            float(
                np.clip(
                    max(rc_hz_rate - gt_hz_rate, 0.0) / gt_rate,
                    0.0,
                    2.0,
                )
            )
        )

        # A high-risk rollout is considered saturated when its temporal spread
        # is less than 20% of the corresponding GT node-level spread.
        reference = np.median(gt_std, axis=0, keepdims=True)
        saturation_fractions.append(
            float(np.mean(rc_std < 0.20 * np.maximum(reference, 1e-8)))
        )

    def average(values: list[float], fallback: float = 0.0) -> float:
        return float(np.mean(values)) if values else fallback

    return {
        "recruitment_count_mae": average(recruitment_count_errors),
        "onset_timing_mae": average(onset_errors),
        "ictal_variability_deficit": average(variability_deficits),
        "ictal_derivative_deficit": average(derivative_deficits),
        "ictal_trough_rate_deficit": average(trough_rate_deficits),
        "ictal_variability_error": average(variability_errors),
        "ictal_derivative_error": average(derivative_errors),
        "ictal_trough_rate_error": average(trough_rate_errors),
        "ictal_spatial_leakage": average(spatial_leakage),
        "saturation_fraction": average(saturation_fractions),
    }


def parameters_from_unit(
    unit: np.ndarray,
    seed: int,
    n_reservoir: int,
    ridge_log10_min: float,
    ridge_log10_max: float,
) -> RCParams:
    bounds = np.asarray(
        [
            [0.30, 2.00],
            [0.05, 0.80],
            [0.15, 0.75],
            [0.02, 0.40],
            [ridge_log10_min, ridge_log10_max],
            [0.001, 0.012],
            [0.00, 0.50],
        ]
    )
    values = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    return RCParams(
        n_reservoir=n_reservoir,
        spectral_radius=float(values[0]),
        leakage_rate=float(values[1]),
        input_scaling=float(values[2]),
        static_scaling=float(values[3]),
        ridge=float(10.0 ** values[4]),
        density=float(values[5]),
        innovation_scaling=float(values[6]),
        seed=seed,
    )


def coupled_parameters_from_unit(
    unit: np.ndarray,
    seed: int,
    block_size: int,
) -> CoupledRCParams:
    bounds = np.asarray(
        [
            [0.35, 1.05],
            [0.05, 0.32],
            [0.15, 0.75],
            [0.02, 0.40],
            [-7.0, -2.5],
            [0.02, 0.08],
            [0.00, 0.50],
            [0.02, 0.30],
        ]
    )
    values = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    return CoupledRCParams(
        n_reservoir=8 * block_size,
        spectral_radius=float(values[0]),
        leakage_rate=float(values[1]),
        input_scaling=float(values[2]),
        static_scaling=float(values[3]),
        ridge=float(10.0 ** values[4]),
        density=float(values[5]),
        innovation_scaling=float(values[6]),
        seed=seed,
        block_size=block_size,
        block_coupling=float(values[7]),
        cross_density_ratio=0.25,
    )


def graph_local_parameters_from_unit(
    unit: np.ndarray,
    seed: int,
    block_size: int,
    ridge_log10_min: float,
    ridge_log10_max: float,
) -> CoupledRCParams:
    """Map BO coordinates to the strict graph-local search space."""
    bounds = np.asarray(
        [
            [0.30, 2.00],
            [0.05, 0.80],
            [0.10, 0.80],
            [0.01, 0.40],
            [ridge_log10_min, ridge_log10_max],
            [0.003, 0.025],
            [0.00, 0.35],
        ]
    )
    values = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    return CoupledRCParams(
        n_reservoir=8 * block_size,
        spectral_radius=float(values[0]),
        leakage_rate=float(values[1]),
        input_scaling=float(values[2]),
        static_scaling=float(values[3]),
        ridge=float(10.0 ** values[4]),
        density=float(values[5]),
        # Innovation noise can imitate variance without learning an attractor,
        # so phase-preserving BO keeps autonomous dynamics deterministic.
        innovation_scaling=0.0,
        seed=seed,
        block_size=block_size,
        block_coupling=float(values[6]),
        cross_density_ratio=0.0,
    )


def build_rc(
    args: argparse.Namespace,
    params: RCParams | CoupledRCParams,
) -> MultiOutputStaticRC | GraphCoupledBlockRC | GraphLocalStaticRC:
    if args.architecture == "global":
        return MultiOutputStaticRC(
            params,
            dynamic_dim=8,
            static_dim=args.static_dim,
            readout_stride=args.readout_stride,
            readout_size=args.readout_size,
        )
    if not isinstance(params, CoupledRCParams):
        raise TypeError("block architecture requires CoupledRCParams")
    if args.architecture == "graph-local":
        return GraphLocalStaticRC(
            params,
            dynamic_dim=8,
            static_dim=args.static_dim,
            readout_stride=args.readout_stride,
        )
    return GraphCoupledBlockRC(
        params,
        dynamic_dim=8,
        static_dim=args.static_dim,
    )


def lag1_batch(values: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation over time for every trial and channel."""
    centered = values - np.mean(values, axis=0, keepdims=True)
    denominator = np.sum(centered**2, axis=0)
    return np.divide(
        np.sum(centered[:-1] * centered[1:], axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )


def normal_rollout_segments(
    cohort: NetworkSwitchCohort,
    sample_hz: float,
    warmup_s: float,
    rollout_s: float,
    segments_per_trajectory: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build evenly spaced normal warmup/target pairs from each trajectory."""
    warmup_steps = int(round(warmup_s * sample_hz))
    rollout_steps = int(round(rollout_s * sample_hz))
    available = len(cohort.futures_raw) - warmup_steps - rollout_steps
    if warmup_steps < 1 or rollout_steps < 1:
        raise ValueError("normal segment warmup and rollout must be positive")
    if available < 0:
        raise ValueError("normal validation trajectory is too short for segments")
    starts = np.unique(
        np.rint(
            np.linspace(0, available, segments_per_trajectory)
        ).astype(int)
    )
    if len(starts) != segments_per_trajectory:
        raise ValueError("normal validation trajectory cannot provide enough segments")

    warmups = np.concatenate(
        [
            cohort.futures_raw[start : start + warmup_steps]
            for start in starts
        ],
        axis=1,
    )
    targets = np.concatenate(
        [
            cohort.futures_raw[
                start + warmup_steps : start + warmup_steps + rollout_steps
            ]
            for start in starts
        ],
        axis=1,
    )
    coordinates = np.tile(cohort.ez_x0, len(starts))
    return warmups, targets, coordinates


def phase_aligned_nrmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    max_lag_steps: int,
) -> np.ndarray:
    """Return the best lag-tolerant NRMSE for every trial and channel."""
    scale = np.std(targets, axis=0) + 1e-8
    best = np.full(scale.shape, np.inf)
    maximum = min(max(0, int(max_lag_steps)), max(0, len(targets) // 3))
    for lag in range(-maximum, maximum + 1):
        if lag < 0:
            predicted = predictions[:lag]
            target = targets[-lag:]
        elif lag > 0:
            predicted = predictions[lag:]
            target = targets[:-lag]
        else:
            predicted = predictions
            target = targets
        error = np.sqrt(np.mean((predicted - target) ** 2, axis=0)) / scale
        best = np.minimum(best, error)
    return best


def spectral_centroid_batch(values: np.ndarray, sample_hz: float) -> np.ndarray:
    """Power-weighted temporal frequency for every trial and channel."""
    centered = values - np.mean(values, axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_hz)
    if len(frequencies) > 1:
        power[0] = 0.0
    denominator = np.sum(power, axis=0)
    return np.divide(
        np.sum(power * frequencies[:, None, None], axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )


def normal_peak_rate_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    sample_hz: float,
) -> float:
    """Compare broad normal-spike counts without requiring phase agreement."""
    duration_s = len(targets) / sample_hz
    minimum_rate = 1.0 / max(duration_s, 1e-8)
    errors: list[float] = []
    for trial in range(targets.shape[1]):
        for node in range(targets.shape[2]):
            reference = targets[:, trial, node]
            candidate = predictions[:, trial, node]
            reference_range = float(
                np.quantile(reference, 0.98) - np.quantile(reference, 0.10)
            )
            prominence = max(0.30 * reference_range, 0.08)
            reference_peaks, _ = find_peaks(reference, prominence=prominence)
            candidate_peaks, _ = find_peaks(candidate, prominence=prominence)
            reference_rate = len(reference_peaks) / duration_s
            candidate_rate = len(candidate_peaks) / duration_s
            errors.append(
                min(
                    abs(candidate_rate - reference_rate)
                    / max(reference_rate, minimum_rate),
                    5.0,
                )
            )
    return float(np.mean(errors))


def evaluate_normal_rc(
    rc: MultiOutputStaticRC | GraphCoupledBlockRC | GraphLocalStaticRC,
    cohort: NetworkSwitchCohort,
    coefficients: np.ndarray,
    dynamic_center: np.ndarray,
    dynamic_scale: np.ndarray,
    static_center: np.ndarray,
    static_scale: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    include_full_rollout: bool = False,
) -> dict:
    """Score repeated short normal rollouts with the selected BO objective."""
    sample_hz = 256.0 / args.rc_stride
    warmups_raw, gt, segment_x0 = normal_rollout_segments(
        cohort,
        sample_hz,
        args.normal_segment_warmup_s,
        args.normal_short_s,
        args.normal_segments_per_trajectory,
    )
    warmups = normalize(warmups_raw, dynamic_center, dynamic_scale)
    segment_static_raw = static_from_x0(segment_x0, coefficients)
    segment_static = normalize(
        segment_static_raw,
        static_center,
        static_scale,
    )
    prediction_norm = rc.rollout_switch_batch(
        warmups,
        segment_static,
        segment_static,
        len(gt),
        seed,
        0,
    )
    predictions = prediction_norm * dynamic_scale + dynamic_center
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("non-finite normal rollout")

    # Normalize each start independently. With a full-trajectory scale, quiet
    # short windows make a constant global-mean prediction score artificially
    # below one even though it carries no local trajectory information.
    segment_scale = np.std(gt, axis=0) + 1e-8
    segment_channel_nrmse = (
        np.sqrt(np.mean((predictions - gt) ** 2, axis=0)) / segment_scale
    )
    segment_scores = np.mean(segment_channel_nrmse, axis=1)
    mean_nrmse = float(np.mean(segment_scores))

    gt_std = np.std(gt, axis=0) + 1e-8
    rc_std = np.std(predictions, axis=0) + 1e-8
    mean_error = float(
        np.median(np.abs(np.mean(predictions, axis=0) - np.mean(gt, axis=0)) / gt_std)
    )
    std_log_error = float(
        np.median(np.clip(np.abs(np.log(rc_std / gt_std)), 0.0, 8.0))
    )
    gt_derivative = np.sqrt(np.mean(np.diff(gt, axis=0) ** 2, axis=0)) + 1e-8
    rc_derivative = (
        np.sqrt(np.mean(np.diff(predictions, axis=0) ** 2, axis=0)) + 1e-8
    )
    derivative_log_error = float(
        np.median(
            np.clip(np.abs(np.log(rc_derivative / gt_derivative)), 0.0, 8.0)
        )
    )
    ac1_error = float(np.median(np.abs(lag1_batch(predictions) - lag1_batch(gt))))
    lag_scores = phase_aligned_nrmse(
        predictions,
        gt,
        int(round(args.phase_max_lag_s * sample_hz)),
    )
    phase_nrmse = float(np.mean(lag_scores))
    phase_p90_nrmse = float(np.quantile(np.mean(lag_scores, axis=1), 0.90))
    distribution_channel_nrmse = (
        np.sqrt(
            np.mean(
                (
                    np.sort(predictions, axis=0)
                    - np.sort(gt, axis=0)
                )
                ** 2,
                axis=0,
            )
        )
        / gt_std
    )
    distribution_nrmse = float(np.mean(distribution_channel_nrmse))
    gt_centroid = spectral_centroid_batch(gt, sample_hz)
    rc_centroid = spectral_centroid_batch(predictions, sample_hz)
    spectral_centroid_log_error = float(
        np.median(
            np.clip(
                np.abs(np.log((rc_centroid + 0.05) / (gt_centroid + 0.05))),
                0.0,
                4.0,
            )
        )
    )
    peak_rate_error = normal_peak_rate_error(predictions, gt, sample_hz)
    collapse_fraction = float(
        np.mean(
            (rc_std < args.collapse_std_ratio * gt_std)
            | (
                rc_derivative
                < args.collapse_derivative_ratio * gt_derivative
            )
        )
    )
    mean_shift = np.abs(
        np.mean(predictions, axis=0) - np.mean(gt, axis=0)
    ) / gt_std
    runaway_fraction = float(
        np.mean(
            (rc_std > args.runaway_std_ratio * gt_std)
            | (mean_shift > args.runaway_mean_z)
        )
    )
    if args.objective == "phase":
        collapse_penalty = (
            40.0 * collapse_fraction
            + 80.0 * max(collapse_fraction - 0.10, 0.0)
        )
        runaway_penalty = (
            40.0 * runaway_fraction
            + 80.0 * max(runaway_fraction - 0.10, 0.0)
        )
        score = (
            0.25 * min(phase_nrmse, 10.0)
            + min(distribution_nrmse, 10.0)
            + 0.05 * min(mean_nrmse, 10.0)
            + 0.60 * std_log_error
            + 0.40 * derivative_log_error
            + 0.25 * ac1_error
            + 1.00 * spectral_centroid_log_error
            + 0.75 * peak_rate_error
            + collapse_penalty
            + runaway_penalty
        )
    else:
        collapse_penalty = 0.0
        runaway_penalty = 0.0
        score = mean_nrmse
    result = {
        "score": float(score),
        "short_nrmse": mean_nrmse,
        "segment_mean_nrmse": mean_nrmse,
        "segment_median_nrmse": float(np.median(segment_scores)),
        "segment_p90_nrmse": float(np.quantile(segment_scores, 0.90)),
        "phase_nrmse": phase_nrmse,
        "phase_p90_nrmse": phase_p90_nrmse,
        "distribution_nrmse": distribution_nrmse,
        "spectral_centroid_log_error": spectral_centroid_log_error,
        "normal_peak_rate_error": peak_rate_error,
        "collapse_fraction": collapse_fraction,
        "collapse_penalty": collapse_penalty,
        "runaway_fraction": runaway_fraction,
        "runaway_penalty": runaway_penalty,
        "normal_mean_error": mean_error,
        "normal_std_log_error": std_log_error,
        "normal_derivative_log_error": derivative_log_error,
        "normal_ac1_error": ac1_error,
    }
    if include_full_rollout:
        result["segment_predictions"] = predictions
        result["segment_targets"] = gt
        result["segment_x0"] = segment_x0
        full_warmups = normalize(
            cohort.warmups_raw,
            dynamic_center,
            dynamic_scale,
        )
        target_raw = static_from_x0(cohort.ez_x0, coefficients)
        target_static = normalize(target_raw, static_center, static_scale)
        base_raw = static_from_x0([args.base_ez_x0], coefficients)[0]
        base_static = normalize(base_raw, static_center, static_scale)
        warmup_static = np.repeat(
            base_static[None, :],
            len(cohort.ez_x0),
            axis=0,
        )
        full_prediction_norm = rc.rollout_switch_batch(
            full_warmups,
            warmup_static,
            target_static,
            cohort.futures_raw.shape[0],
            seed + 1,
            args.ramp_steps,
        )
        result["predictions"] = (
            full_prediction_norm * dynamic_scale + dynamic_center
        )
    return result


def evaluate_rc(
    rc: MultiOutputStaticRC | GraphCoupledBlockRC | GraphLocalStaticRC,
    cohort: NetworkSwitchCohort,
    coefficients: np.ndarray,
    dynamic_center: np.ndarray,
    dynamic_scale: np.ndarray,
    static_center: np.ndarray,
    static_scale: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    warmups = normalize(cohort.warmups_raw, dynamic_center, dynamic_scale)
    target_raw = static_from_x0(cohort.ez_x0, coefficients)
    target_static = normalize(target_raw, static_center, static_scale)
    base_raw = static_from_x0([args.base_ez_x0], coefficients)[0]
    base_static = normalize(base_raw, static_center, static_scale)
    warmup_static = np.repeat(base_static[None, :], len(cohort.ez_x0), axis=0)
    prediction_norm = rc.rollout_switch_batch(
        warmups,
        warmup_static,
        target_static,
        cohort.futures_raw.shape[0],
        seed,
        args.ramp_steps,
    )
    predictions = prediction_norm * dynamic_scale + dynamic_center
    rc_events = event_times(predictions, 256.0 / args.rc_stride)
    gt = grouped_recruitment(
        cohort.ez_x0,
        cohort.gt_events,
        args.network_event_nodes,
    )
    predicted = grouped_recruitment(
        cohort.ez_x0,
        rc_events,
        args.network_event_nodes,
    )

    gt_local = gt["node_survival"][:, 0]
    rc_local = predicted["node_survival"][:, 0]
    gt_network = gt["network_survival"]
    rc_network = predicted["network_survival"]
    local_mae = float(np.mean(np.abs(gt_local - rc_local)))
    network_mae = float(np.mean(np.abs(gt_network - rc_network)))
    pz_mae = float(
        np.mean(
            np.abs(
                gt["node_probability"][:, 1:4]
                - predicted["node_probability"][:, 1:4]
            )
        )
    )
    hz_false_recruitment = float(
        np.mean(predicted["node_probability"][:, 4:])
    )
    false_local = float(np.mean(np.maximum(gt_local - rc_local, 0.0)))
    missed_local = float(np.mean(np.maximum(rc_local - gt_local, 0.0)))
    false_network = float(np.mean(np.maximum(gt_network - rc_network, 0.0)))
    missed_network = float(np.mean(np.maximum(rc_network - gt_network, 0.0)))
    monotonic_penalty = float(
        np.mean(np.maximum(np.diff(rc_local), 0.0))
        + np.mean(np.maximum(np.diff(rc_network), 0.0))
    )
    ictal_metrics = ictal_regime_metrics(
        cohort,
        predictions,
        rc_events,
        gt,
        predicted,
        256.0 / args.rc_stride,
    )

    stable = np.flatnonzero(~np.isfinite(cohort.gt_events[:, 0]))
    short = min(predictions.shape[0], int(round(8.0 * 256.0 / args.rc_stride)))
    morphology = []
    distribution = []
    for index in stable:
        gt_signal = cohort.futures_raw[:short, index]
        rc_signal = predictions[:short, index]
        denominator = np.std(gt_signal, axis=0) + 1e-8
        morphology.extend(
            (np.sqrt(np.mean((rc_signal - gt_signal) ** 2, axis=0)) / denominator).tolist()
        )
        distribution.extend(
            np.abs(
                np.log(
                    (np.std(rc_signal, axis=0) + 1e-8)
                    / (np.std(gt_signal, axis=0) + 1e-8)
                )
            ).tolist()
        )
    short_nrmse = float(np.median(morphology)) if morphology else 8.0
    distribution_error = float(np.median(distribution)) if distribution else 8.0
    constraint_penalty = (
        80.0 * max(false_local - 0.05, 0.0)
        + 100.0 * max(false_network - 0.05, 0.0)
        + 80.0 * max(missed_local - 0.05, 0.0)
        + 150.0 * max(missed_network - 0.05, 0.0)
        + 250.0 * max(hz_false_recruitment - 0.02, 0.0)
        + 60.0 * max(ictal_metrics["saturation_fraction"] - 0.10, 0.0)
        + 40.0 * max(ictal_metrics["ictal_trough_rate_error"] - 0.35, 0.0)
        + 100.0 * max(ictal_metrics["ictal_spatial_leakage"] - 0.05, 0.0)
        + 25.0 * max(short_nrmse - 3.0, 0.0)
        + 40.0 * max(distribution_error - 0.75, 0.0)
    )
    score = (
        12.0 * false_local
        + 8.0 * missed_local
        + 14.0 * false_network
        + 10.0 * missed_network
        + 5.0 * pz_mae
        + 10.0 * hz_false_recruitment
        + 12.0 * ictal_metrics["recruitment_count_mae"]
        + 8.0 * ictal_metrics["onset_timing_mae"]
        + 10.0 * ictal_metrics["ictal_variability_deficit"]
        + 8.0 * ictal_metrics["ictal_derivative_deficit"]
        + 8.0 * ictal_metrics["ictal_variability_error"]
        + 8.0 * ictal_metrics["ictal_derivative_error"]
        + 12.0 * ictal_metrics["ictal_trough_rate_error"]
        + 25.0 * ictal_metrics["ictal_spatial_leakage"]
        + 20.0 * ictal_metrics["saturation_fraction"]
        + 1.5 * monotonic_penalty
        + 2.0 * min(short_nrmse, 15.0)
        + 2.0 * min(distribution_error, 8.0)
        + constraint_penalty
    )
    return {
        "score": score,
        "local_mae": local_mae,
        "network_mae": network_mae,
        "pz_mae": pz_mae,
        "hz_false_recruitment": hz_false_recruitment,
        "false_local": false_local,
        "missed_local": missed_local,
        "false_network": false_network,
        "missed_network": missed_network,
        "monotonic_penalty": monotonic_penalty,
        "short_nrmse": short_nrmse,
        "distribution_error": distribution_error,
        "constraint_penalty": constraint_penalty,
        **ictal_metrics,
        "predictions": predictions,
        "rc_events": rc_events,
        "gt": gt,
        "rc": predicted,
    }


def optimize_rc(
    args: argparse.Namespace,
    train_sequences: list[np.ndarray],
    train_statics: list[np.ndarray],
    validation: NetworkSwitchCohort,
    coefficients: np.ndarray,
    dynamic_center: np.ndarray,
    dynamic_scale: np.ndarray,
    static_center: np.ndarray,
    static_scale: np.ndarray,
) -> tuple[RCParams | CoupledRCParams, list[dict]]:
    rng = np.random.default_rng(args.seed + 31000)
    unit_points: list[np.ndarray] = []
    scores: list[float] = []
    rows: list[dict] = []
    initial = min(7, max(4, args.bo_calls // 3))
    if args.architecture == "global":
        dimensions = 7
        previous_best = np.asarray(
            [0.927989, 0.716955, 0.634341, 0.266047, 80.897451, 0.00648145, 0.0165382]
        )
        scalar_scdt_prior = np.asarray(
            [
                (previous_best[0] - 0.30) / (2.00 - 0.30),
                (previous_best[1] - 0.05) / (0.80 - 0.05),
                (previous_best[2] - 0.15) / (0.75 - 0.15),
                (previous_best[3] - 0.02) / (0.40 - 0.02),
                (
                    np.log10(previous_best[4]) - args.ridge_log10_min
                )
                / (args.ridge_log10_max - args.ridge_log10_min),
                (previous_best[5] - 0.001) / (0.012 - 0.001),
                previous_best[6] / 0.50,
            ]
        )
        scalar_scdt_prior = np.clip(scalar_scdt_prior, 0.0, 1.0)
    elif args.architecture == "block":
        dimensions = 8
        scalar_scdt_prior = np.asarray(
            [0.286, 0.926, 0.417, 0.263, 0.756, 0.500, 0.500, 0.286]
        )
    else:
        dimensions = 7
        # Moderate memory, slow leakage, and weak graph communication are a
        # stable starting point for the topology-aware architecture.
        scalar_scdt_prior = np.asarray(
            [0.38, 0.45, 0.50, 0.30, 0.50, 0.35, 0.30]
        )
    seeded_units = [scalar_scdt_prior]
    if args.architecture == "graph-local":
        seeded_units = []
        for ridge_log10 in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            prior = scalar_scdt_prior.copy()
            prior[4] = np.clip(
                (ridge_log10 - args.ridge_log10_min)
                / (args.ridge_log10_max - args.ridge_log10_min),
                0.0,
                1.0,
            )
            seeded_units.append(prior)
        initial = max(initial, len(seeded_units) + 2)

    def make_params(unit: np.ndarray, iteration: int) -> RCParams | CoupledRCParams:
        # Keep the random topology fixed so BO compares hyperparameters rather
        # than a different reservoir realization at every iteration.
        seed = args.seed + 40000
        if args.architecture == "global":
            return parameters_from_unit(
                unit,
                seed,
                args.reservoir_size,
                args.ridge_log10_min,
                args.ridge_log10_max,
            )
        if args.architecture == "graph-local":
            return graph_local_parameters_from_unit(
                unit,
                seed,
                args.block_size,
                args.ridge_log10_min,
                args.ridge_log10_max,
            )
        return coupled_parameters_from_unit(unit, seed, args.block_size)

    for iteration in range(args.bo_calls):
        if iteration < len(seeded_units):
            unit = seeded_units[iteration]
        elif iteration < initial:
            unit = rng.random(dimensions)
        else:
            gp = GaussianProcessRegressor(
                kernel=Matern(length_scale=np.ones(dimensions), nu=2.5)
                + WhiteKernel(1e-5),
                normalize_y=True,
                random_state=args.seed,
                n_restarts_optimizer=1,
            )
            gp.fit(np.asarray(unit_points), np.asarray(scores))
            candidates = rng.random((500, dimensions))
            unit = candidates[
                int(np.argmax(expected_improvement(candidates, gp, min(scores))))
            ]
        params = make_params(unit, iteration)
        try:
            rc = build_rc(args, params)
            rc.fit(train_sequences, train_statics, args.fit_warmup)
            evaluator = (
                evaluate_normal_rc
                if args.objective in {"normal", "phase"}
                else evaluate_rc
            )
            result = evaluator(
                    rc,
                    validation,
                    coefficients,
                    dynamic_center,
                    dynamic_scale,
                    static_center,
                    static_scale,
                    args,
                    args.seed + 50000 + iteration,
                )
        except Exception as error:
            result = {"score": 1e6}
            print(f"BO candidate failed: {error}", flush=True)
        unit_points.append(unit)
        scores.append(float(result["score"]))
        row = {
            "iteration": iteration + 1,
            **{
                key: float(value)
                for key, value in result.items()
                if np.isscalar(value)
            },
            **asdict(params),
        }
        rows.append(row)
        if args.objective in {"normal", "phase"}:
            print(
                f"BO {iteration + 1:02d}/{args.bo_calls}: score={row['score']:.4f}, "
                f"phase={row['phase_nrmse']:.3f}, "
                f"collapse={row['collapse_fraction']:.2f}, "
                f"runaway={row['runaway_fraction']:.2f}, "
                f"SP/LR={row['spectral_radius']:.3f}/{row['leakage_rate']:.3f}, "
                f"ridge={row['ridge']:.3e}",
                flush=True,
            )
        else:
            print(
                f"BO {iteration + 1:02d}/{args.bo_calls}: score={row['score']:.3f}, "
                f"node0/network MAE={row['local_mae']:.3f}/{row['network_mae']:.3f}, "
                f"missed={row['missed_local']:.2f}/{row['missed_network']:.2f}, "
                f"ictal trough/leak/saturation="
                f"{row['ictal_trough_rate_error']:.2f}/"
                f"{row['ictal_spatial_leakage']:.2f}/"
                f"{row['saturation_fraction']:.2f}",
                flush=True,
            )
    best = int(np.argmin(scores))
    return make_params(unit_points[best], best), rows


def plot_static_space(
    output: Path,
    frame: pd.DataFrame,
    selected: list[str],
    coefficients: np.ndarray,
    x0_min: float,
    x0_max: float,
) -> None:
    fig, axes = plt.subplots(len(selected), 1, figsize=(9.5, 3.5 * len(selected)), squeeze=False)
    grid = np.linspace(x0_min, x0_max, 300)
    fitted = static_from_x0(grid, coefficients)
    grouped = frame.groupby("ez_x0")
    coordinates = np.asarray(sorted(frame["ez_x0"].unique()))
    for feature, (axis, name) in enumerate(zip(axes[:, 0], selected)):
        axis.axvspan(
            coordinates.min(),
            coordinates.max(),
            color="#7f7f7f",
            alpha=0.08,
            label="feature-calibration range",
        )
        axis.scatter(frame["ez_x0"], frame[name], s=22, alpha=0.35, label="stable networks")
        means = grouped[name].mean().reindex(coordinates).to_numpy()
        stds = grouped[name].std().reindex(coordinates).to_numpy()
        axis.errorbar(
            coordinates,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            lw=0,
            label="calibration mean +/- SD",
        )
        axis.plot(grid, fitted[:, feature], lw=2.0, label="precritical fit")
        axis.axvline(
            X0_CRITICAL,
            color="#d62728",
            ls="--",
            lw=1.5,
            label=rf"single-node $x_0^c={X0_CRITICAL:.1f}$",
        )
        axis.axvline(
            NETWORK_OPERATIONAL_X0,
            color="#ff7f0e",
            ls="-.",
            lw=1.5,
            label=rf"network $P(\mathrm{{event}})=0.5$ at {NETWORK_OPERATIONAL_X0:.4f}",
        )
        axis.set_ylabel(name)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2, loc="best")
    axes[-1, 0].set_xlabel("fixed node-0 excitability x0")
    fig.suptitle("Multi-node long-trajectory static coordinate", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=230)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_feature_overview(
    output_dir: Path,
    frame: pd.DataFrame,
    selected: list[str],
    ranking: pd.DataFrame,
) -> None:
    """Plot candidate groups in raw units and the leading effects as z-scores."""
    grouped = frame.groupby("ez_x0")
    coordinates = np.asarray(sorted(frame["ez_x0"].unique()))
    colors = plt.get_cmap("tab10").colors
    q_values = ranking.set_index("feature")["spearman_fdr_q"].to_dict()

    for group_name, requested_features in FEATURE_GROUPS.items():
        features = [name for name in requested_features if name in frame]
        columns = 3
        rows = int(np.ceil(len(features) / columns))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(15.5, 3.45 * rows),
            sharex=True,
            squeeze=False,
        )
        for index, (axis, name) in enumerate(zip(axes.flat, features)):
            means = grouped[name].mean().reindex(coordinates).to_numpy()
            stds = grouped[name].std().reindex(coordinates).fillna(0.0).to_numpy()
            association = spearmanr(coordinates, means)
            rho = float(association.statistic)
            p_value = float(association.pvalue)
            q_value = float(q_values[name])
            chosen = name in selected
            color = "#0072B2" if chosen else colors[index % len(colors)]
            axis.scatter(
                frame["ez_x0"],
                frame[name],
                s=18,
                color=color,
                alpha=0.18,
                edgecolors="none",
            )
            axis.errorbar(
                coordinates,
                means,
                yerr=stds,
                color=color,
                marker="o",
                markersize=4.2,
                capsize=2,
                linewidth=1.7,
            )
            trend = np.polyfit(coordinates, means, 1)
            trend_grid = np.linspace(coordinates.min(), NETWORK_OPERATIONAL_X0, 200)
            axis.plot(
                trend_grid,
                np.polyval(trend, trend_grid),
                color="0.20",
                linestyle=":",
                linewidth=1.5,
            )
            axis.axvline(
                NETWORK_OPERATIONAL_X0,
                color="#D55E00",
                linestyle="--",
                linewidth=1.7,
            )
            marker = " [selected]" if chosen else ""
            axis.set_title(
                f"{FEATURE_LABELS.get(name, name)}{marker}\n"
                f"rho={rho:+.2f}, p={p_value:.2g}, FDR q={q_value:.2g}",
                fontsize=10,
                fontweight="bold" if chosen else "normal",
            )
            axis.set_ylabel("raw value")
            axis.grid(alpha=0.20)
        for axis in axes.flat[len(features):]:
            axis.axis("off")
        for axis in axes[-1]:
            if axis.axison:
                axis.set_xlabel("fixed node-0 excitability x0")
        fig.suptitle(
            f"Network statistics: {group_name.replace('_', ' ')}",
            fontsize=17,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(output_dir / f"network_stats_{group_name}.png", dpi=220)
        fig.savefig(output_dir / f"network_stats_{group_name}.pdf")
        plt.close(fig)

    # Normalize against the three safest coordinates to compare direction and
    # effect size without hiding the original units in the raw figure.
    baseline_x0 = coordinates[: min(3, len(coordinates))]
    baseline = frame[frame["ez_x0"].isin(baseline_x0)]
    fig, axis = plt.subplots(figsize=(12.0, 6.8))
    top_features = ranking.head(12)["feature"].tolist()
    for index, name in enumerate(top_features):
        center = float(baseline[name].mean())
        scale = float(baseline[name].std(ddof=1))
        scale = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
        means = grouped[name].mean().reindex(coordinates).to_numpy()
        stds = grouped[name].std().reindex(coordinates).fillna(0.0).to_numpy()
        color = colors[index % len(colors)]
        width = 2.8 if name in selected else 1.5
        alpha = 1.0 if name in selected else 0.62
        axis.plot(
            coordinates,
            (means - center) / scale,
            marker="o",
            color=color,
            lw=width,
            alpha=alpha,
            label=FEATURE_LABELS.get(name, name),
        )
        if name in selected:
            axis.fill_between(
                coordinates,
                (means - stds - center) / scale,
                (means + stds - center) / scale,
                color=color,
                alpha=0.10,
            )
    axis.axhline(0.0, color="0.35", lw=1.0)
    axis.axvline(
        NETWORK_OPERATIONAL_X0,
        color="#D55E00",
        linestyle="--",
        linewidth=2.2,
        label="GT P(event)=0.5 boundary",
    )
    axis.set_xlabel("fixed node-0 excitability x0")
    axis.set_ylabel("baseline z-score")
    axis.set_title(
        "Top network-statistic effects near the recruitment boundary",
        fontsize=16,
        fontweight="bold",
    )
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "network_stats_top12_zscore.png", dpi=230)
    fig.savefig(output_dir / "network_stats_top12_zscore.pdf")
    plt.close(fig)


def plot_representative(
    output: Path,
    cohort: NetworkSwitchCohort,
    result: dict,
    sample_hz: float,
) -> None:
    roles = list(ROLE_EZ_X0)
    fig, axes = plt.subplots(
        2, 2, figsize=(14.0, 8.0), sharex=True, sharey=True,
        layout="constrained",
    )
    time_s = np.arange(cohort.futures_raw.shape[0]) / sample_hz
    images = []
    for column, role in enumerate(roles):
        index = np.flatnonzero(np.isclose(cohort.ez_x0, ROLE_EZ_X0[role]))[0]
        for row, (label, signals, events) in enumerate(
            [
                ("GT", cohort.futures_raw, cohort.gt_events),
                ("RC", result["predictions"], result["rc_events"]),
            ]
        ):
            axis = axes[row, column]
            image = axis.imshow(
                signals[:, index].T,
                origin="lower",
                aspect="auto",
                extent=[time_s[0], time_s[-1], -0.5, 7.5],
                cmap="RdBu_r",
                vmin=-3.0,
                vmax=3.0,
            )
            images.append(image)
            for node, onset in enumerate(events[index]):
                if np.isfinite(onset):
                    axis.scatter(onset, node, s=30, marker="D", color="black", edgecolor="white", lw=0.5)
            axis.set_title(
                f"{label} {role} (node-0 x0={ROLE_EZ_X0[role]:.3f})",
                loc="left",
                fontweight="bold",
            )
            axis.set_yticks(np.arange(8))
            axis.set_ylabel("node")
            axis.grid(False)
        axes[1, column].set_xlabel("time after static switch (s)")
    fig.colorbar(images[-1], ax=axes, shrink=0.86, label="source proxy x2-x1")
    fig.suptitle("GT vs RC multi-node recruitment", fontsize=15, fontweight="bold")
    fig.savefig(output.with_suffix(".png"), dpi=230)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_survival(output: Path, result: dict) -> None:
    gt = result["gt"]
    rc = result["rc"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    axes[0].plot(gt["x0"], gt["node_survival"][:, 0], marker="o", lw=1.8, label="GT")
    axes[0].plot(rc["x0"], rc["node_survival"][:, 0], marker="s", lw=1.8, label="RC")
    axes[0].set_title(f"Node-0 onset survival (MAE={result['local_mae']:.3f})")
    axes[0].set_ylabel("survival probability")

    axes[1].plot(gt["x0"], gt["network_survival"], marker="o", lw=1.8, label="GT")
    axes[1].plot(rc["x0"], rc["network_survival"], marker="s", lw=1.8, label="RC")
    axes[1].set_title(
        f"EZ/PZ 3-node survival (MAE={result['network_mae']:.3f})"
    )

    for node in range(4):
        axes[2].plot(
            gt["x0"],
            gt["node_probability"][:, node],
            color=NODE_COLORS[node],
            marker="o",
            lw=1.5,
            label=f"GT node {node}",
        )
        axes[2].plot(
            rc["x0"],
            rc["node_probability"][:, node],
            color=NODE_COLORS[node],
            marker="s",
            ls="--",
            lw=1.3,
            label=f"RC node {node}",
        )
    axes[2].set_title(f"EZ/PZ recruitment probability (MAE={result['pz_mae']:.3f})")
    axes[2].legend(fontsize=7, ncol=2)

    for axis in axes:
        axis.axvline(X0_CRITICAL, color="#d62728", ls="--", lw=1.1)
        axis.set_xlabel("fixed node-0 excitability x0")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.22)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("GT vs RC recruitment survival", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=230)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_normal_bo_history(
    output: Path,
    rows: list[dict],
    trajectory_horizon_s: float,
) -> None:
    frame = pd.DataFrame(rows)
    iterations = frame["iteration"].to_numpy()
    scores = frame["score"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5))
    axes[0].plot(iterations, scores, marker="o", lw=1.2, label="candidate")
    axes[0].plot(
        iterations,
        np.minimum.accumulate(scores),
        color="#d62728",
        lw=2.2,
        label="best so far",
    )
    axes[0].set_xlabel("BO iteration")
    axes[0].set_ylabel("normal-attractor objective")
    axes[0].set_title("Bayesian optimization")
    axes[0].legend()

    scatter = axes[1].scatter(
        frame["spectral_radius"],
        frame["leakage_rate"],
        c=scores,
        cmap="viridis_r",
        s=65,
        edgecolor="white",
        linewidth=0.5,
    )
    best = int(np.argmin(scores))
    axes[1].scatter(
        frame.loc[best, "spectral_radius"],
        frame.loc[best, "leakage_rate"],
        marker="*",
        s=230,
        color="#d62728",
        edgecolor="black",
        linewidth=0.8,
        label="best",
    )
    axes[1].set_xlabel("spectral radius")
    axes[1].set_ylabel("leakage rate")
    axes[1].set_title(
        f"SP-LR search colored by {trajectory_horizon_s:g} s objective"
    )
    axes[1].legend()
    fig.colorbar(
        scatter,
        ax=axes[1],
        label=f"{trajectory_horizon_s:g} s objective",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=230)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_normal_validation_2x2(
    output: Path,
    cohort: NetworkSwitchCohort,
    result: dict,
    sample_hz: float,
) -> None:
    predictions = result["predictions"]
    conditions = [float(np.min(cohort.ez_x0)), float(np.max(cohort.ez_x0))]
    selected = [int(np.flatnonzero(np.isclose(cohort.ez_x0, value))[0]) for value in conditions]
    time_s = np.arange(len(cohort.futures_raw), dtype=float) / sample_hz
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 7.5), sharex=True, sharey=True)
    y_min = min(
        float(np.min(cohort.futures_raw[:, selected, 0])),
        float(np.min(predictions[:, selected, 0])),
    )
    y_max = max(
        float(np.max(cohort.futures_raw[:, selected, 0])),
        float(np.max(predictions[:, selected, 0])),
    )
    labels = [["GT", "GT"], ["RC", "RC"]]
    for column, (condition, index) in enumerate(zip(conditions, selected)):
        axes[0, column].plot(time_s, cohort.futures_raw[:, index, 0], lw=1.0)
        axes[1, column].plot(time_s, predictions[:, index, 0], lw=1.0)
        for row in range(2):
            axes[row, column].set_title(
                f"({chr(97 + row)}{column + 1}) {labels[row][column]}, "
                rf"$x_0={condition:.2f}$",
                loc="left",
                fontweight="bold",
            )
            axes[row, column].set_ylim(y_min - 0.08, y_max + 0.08)
            axes[row, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("node 0 activity")
    axes[1, 0].set_ylabel("node 0 activity")
    axes[1, 0].set_xlabel("rollout time (s)")
    axes[1, 1].set_xlabel("rollout time (s)")
    fig.suptitle("Normal-only BO: GT and autonomous RC rollout", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=230)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_short_segment_validation_2x2(
    output: Path,
    cohort: NetworkSwitchCohort,
    result: dict,
    sample_hz: float,
    segments_per_trajectory: int,
) -> None:
    predictions = result["segment_predictions"]
    targets = result["segment_targets"]
    segment_x0 = result["segment_x0"]
    conditions = [float(np.min(cohort.ez_x0)), float(np.max(cohort.ez_x0))]
    base_indices = [
        int(np.flatnonzero(np.isclose(cohort.ez_x0, value))[0])
        for value in conditions
    ]
    time_s = np.arange(len(targets), dtype=float) / sample_hz
    batch = len(cohort.ez_x0)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, segments_per_trajectory))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), sharex=True, sharey=True)
    for column, (condition, base_index) in enumerate(zip(conditions, base_indices)):
        for segment, color in enumerate(colors):
            index = segment * batch + base_index
            if not np.isclose(segment_x0[index], condition):
                raise RuntimeError("normal segment batch ordering is inconsistent")
            axes[0, column].plot(
                time_s,
                targets[:, index, 0],
                color=color,
                lw=1.15,
                alpha=0.82,
            )
            axes[1, column].plot(
                time_s,
                predictions[:, index, 0],
                color=color,
                lw=1.15,
                alpha=0.82,
            )
        axes[0, column].set_title(
            rf"(a{column + 1}) GT segments, $x_0={condition:.2f}$",
            loc="left",
            fontweight="bold",
        )
        axes[1, column].set_title(
            rf"(b{column + 1}) RC segments, $x_0={condition:.2f}$",
            loc="left",
            fontweight="bold",
        )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("node 0 activity")
    axes[1, 0].set_ylabel("node 0 activity")
    axes[1, 0].set_xlabel("short rollout time (s)")
    axes[1, 1].set_xlabel("short rollout time (s)")
    fig.suptitle(
        f"Multi-start normal validation: {segments_per_trajectory} autonomous rollouts",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=230)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--architecture",
        choices=["global", "block", "graph-local"],
        default="global",
    )
    parser.add_argument(
        "--objective",
        choices=["normal", "phase", "full"],
        default="normal",
    )
    parser.add_argument("--long-trajectory-s", type=float, default=160.0)
    parser.add_argument("--warmup-s", type=float, default=20.0)
    parser.add_argument("--rollout-s", type=float, default=140.0)
    parser.add_argument("--rc-stride", type=int, default=8)
    parser.add_argument("--feature-seeds", type=int, default=4)
    parser.add_argument("--train-seeds", type=int, default=2)
    parser.add_argument("--validation-seeds", type=int, default=4)
    parser.add_argument("--survival-seeds", type=int, default=10)
    parser.add_argument("--static-dim", type=int, default=len(SCDT_STATIC_FEATURES))
    parser.add_argument("--correlation-limit", type=float, default=0.92)
    parser.add_argument("--fit-warmup", type=int, default=160)
    parser.add_argument("--ramp-steps", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=200)
    parser.add_argument("--reservoir-size", type=int, default=4000)
    parser.add_argument("--readout-size", type=int, default=800)
    parser.add_argument("--readout-stride", type=int, default=4)
    parser.add_argument(
        "--normal-short-s",
        type=float,
        default=4.0,
        help="Autonomous horizon of each normal-only BO segment in seconds.",
    )
    parser.add_argument(
        "--normal-segment-warmup-s",
        type=float,
        default=6.0,
        help="GT teacher-forcing duration before each short normal rollout.",
    )
    parser.add_argument(
        "--normal-segments-per-trajectory",
        type=int,
        default=5,
        help="Number of evenly spaced rollout starts per validation trajectory.",
    )
    parser.add_argument(
        "--phase-max-lag-s",
        type=float,
        default=0.50,
        help="Maximum temporal alignment allowed by the phase-aware objective.",
    )
    parser.add_argument("--collapse-std-ratio", type=float, default=0.30)
    parser.add_argument("--collapse-derivative-ratio", type=float, default=0.20)
    parser.add_argument("--runaway-std-ratio", type=float, default=3.0)
    parser.add_argument("--runaway-mean-z", type=float, default=4.0)
    parser.add_argument("--ridge-log10-min", type=float, default=-6.0)
    parser.add_argument("--ridge-log10-max", type=float, default=2.0)
    parser.add_argument("--bo-calls", type=int, default=12)
    parser.add_argument("--survival-x0", type=int, default=21)
    parser.add_argument("--x0-min", type=float, default=-2.25)
    parser.add_argument("--x0-max", type=float, default=-1.93)
    parser.add_argument("--base-ez-x0", type=float, default=-2.18)
    parser.add_argument("--network-event-nodes", type=int, default=3)
    parser.add_argument("--global-coupling", type=float, default=1.0)
    parser.add_argument("--noise-variance", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=130901)
    parser.add_argument("--feature-seed-base", type=int, default=230901)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help=(
            "Optional CSV produced by analyze_multinode_network_features.py. "
            "When supplied, reuse those calibrated event-free statistics."
        ),
    )
    parser.add_argument("--train-seed-base", type=int, default=330901)
    parser.add_argument("--validation-seed-base", type=int, default=430901)
    parser.add_argument("--test-seed-base", type=int, default=530901)
    args = parser.parse_args()

    if np.any(TRAIN_EZ_X0 > NETWORK_TRAIN_X0_CEILING):
        raise RuntimeError("Training x0 exceeds the calibrated event-free band")
    if args.base_ez_x0 >= X0_CRITICAL:
        raise RuntimeError("Warmup node-0 x0 must be precritical")
    if not 1 <= args.network_event_nodes <= 8:
        raise ValueError("network-event-nodes must be between 1 and 8")
    if args.architecture in {"block", "graph-local"} and args.block_size < 10:
        raise ValueError("block-size must be at least 10")
    if args.reservoir_size < 10:
        raise ValueError("reservoir-size must be at least 10")
    if args.readout_stride < 1:
        raise ValueError("readout-stride must be positive")
    if args.readout_size < 1 or args.readout_size > args.reservoir_size:
        raise ValueError("readout-size must be between 1 and reservoir-size")
    if args.normal_short_s <= 0.0 or args.normal_segment_warmup_s <= 0.0:
        raise ValueError("normal segment durations must be positive")
    if args.normal_segments_per_trajectory < 2:
        raise ValueError("normal-segments-per-trajectory must be at least two")
    if args.phase_max_lag_s < 0.0:
        raise ValueError("phase-max-lag-s cannot be negative")
    if args.ridge_log10_min >= args.ridge_log10_max:
        raise ValueError("ridge-log10-min must be smaller than ridge-log10-max")
    if args.static_dim != len(SCDT_STATIC_FEATURES):
        raise ValueError(
            "static-dim must match the explicit SCDT static feature set "
            f"({len(SCDT_STATIC_FEATURES)})"
        )

    experiment_name = f"paper_multinode_{args.architecture}_{args.objective}_bo"
    data_dir = args.out_dir / "data" / experiment_name
    figure_dir = args.out_dir / "figures" / "RC" / experiment_name
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sample_hz = 256.0 / args.rc_stride

    if args.feature_cache is not None:
        feature_frame = pd.read_csv(args.feature_cache)
        missing = [
            name
            for name in ["ez_x0", *SCDT_STATIC_FEATURES]
            if name not in feature_frame.columns
        ]
        if missing:
            raise ValueError(f"feature-cache is missing columns: {missing}")
        cached_x0 = np.sort(feature_frame["ez_x0"].unique())
        if not np.allclose(cached_x0, FEATURE_EZ_X0):
            raise ValueError("feature-cache x0 grid does not match FEATURE_EZ_X0")
        print(f"Reused feature calibration: {args.feature_cache}", flush=True)
    else:
        feature_frame = build_feature_table(args)
    ranking = rank_features(feature_frame)
    selected = list(SCDT_STATIC_FEATURES)
    selected_ranking = ranking.set_index("feature").loc[selected]
    if np.any(selected_ranking["spearman_fdr_q"].to_numpy() >= 0.05):
        raise RuntimeError("A fixed SCDT static feature failed the FDR screen")
    observed_static, fitted_static, coefficients = fit_static_mapping(
        feature_frame,
        selected,
    )
    print(f"Selected multi-node static features: {selected}", flush=True)

    raw_sequences, train_coordinates, training_rows = build_training_sequences(args)
    dynamic_values = np.concatenate(raw_sequences, axis=0)
    dynamic_center = np.mean(dynamic_values, axis=0)
    dynamic_scale = np.std(dynamic_values, axis=0)
    dynamic_scale = np.where(dynamic_scale > 1e-9, dynamic_scale, 1.0)
    train_sequences = [
        normalize(sequence, dynamic_center, dynamic_scale)
        for sequence in raw_sequences
    ]
    train_static_raw = static_from_x0(train_coordinates, coefficients)
    static_center, static_scale = fit_normalizer(
        static_from_x0(TRAIN_EZ_X0, coefficients)
    )
    train_statics = [
        row for row in normalize(train_static_raw, static_center, static_scale)
    ]

    validation_x0 = (
        NORMAL_VALIDATION_EZ_X0
        if args.objective in {"normal", "phase"}
        else VALIDATION_EZ_X0
    )
    validation = build_switch_cohort(
        validation_x0,
        args.validation_seeds,
        args,
        args.validation_seed_base,
    )
    params, bo_rows = optimize_rc(
        args,
        train_sequences,
        train_statics,
        validation,
        coefficients,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
    )
    rc = build_rc(args, params)
    rc.fit(train_sequences, train_statics, args.fit_warmup)
    normal_validation_result = (
        evaluate_normal_rc(
            rc,
            validation,
            coefficients,
            dynamic_center,
            dynamic_scale,
            static_center,
            static_scale,
            args,
            args.seed + 80000,
            include_full_rollout=True,
        )
        if args.objective in {"normal", "phase"}
        else None
    )

    test_grid = np.linspace(args.x0_min, args.x0_max, args.survival_x0)
    test = build_switch_cohort(
        test_grid,
        args.survival_seeds,
        args,
        args.test_seed_base,
    )
    test_result = evaluate_rc(
        rc,
        test,
        coefficients,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        args,
        args.seed + 90000,
    )
    role_cohort = build_switch_cohort(
        np.asarray(list(ROLE_EZ_X0.values())),
        1,
        args,
        args.test_seed_base + 90000,
    )
    role_result = evaluate_rc(
        rc,
        role_cohort,
        coefficients,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        args,
        args.seed + 100000,
    )

    feature_frame.to_csv(data_dir / "long_trajectory_features.csv", index=False)
    ranking.to_csv(data_dir / "feature_ranking.csv", index=False)
    write_rows(data_dir / "training_trajectories.csv", training_rows)
    write_rows(data_dir / "bo_history.csv", bo_rows)

    survival_rows = []
    node_rows = []
    for condition, value in enumerate(test_result["gt"]["x0"]):
        survival_rows.append(
            {
                "node0_x0": value,
                "gt_node0_survival": test_result["gt"]["node_survival"][condition, 0],
                "rc_node0_survival": test_result["rc"]["node_survival"][condition, 0],
                "gt_network_survival": test_result["gt"]["network_survival"][condition],
                "rc_network_survival": test_result["rc"]["network_survival"][condition],
            }
        )
        for node in range(8):
            node_rows.append(
                {
                    "node0_x0": value,
                    "node": node,
                    "gt_recruitment_probability": test_result["gt"]["node_probability"][condition, node],
                    "rc_recruitment_probability": test_result["rc"]["node_probability"][condition, node],
                    "gt_median_onset_s": test_result["gt"]["median_onset"][condition, node],
                    "rc_median_onset_s": test_result["rc"]["median_onset"][condition, node],
                }
            )
    write_rows(data_dir / "survival_rates.csv", survival_rows)
    write_rows(data_dir / "node_recruitment.csv", node_rows)

    if args.architecture == "global":
        architecture_record = {
            "reservoir": "one monolithic sparse reservoir",
            "nodes": 8,
            "total_units": args.reservoir_size,
            "recurrent_topology": "global random sparse matrix without node blocks",
            "readout": "joint eight-channel readout from a fixed random subset of globally coupled states",
            "readout_state_subset": args.readout_size,
            "readout_stride": args.readout_stride,
            "static_routing": "selected static features to every reservoir unit",
        }
    elif args.architecture == "block":
        architecture_record = {
            "reservoir": "one graph-coupled reservoir",
            "nodes": 8,
            "block_size": args.block_size,
            "total_units": 8 * args.block_size,
            "inter_block_topology": "synthetic structural proxy",
            "readout": "node-local block state plus current eight-channel signal",
            "static_routing": "all selected static features to all blocks",
        }
    else:
        architecture_record = {
            "reservoir": "one topology-aware graph-local reservoir",
            "nodes": 8,
            "block_size": args.block_size,
            "total_units": 8 * args.block_size,
            "dynamic_input": "self and direct structural-neighbour channels only",
            "inter_block_topology": "row-normalized synthetic structural proxy",
            "readout": "node-local block only; no direct dynamic-input skip",
            "readout_stride": args.readout_stride,
            "static_routing": "network SCDT coordinate broadcast to all blocks",
        }
    if args.objective == "normal":
        objective_record = {
            "name": "multi-start short normal trajectory rollout only",
            "formula": (
                "mean segment-wise trajectory RMSE divided by each segment's "
                "own temporal standard deviation"
            ),
            "segment_rollout_s": args.normal_short_s,
            "segment_warmup_s": args.normal_segment_warmup_s,
            "segments_per_trajectory": args.normal_segments_per_trajectory,
            "validation_segment_count": (
                len(validation.ez_x0) * args.normal_segments_per_trajectory
            ),
            "statistical_terms": False,
            "seizure_terms": False,
            "recruitment_terms": False,
            "constraint_penalties": False,
        }
    elif args.objective == "phase":
        objective_record = {
            "name": "phase-tolerant normal-attractor preservation",
            "formula": (
                "0.25 clipped lag-aligned NRMSE + distribution NRMSE + "
                "0.05 clipped pointwise NRMSE + 0.60 log-SD error + "
                "0.40 log-derivative-RMS error + 0.25 lag-1 AC error + "
                "spectral-centroid error + 0.75 peak-rate error + finite "
                "collapse and runaway penalties"
            ),
            "segment_rollout_s": args.normal_short_s,
            "segment_warmup_s": args.normal_segment_warmup_s,
            "segments_per_trajectory": args.normal_segments_per_trajectory,
            "maximum_phase_lag_s": args.phase_max_lag_s,
            "collapse_std_ratio": args.collapse_std_ratio,
            "collapse_derivative_ratio": args.collapse_derivative_ratio,
            "runaway_std_ratio": args.runaway_std_ratio,
            "runaway_mean_z": args.runaway_mean_z,
            "seizure_terms": False,
            "recruitment_terms": False,
        }
    else:
        objective_record = {"name": "full recruitment and morphology objective"}

    protocol = {
        "experiment": f"{args.architecture}-reservoir Network Epileptor SCDT",
        "rollout_alignment": (
            "last_observation_once_v2"
            if args.architecture == "graph-local" else "legacy"
        ),
        "architecture": architecture_record,
        "objective": objective_record,
        "bo_search_space": {
            "spectral_radius": [0.30, 2.00],
            "leakage_rate": [0.05, 0.80],
            "ridge": [
                10.0 ** args.ridge_log10_min,
                10.0 ** args.ridge_log10_max,
            ],
            "ridge_scale": "log10",
        },
        "controlled_condition": "fixed node-0 excitability per trial",
        "event_definition": (
            "first prominent trough <= -2 with at least three prominent "
            "troughs in the following 2 s"
        ),
        "training_claim": "calibrated event-free network trajectories only",
        "clinical_short_window_claim": False,
        "training_node0_x0": TRAIN_EZ_X0.tolist(),
        "feature_calibration_node0_x0": FEATURE_EZ_X0.tolist(),
        "validation_node0_x0": validation_x0.tolist(),
        "selected_features": selected,
        "static_feature_policy": {
            "selection": "fixed increasing pre-recruitment SCDT coordinate",
            "components": {
                "ez_std": "local fluctuation amplitude",
            },
            "first_ablation": SCDT_STATIC_ABLATION,
            "future_value_policy": (
                "linear precritical x0-to-feature extrapolation for the "
                "simulation proof-of-concept; causal deployment must use "
                "statistics from the current past-only EEG window"
            ),
        },
        "feature_coefficients": coefficients.tolist(),
        "observed_training_static": observed_static.tolist(),
        "fitted_training_static": fitted_static.tolist(),
        "dynamic_center": dynamic_center.tolist(),
        "dynamic_scale": dynamic_scale.tolist(),
        "static_center": static_center.tolist(),
        "static_scale": static_scale.tolist(),
        "best_rc_parameters": asdict(params),
        "best_validation_metrics": min(bo_rows, key=lambda row: row["score"]),
        "test_metrics": {
            key: test_result[key]
            for key in [
                "local_mae",
                "network_mae",
                "pz_mae",
                "hz_false_recruitment",
                "short_nrmse",
                "recruitment_count_mae",
                "onset_timing_mae",
                "ictal_variability_deficit",
                "ictal_derivative_deficit",
                "ictal_trough_rate_deficit",
                "ictal_variability_error",
                "ictal_derivative_error",
                "ictal_trough_rate_error",
                "ictal_spatial_leakage",
                "saturation_fraction",
            ]
        },
        "arguments": vars(args)
        | {
            "out_dir": str(args.out_dir),
            "feature_cache": (
                str(args.feature_cache) if args.feature_cache is not None else None
            ),
        },
    }
    (data_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    plot_static_space(
        figure_dir / "static_space",
        feature_frame,
        selected,
        coefficients,
        args.x0_min,
        args.x0_max,
    )
    plot_feature_overview(figure_dir, feature_frame, selected, ranking)
    plot_normal_bo_history(
        figure_dir / "bo_normal_objective",
        bo_rows,
        args.normal_short_s,
    )
    if normal_validation_result is not None:
        plot_normal_validation_2x2(
            figure_dir / "normal_validation_2x2",
            validation,
            normal_validation_result,
            sample_hz,
        )
        plot_short_segment_validation_2x2(
            figure_dir / "short_segment_validation_2x2",
            validation,
            normal_validation_result,
            sample_hz,
            args.normal_segments_per_trajectory,
        )
    plot_representative(
        figure_dir / "gt_rc_recruitment",
        role_cohort,
        role_result,
        sample_hz,
    )
    plot_survival(figure_dir / "survival_rates", test_result)

    print(f"\n{args.architecture.title()} multi-node SCDT result")
    if args.architecture == "global":
        print(f"  reservoir architecture: one global {args.reservoir_size}-unit reservoir")
    else:
        print(
            f"  reservoir architecture: 8 blocks x {args.block_size} units "
            f"= {8 * args.block_size} total"
        )
    if normal_validation_result is not None:
        print(
            "  normal validation score / short NRMSE: "
            f"{normal_validation_result['score']:.4f} / "
            f"{normal_validation_result['short_nrmse']:.4f}"
        )
    print(f"  selected static features: {selected}")
    print(f"  node-0 survival MAE: {test_result['local_mae']:.4f}")
    print(f"  network survival MAE: {test_result['network_mae']:.4f}")
    print(f"  PZ recruitment MAE: {test_result['pz_mae']:.4f}")
    print(
        "  ictal trough error / HZ leakage / saturation: "
        f"{test_result['ictal_trough_rate_error']:.4f} / "
        f"{test_result['ictal_spatial_leakage']:.4f} / "
        f"{test_result['saturation_fraction']:.4f}"
    )
    print(
        "  HZ false recruitment probability: "
        f"{test_result['hz_false_recruitment']:.4f}"
    )
    print(f"  best RC parameters: {asdict(params)}")
    print(f"  outputs: {figure_dir}")


if __name__ == "__main__":
    main()
