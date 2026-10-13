#!/usr/bin/env python3
"""Three-node bifurcation-path experiment with a 1,500-unit reservoir.

All three Epileptor excitability coordinates vary smoothly within a trial.
The reservoir never receives x0 directly.  Its causal input is the three
source-proxy samples plus per-node rolling mean, standard deviation, and
lag-1 autocorrelation.  During autonomous rollout those statistics are
recomputed from the generated signal history.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from paper_epileptor_core import PaperEpileptorConfig, X0_CRITICAL, source_proxy
from paper_multinode_coupled_scdt import initial_state_batch, network_heun_step_batch


ROOT = Path(__file__).resolve().parents[1]
N_NODES = 3
RC_HZ = 32
STAT_NAMES = ("mean", "std", "ac1")


@dataclass(frozen=True)
class ReservoirConfig:
    n_reservoir: int = 1500
    spectral_radius: float = 0.85
    leakage_rate: float = 0.25
    input_scaling: float = 0.35
    density: float = 0.003
    ridge: float = 1.0e-3
    readout_size: int = 750
    innovation_scaling: float = 0.10
    seed: int = 260910


def three_node_connectivity() -> np.ndarray:
    """Transparent symmetric triangle used only as a synthetic proxy."""
    return np.asarray(
        [[0.0, 1.0, 0.45], [1.0, 0.0, 0.70], [0.45, 0.70, 0.0]],
        dtype=float,
    )


def latin_hypercube(
    count: int,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.empty((count, N_NODES), dtype=float)
    for node in range(N_NODES):
        bins = (np.arange(count) + rng.random(count)) / count
        values[:, node] = low + (high - low) * bins[rng.permutation(count)]
    return values


def parameter_paths(
    count: int,
    seed: int,
    start_range: tuple[float, float] = (-2.55, -2.35),
    end_range: tuple[float, float] = (-2.18, -1.90),
) -> tuple[np.ndarray, np.ndarray]:
    """Create diverse paths for which every node changes excitability."""
    rng = np.random.default_rng(seed)
    starts = latin_hypercube(count, *start_range, rng)
    ends = latin_hypercube(count, *end_range, rng)
    ends = np.maximum(ends, starts + 0.16)
    if np.any(np.isclose(starts, ends)):
        raise AssertionError("Every x0 coordinate must change within every path")
    return starts, ends


def smooth_fraction(sample: np.ndarray | float, hold_fraction: float) -> np.ndarray:
    phase = np.clip((np.asarray(sample) - hold_fraction) / (1.0 - hold_fraction), 0.0, 1.0)
    return phase * phase * (3.0 - 2.0 * phase)


def x0_schedule(
    starts: np.ndarray,
    ends: np.ndarray,
    n_samples: int,
    hold_fraction: float,
) -> np.ndarray:
    fractions = smooth_fraction(
        (np.arange(n_samples, dtype=float) + 1.0) / n_samples,
        hold_fraction,
    )
    return starts[None, :, :] + fractions[:, None, None] * (ends - starts)[None, :, :]


def simulate_paths(
    starts: np.ndarray,
    ends: np.ndarray,
    duration_s: float,
    hold_s: float,
    seed: int,
    coupling: float = 0.7,
    noise_variance: float = 0.0025,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate independent batches of three-node parameter paths."""
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if starts.shape != ends.shape or starts.shape[1:] != (N_NODES,):
        raise ValueError("starts and ends must have shape (batch, 3)")
    config = PaperEpileptorConfig(
        n_nodes=N_NODES,
        duration_s=duration_s,
        global_coupling=coupling,
        noise_variance=noise_variance,
        seed=seed,
    )
    stride = int(round(config.sampling_hz / RC_HZ))
    internal_steps = int(round(duration_s * config.sampling_hz))
    n_samples = internal_steps // stride
    hold_fraction = hold_s / duration_s
    schedule = x0_schedule(starts, ends, n_samples, hold_fraction)
    states = initial_state_batch(len(starts), N_NODES, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    signals = np.empty((n_samples, len(starts), N_NODES), dtype=float)
    out = 0
    for step in range(internal_steps):
        fraction = smooth_fraction((step + 1.0) / internal_steps, hold_fraction)
        current_x0 = starts + fraction * (ends - starts)
        states = network_heun_step_batch(
            states,
            current_x0,
            three_node_connectivity(),
            config,
            rng,
        )
        if not np.all(np.isfinite(states)):
            raise FloatingPointError(f"Non-finite Epileptor state at internal step {step}")
        if (step + 1) % stride == 0:
            signals[out] = source_proxy(states)
            out += 1
    return signals, schedule


def causal_statistics(signals: np.ndarray, window: int) -> np.ndarray:
    """Return causal rolling mean/std/ac1 for each node.

    Input has shape (time, batch, node); output has shape
    (time, batch, 3 statistics * node).  Values at t never use samples > t.
    """
    values = np.asarray(signals, dtype=float)
    if values.ndim != 3 or values.shape[-1] != N_NODES:
        raise ValueError("signals must have shape (time, batch, 3)")
    if window < 2:
        raise ValueError("window must be at least two samples")
    time, batch, nodes = values.shape
    means = np.empty_like(values)
    stds = np.empty_like(values)
    ac1 = np.zeros_like(values)
    cumulative = np.concatenate([np.zeros((1, batch, nodes)), np.cumsum(values, axis=0)])
    cumulative_sq = np.concatenate([np.zeros((1, batch, nodes)), np.cumsum(values * values, axis=0)])
    for index in range(time):
        begin = max(0, index - window + 1)
        length = index - begin + 1
        total = cumulative[index + 1] - cumulative[begin]
        total_sq = cumulative_sq[index + 1] - cumulative_sq[begin]
        means[index] = total / length
        stds[index] = np.sqrt(np.maximum(total_sq / length - means[index] ** 2, 0.0))
        if length >= 3:
            segment = values[begin : index + 1]
            centered = segment - means[index][None, :, :]
            denominator = np.sum(centered * centered, axis=0)
            numerator = np.sum(centered[:-1] * centered[1:], axis=0)
            ac1[index] = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 1.0e-12,
            )
    return np.concatenate([means, stds, ac1], axis=-1)


def fit_normalizer(values: np.ndarray, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values[start:], dtype=float).reshape(-1, values.shape[-1])
    center = flat.mean(axis=0)
    scale = flat.std(axis=0)
    return center, np.where(scale > 1.0e-8, scale, 1.0)


def normalize(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - center) / scale


class GeneralStatReservoir:
    """One monolithic sparse reservoir driven by signals and causal statistics."""

    def __init__(self, config: ReservoirConfig, input_dim: int = 12):
        self.config = config
        self.input_dim = int(input_dim)
        rng = np.random.default_rng(config.seed)
        self.w_in = rng.uniform(
            -config.input_scaling,
            config.input_scaling,
            (config.n_reservoir, self.input_dim),
        )
        nnz = max(1, int(config.density * config.n_reservoir**2))
        matrix = sparse.coo_matrix(
            (
                rng.normal(size=nnz),
                (
                    rng.integers(0, config.n_reservoir, size=nnz),
                    rng.integers(0, config.n_reservoir, size=nnz),
                ),
            ),
            shape=(config.n_reservoir, config.n_reservoir),
        ).tocsr()
        radius = float(np.max(np.abs(eigs(matrix, k=1, which="LM", return_eigenvectors=False))))
        self.a = matrix * (config.spectral_radius / max(radius, 1.0e-9))
        size = min(config.n_reservoir, config.readout_size)
        self.readout_indices = np.sort(rng.choice(config.n_reservoir, size=size, replace=False))
        self.w_out: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.clip_low = np.full(N_NODES, -8.0)
        self.clip_high = np.full(N_NODES, 8.0)

    def update(self, state: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        activation = self.a @ state + self.w_in @ inputs
        return (
            (1.0 - self.config.leakage_rate) * state
            + self.config.leakage_rate * np.tanh(activation)
        )

    def augment(self, state: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        selected = state[self.readout_indices].copy()
        selected[1::2] = selected[1::2] ** 2
        return np.vstack([selected, inputs, np.ones((1, state.shape[1]))])

    def fit(
        self,
        signals: np.ndarray,
        statistics: np.ndarray,
        washout: int,
        readout_stride: int,
    ) -> dict[str, float]:
        time, batch, _ = signals.shape
        state = np.zeros((self.config.n_reservoir, batch), dtype=float)
        rows = len(range(washout, time - 1, readout_stride)) * batch
        width = len(self.readout_indices) + self.input_dim + 1
        features = np.empty((rows, width), dtype=np.float32)
        targets = np.empty((rows, N_NODES), dtype=np.float32)
        row = 0
        for index in range(time - 1):
            inputs = np.concatenate([signals[index], statistics[index]], axis=-1).T
            state = self.update(state, inputs)
            if index >= washout and (index - washout) % readout_stride == 0:
                next_row = row + batch
                features[row:next_row] = self.augment(state, inputs).T
                targets[row:next_row] = signals[index + 1]
                row = next_row
        model = Ridge(alpha=self.config.ridge, fit_intercept=False, solver="cholesky")
        model.fit(features, targets)
        self.w_out = np.asarray(model.coef_, dtype=float)
        residuals = targets - model.predict(features)
        low, high = np.quantile(residuals, [0.0025, 0.9975], axis=0)
        self.residuals = np.clip(residuals, low, high)
        prediction = model.predict(features)
        return {
            "fit_rows": int(rows),
            "feature_width": int(width),
            "teacher_rmse_normalized": float(np.sqrt(np.mean((prediction - targets) ** 2))),
        }

    def teacher_forced(
        self,
        signals: np.ndarray,
        statistics: np.ndarray,
        start: int,
    ) -> np.ndarray:
        if self.w_out is None:
            raise RuntimeError("Reservoir must be fitted")
        time, batch, _ = signals.shape
        state = np.zeros((self.config.n_reservoir, batch), dtype=float)
        output = np.full_like(signals, np.nan)
        for index in range(time - 1):
            inputs = np.concatenate([signals[index], statistics[index]], axis=-1).T
            state = self.update(state, inputs)
            if index >= start:
                output[index + 1] = (self.w_out @ self.augment(state, inputs)).T
        return output

    def rollout(
        self,
        warmup_raw: np.ndarray,
        n_steps: int,
        stat_window: int,
        dynamic_center: np.ndarray,
        dynamic_scale: np.ndarray,
        stat_center: np.ndarray,
        stat_scale: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        if self.w_out is None or self.residuals is None:
            raise RuntimeError("Reservoir must be fitted")
        warm_stats = causal_statistics(warmup_raw, stat_window)
        warm_signal_n = normalize(warmup_raw, dynamic_center, dynamic_scale)
        warm_stats_n = normalize(warm_stats, stat_center, stat_scale)
        batch = warmup_raw.shape[1]
        state = np.zeros((self.config.n_reservoir, batch), dtype=float)
        for index in range(len(warmup_raw)):
            inputs = np.concatenate([warm_signal_n[index], warm_stats_n[index]], axis=-1).T
            state = self.update(state, inputs)
        history = warmup_raw[-stat_window:].copy()
        current_raw = warmup_raw[-1].copy()
        output = np.empty((n_steps, batch, N_NODES), dtype=float)
        rng = np.random.default_rng(seed)
        for index in range(n_steps):
            current_stats = causal_statistics(history, stat_window)[-1]
            signal_n = normalize(current_raw, dynamic_center, dynamic_scale)
            stats_n = normalize(current_stats, stat_center, stat_scale)
            inputs = np.concatenate([signal_n, stats_n], axis=-1).T
            state = self.update(state, inputs)
            next_n = (self.w_out @ self.augment(state, inputs)).T
            residual_index = rng.integers(0, len(self.residuals), size=batch)
            next_n += self.config.innovation_scaling * self.residuals[residual_index]
            next_raw = next_n * dynamic_scale + dynamic_center
            next_raw = np.clip(next_raw, self.clip_low, self.clip_high)
            output[index] = next_raw
            history = np.concatenate([history[1:], next_raw[None, :, :]], axis=0)
            current_raw = next_raw
        return output


def first_minimum_events(signals: np.ndarray, threshold: float = -2.0) -> np.ndarray:
    values = np.asarray(signals, dtype=float)
    local = np.zeros_like(values, dtype=bool)
    local[1:-1] = (values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])
    events = np.full(values.shape[1:], np.nan)
    for batch in range(values.shape[1]):
        for node in range(values.shape[2]):
            hits = np.flatnonzero(local[:, batch, node] & (values[:, batch, node] <= threshold))
            if hits.size:
                events[batch, node] = (hits[0] + 1) / RC_HZ
    return events


def first_network_event(events: np.ndarray) -> np.ndarray:
    values = np.where(np.isfinite(events), events, np.inf).min(axis=1)
    return np.where(np.isfinite(values), values, np.nan)


def event_counts(gt: np.ndarray, rc: np.ndarray) -> dict[str, int]:
    gt_hit = np.isfinite(gt)
    rc_hit = np.isfinite(rc)
    return {
        "hits": int(np.sum(gt_hit & rc_hit)),
        "misses": int(np.sum(gt_hit & ~rc_hit)),
        "false_alarms": int(np.sum(~gt_hit & rc_hit)),
        "correct_nonexits": int(np.sum(~gt_hit & ~rc_hit)),
    }


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def plot_parameter_space(
    figdir: Path,
    train_start: np.ndarray,
    train_end: np.ndarray,
    test_start: np.ndarray,
    test_end: np.ndarray,
    events: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    for start, end in zip(train_start, train_end):
        ax.plot(*np.vstack([start, end]).T, color="#8ba69a", alpha=0.35, lw=1)
    colors = np.where(np.isfinite(events), events, np.nanmax(np.where(np.isfinite(events), events, 0)) + 30)
    for start, end, color in zip(test_start, test_end, colors):
        ax.plot(*np.vstack([start, end]).T, color="#163f2b", alpha=0.7, lw=1.8)
        ax.scatter(*end, c=[color], cmap="viridis_r", vmin=0, vmax=max(1, np.nanmax(colors)), s=55)
    ax.set(xlabel="$x_{0,1}$", ylabel="$x_{0,2}$", zlabel="$x_{0,3}$",
           title="Three-node paths through bifurcation parameter space")
    fig.savefig(figdir / "parameter_paths.png", dpi=180)
    plt.close(fig)


def plot_statistics(figdir: Path, x0: np.ndarray, statistics: np.ndarray) -> None:
    sample = slice(None, None, 16)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, layout="constrained")
    for node in range(N_NODES):
        coordinate = (x0[sample, :, node] - X0_CRITICAL).ravel()
        std = statistics[sample, :, N_NODES + node].ravel()
        ac1 = statistics[sample, :, 2 * N_NODES + node].ravel()
        axes[0, node].scatter(coordinate, std, s=5, alpha=0.22, color="#176b3a")
        axes[1, node].scatter(coordinate, ac1, s=5, alpha=0.22, color="#38598b")
        axes[0, node].axvline(0, ls="--", color="black", lw=1)
        axes[1, node].axvline(0, ls="--", color="black", lw=1)
        axes[0, node].set_title(f"Node {node + 1}")
        axes[1, node].set_xlabel("$x_{0,i}-x_0^c$")
    axes[0, 0].set_ylabel("Rolling standard deviation")
    axes[1, 0].set_ylabel("Rolling lag-1 autocorrelation")
    fig.suptitle("Observable statistics while each excitability coordinate approaches the boundary")
    fig.savefig(figdir / "statistics_vs_boundary.png", dpi=180)
    plt.close(fig)


def plot_rollout(
    figdir: Path,
    warmup: np.ndarray,
    gt: np.ndarray,
    rc: np.ndarray,
    gt_event: np.ndarray,
) -> None:
    finite = np.flatnonzero(np.isfinite(gt_event))
    case = int(finite[0]) if finite.size else 0
    observed_s = len(warmup) / RC_HZ
    future_time = observed_s + (np.arange(len(gt)) + 1) / RC_HZ
    warm_time = np.arange(len(warmup)) / RC_HZ
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout="constrained")
    for node, ax in enumerate(axes):
        ax.plot(warm_time, warmup[:, case, node], color="#777777", lw=0.9, label="observed warmup")
        ax.plot(future_time, gt[:, case, node], color="black", lw=1.0, label="GT")
        ax.plot(future_time, rc[:, case, node], color="#176b3a", lw=1.0, alpha=0.9, label="RC")
        ax.axhline(-2.0, color="#a33a2b", ls="--", lw=0.8)
        ax.axvline(observed_s, color="#555555", ls=":", lw=0.9)
        ax.set_ylabel(f"Node {node + 1}")
        ax.grid(alpha=0.18)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Display time (s)")
    fig.suptitle(f"Held-out autonomous rollout | path {case}")
    fig.savefig(figdir / "rollout_example.png", dpi=180)
    plt.close(fig)


def write_results(path: Path, metrics: dict) -> None:
    path.write_text(
        "# Three-node bifurcation-path RC pilot\n\n"
        "All three node-wise excitability coordinates vary smoothly. The 1,500-unit "
        "monolithic reservoir receives three source signals and causal per-node rolling "
        "mean, standard deviation, and lag-1 autocorrelation; x0 is not an input.\n\n"
        f"- Teacher-forced normalized RMSE: {metrics['teacher_rmse_normalized']:.6f}\n"
        f"- Autonomous normalized RMSE: {metrics['autonomous_rmse_normalized']:.6f}\n"
        f"- GT network exits: {metrics['gt_network_exits']}/{metrics['test_paths']}\n"
        f"- RC network exits: {metrics['rc_network_exits']}/{metrics['test_paths']}\n"
        f"- Hits / misses / false alarms / correct non-exits: "
        f"{metrics['hits']} / {metrics['misses']} / {metrics['false_alarms']} / {metrics['correct_nonexits']}\n"
        f"- Mean node-wise Spearman rho(x0, rolling std): {metrics['mean_rho_x0_std']:.4f}\n"
        f"- Mean node-wise Spearman rho(x0, rolling ac1): {metrics['mean_rho_x0_ac1']:.4f}\n\n"
        "This is a synthetic-method pilot. The threshold is an operational project endpoint, "
        "not a clinical seizure label. Long autonomous waveform RMSE is descriptive because "
        "phase errors accumulate in nonlinear rollout.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "three_node_bifurcation_rc")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "figures" / "RC" / "three_node_bifurcation_rc")
    parser.add_argument("--train-paths", type=int, default=18)
    parser.add_argument("--test-paths", type=int, default=9)
    parser.add_argument("--train-duration", type=float, default=120.0)
    parser.add_argument("--test-duration", type=float, default=160.0)
    parser.add_argument("--observation", type=float, default=40.0)
    parser.add_argument("--hold", type=float, default=16.0)
    parser.add_argument("--stat-window", type=float, default=2.0)
    parser.add_argument("--reservoir-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=260910)
    args = parser.parse_args()
    if args.observation >= args.test_duration:
        raise ValueError("observation must be shorter than test-duration")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    train_start, train_end = parameter_paths(args.train_paths, args.seed)
    test_start, test_end = parameter_paths(args.test_paths, args.seed + 1000)
    train_raw, train_x0 = simulate_paths(
        train_start, train_end, args.train_duration, args.hold, args.seed + 10
    )
    test_raw, test_x0 = simulate_paths(
        test_start, test_end, args.test_duration, args.hold, args.seed + 20
    )
    stat_window = int(round(args.stat_window * RC_HZ))
    train_stats = causal_statistics(train_raw, stat_window)
    test_stats = causal_statistics(test_raw, stat_window)
    dynamic_center, dynamic_scale = fit_normalizer(train_raw, stat_window)
    stat_center, stat_scale = fit_normalizer(train_stats, stat_window)
    train_signal_n = normalize(train_raw, dynamic_center, dynamic_scale)
    train_stats_n = normalize(train_stats, stat_center, stat_scale)
    test_signal_n = normalize(test_raw, dynamic_center, dynamic_scale)
    test_stats_n = normalize(test_stats, stat_center, stat_scale)

    config = ReservoirConfig(n_reservoir=args.reservoir_size, seed=args.seed)
    reservoir = GeneralStatReservoir(config)
    reservoir.clip_low = train_raw.min(axis=(0, 1)) - 1.5 * np.ptp(train_raw, axis=(0, 1))
    reservoir.clip_high = train_raw.max(axis=(0, 1)) + 1.0 * np.ptp(train_raw, axis=(0, 1))
    fit_info = reservoir.fit(
        train_signal_n,
        train_stats_n,
        washout=int(round(5 * RC_HZ)),
        readout_stride=4,
    )

    observation_samples = int(round(args.observation * RC_HZ))
    teacher_n = reservoir.teacher_forced(test_signal_n, test_stats_n, observation_samples)
    teacher_slice = slice(observation_samples + 1, None)
    teacher_rmse = float(
        np.sqrt(np.nanmean((teacher_n[teacher_slice] - test_signal_n[teacher_slice]) ** 2))
    )
    future_steps = len(test_raw) - observation_samples
    autonomous = reservoir.rollout(
        test_raw[:observation_samples],
        future_steps,
        stat_window,
        dynamic_center,
        dynamic_scale,
        stat_center,
        stat_scale,
        args.seed + 30,
    )
    gt_future = test_raw[observation_samples:]
    autonomous_rmse = float(
        np.sqrt(np.mean(((autonomous - gt_future) / dynamic_scale) ** 2))
    )
    gt_events = first_minimum_events(gt_future)
    rc_events = first_minimum_events(autonomous)
    gt_network = first_network_event(gt_events)
    rc_network = first_network_event(rc_events)
    counts = event_counts(gt_network, rc_network)

    rhos_std, rhos_ac1 = [], []
    for node in range(N_NODES):
        coordinate = test_x0[::16, :, node].ravel()
        std = test_stats[::16, :, N_NODES + node].ravel()
        ac1 = test_stats[::16, :, 2 * N_NODES + node].ravel()
        rhos_std.append(float(spearmanr(coordinate, std).statistic))
        rhos_ac1.append(float(spearmanr(coordinate, ac1).statistic))

    metrics = {
        **fit_info,
        "teacher_rmse_normalized": teacher_rmse,
        "autonomous_rmse_normalized": autonomous_rmse,
        "train_paths": args.train_paths,
        "test_paths": args.test_paths,
        "gt_network_exits": int(np.isfinite(gt_network).sum()),
        "rc_network_exits": int(np.isfinite(rc_network).sum()),
        **counts,
        "rho_x0_std_by_node": rhos_std,
        "rho_x0_ac1_by_node": rhos_ac1,
        "mean_rho_x0_std": float(np.mean(rhos_std)),
        "mean_rho_x0_ac1": float(np.mean(rhos_ac1)),
    }
    protocol = {
        "status": "complete",
        "date": "2026-09-10",
        "nodes": N_NODES,
        "x0_policy": "all three x0_i vary smoothly on every path",
        "x0_input_to_rc": False,
        "dynamic_inputs": ["source_1", "source_2", "source_3"],
        "causal_statistics_per_node": list(STAT_NAMES),
        "stat_window_s": args.stat_window,
        "rc_hz": RC_HZ,
        "reservoir": asdict(config),
        "train_duration_s": args.train_duration,
        "test_duration_s": args.test_duration,
        "observation_s": args.observation,
        "hold_s": args.hold,
        "connectivity": three_node_connectivity().tolist(),
        "critical_reference": X0_CRITICAL,
        "event_endpoint": "first interior local minimum at or below -2 in any node",
        "seeds": {"base": args.seed, "train_ode": args.seed + 10, "test_ode": args.seed + 20},
    }
    save_json(args.out_dir / "protocol.json", protocol)
    save_json(args.out_dir / "summary.json", metrics)
    np.savez_compressed(
        args.out_dir / "training.npz",
        signals=train_raw,
        statistics=train_stats,
        x0=train_x0,
        starts=train_start,
        ends=train_end,
    )
    np.savez_compressed(
        args.out_dir / "validation.npz",
        signals=test_raw,
        statistics=test_stats,
        x0=test_x0,
        autonomous=autonomous,
        teacher_forced_normalized=teacher_n,
        gt_events=gt_events,
        rc_events=rc_events,
        starts=test_start,
        ends=test_end,
    )
    np.savez_compressed(
        args.out_dir / "model.npz",
        a_data=reservoir.a.data,
        a_indices=reservoir.a.indices,
        a_indptr=reservoir.a.indptr,
        a_shape=reservoir.a.shape,
        w_in=reservoir.w_in,
        w_out=reservoir.w_out,
        residuals=reservoir.residuals,
        readout_indices=reservoir.readout_indices,
        dynamic_center=dynamic_center,
        dynamic_scale=dynamic_scale,
        stat_center=stat_center,
        stat_scale=stat_scale,
    )
    with (args.out_dir / "path_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "gt_event_s", "rc_event_s", "start_x01", "start_x02", "start_x03", "end_x01", "end_x02", "end_x03"],
        )
        writer.writeheader()
        for index in range(args.test_paths):
            writer.writerow(
                {
                    "path": index,
                    "gt_event_s": gt_network[index],
                    "rc_event_s": rc_network[index],
                    **{f"start_x0{node + 1}": test_start[index, node] for node in range(N_NODES)},
                    **{f"end_x0{node + 1}": test_end[index, node] for node in range(N_NODES)},
                }
            )
    write_results(args.out_dir / "RESULTS.md", metrics)
    plot_parameter_space(args.fig_dir, train_start, train_end, test_start, test_end, gt_network)
    plot_statistics(args.fig_dir, test_x0, test_stats)
    plot_rollout(args.fig_dir, test_raw[:observation_samples], gt_future, autonomous, gt_network)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
