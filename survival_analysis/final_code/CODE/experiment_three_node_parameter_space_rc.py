#!/usr/bin/env python3
"""Three-node Epileptor parameter-space experiment with a general reservoir.

Each trajectory is generated at one fixed three-dimensional excitability
coordinate (x0_1, x0_2, x0_3).  All three coordinates vary across the cohort.
The 1,500-unit monolithic reservoir receives the three signals plus nine
trajectory-context statistics computed only from the observed warmup.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from experiment_three_node_bifurcation_rc import (
    N_NODES,
    RC_HZ,
    causal_statistics,
    event_counts,
    first_minimum_events,
    first_network_event,
    latin_hypercube,
    normalize,
    save_json,
    simulate_paths,
    three_node_connectivity,
)
from paper_epileptor_core import X0_CRITICAL
from paper_multinode_coupled_scdt import MultiOutputStaticRC
from paper_single_node_scrc import RCParams


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_NAMES = tuple(
    f"node_{node + 1}_{stat}"
    for stat in ("mean", "std", "ac1")
    for node in range(N_NODES)
)


def sample_profiles(count: int, seed: int) -> np.ndarray:
    """Sample the 3-D x0 space without holding any node coordinate fixed."""
    return latin_hypercube(count, -2.50, -1.90, np.random.default_rng(seed))


def trajectory_context(warmup: np.ndarray) -> np.ndarray:
    """Nine statistics from the observed prefix only."""
    return causal_statistics(warmup, len(warmup))[-1]


def make_cohort(
    count: int,
    seed: int,
    duration_s: float,
    observation_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate and risk-stratify trajectories with event-free observations."""
    candidate_count = max(count * 3, count + 12)
    profiles = sample_profiles(candidate_count, seed)
    signals, _ = simulate_paths(
        profiles,
        profiles,
        duration_s=duration_s,
        hold_s=0.0,
        seed=seed + 1,
    )
    full_events = first_network_event(first_minimum_events(signals))
    eligible = np.flatnonzero(~np.isfinite(full_events) | (full_events > observation_s))
    if len(eligible) < count:
        raise RuntimeError(
            f"Only {len(eligible)} of {candidate_count} profiles have event-free observations"
        )
    # Evenly cover the weakest-to-strongest coordinate among eligible profiles.
    order = eligible[np.argsort(np.max(profiles[eligible], axis=1))]
    positions = np.linspace(0, len(order) - 1, count).round().astype(int)
    chosen = order[positions]
    chosen_profiles = profiles[chosen]
    chosen_signals = signals[:, chosen]
    chosen_events = full_events[chosen]
    observation_samples = int(round(observation_s * RC_HZ))
    context = trajectory_context(chosen_signals[:observation_samples])
    return chosen_profiles, chosen_signals, context, chosen_events


def fit_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values).reshape(-1, values.shape[-1])
    center = flat.mean(axis=0)
    scale = flat.std(axis=0)
    return center, np.where(scale > 1.0e-8, scale, 1.0)


class ResidualMultiOutputStaticRC(MultiOutputStaticRC):
    """Monolithic ESN whose readout advances the signal by a learned increment."""

    target_mode = "delta"

    def fit(
        self,
        sequences: list[np.ndarray],
        statics: list[np.ndarray],
        warmup: int,
    ) -> None:
        all_values = np.concatenate(sequences, axis=0)
        span = np.ptp(all_values, axis=0)
        self.clip_low = np.min(all_values, axis=0) - 1.0 * span
        self.clip_high = np.max(all_values, axis=0) + 1.0 * span
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) != 1:
            raise ValueError("global reservoir batching requires equal sequence lengths")
        sequence_batch = np.stack(sequences, axis=1)
        static_batch = np.stack(statics, axis=0).T
        batch = len(sequences)
        sampled_steps = len(range(warmup, len(sequence_batch) - 1, self.readout_stride))
        columns = sampled_steps * batch
        features = np.empty(
            (columns, self.readout_size + self.dynamic_dim + 1),
            dtype=np.float32,
        )
        targets = np.empty((columns, self.dynamic_dim), dtype=np.float32)
        column = 0
        state = np.zeros((self.params.n_reservoir, batch))
        for index in range(len(sequence_batch) - 1):
            current = sequence_batch[index].T
            state = self.update(state, current, static_batch)
            if index >= warmup and (index - warmup) % self.readout_stride == 0:
                next_column = column + batch
                features[column:next_column] = self.augment(state, current).T
                targets[column:next_column] = sequence_batch[index + 1] - sequence_batch[index]
                column = next_column
        model = Ridge(alpha=self.params.ridge, fit_intercept=False, solver="cholesky")
        model.fit(features, targets)
        self.w_out = np.asarray(model.coef_, dtype=float)
        residuals = targets - model.predict(features)
        low = np.quantile(residuals, 0.0025, axis=0)
        high = np.quantile(residuals, 0.9975, axis=0)
        self.residuals = np.clip(residuals, low, high)

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
        batch = warmup.shape[1]
        state = np.zeros((self.params.n_reservoir, batch))
        warmup_static = warmup_statics.T
        target_static = target_statics.T
        for index in range(len(warmup)):
            state = self.update(state, warmup[index].T, warmup_static)
        current = warmup[-1].T
        prediction = np.empty((n_steps, batch, self.dynamic_dim), dtype=float)
        rng = np.random.default_rng(seed)
        for index in range(n_steps):
            if ramp_steps > 0 and index < ramp_steps:
                fraction = (index + 1) / ramp_steps
                static = (1.0 - fraction) * warmup_static + fraction * target_static
            else:
                static = target_static
            state = self.update(state, current, static)
            delta = self.w_out @ self.augment(state, current)
            residual_index = rng.integers(0, len(self.residuals), size=batch)
            delta += self.params.innovation_scaling * self.residuals[residual_index].T
            current = np.clip(
                current + delta,
                self.clip_low[:, None],
                self.clip_high[:, None],
            )
            prediction[index] = current.T
        return prediction


def teacher_forced(
    reservoir: MultiOutputStaticRC,
    signals: np.ndarray,
    context: np.ndarray,
    start: int,
) -> np.ndarray:
    time, batch, nodes = signals.shape
    state = np.zeros((reservoir.params.n_reservoir, batch), dtype=float)
    output = np.full((time, batch, nodes), np.nan)
    static = context.T
    for index in range(time - 1):
        current = signals[index].T
        state = reservoir.update(state, current, static)
        if index >= start:
            estimate = reservoir.w_out @ reservoir.augment(state, current)
            if getattr(reservoir, "target_mode", "next") == "delta":
                estimate = current + estimate
            output[index + 1] = estimate.T
    return output


def context_associations(profiles: np.ndarray, context: np.ndarray) -> dict[str, list[float] | float]:
    std_rho, ac1_rho = [], []
    for node in range(N_NODES):
        std_rho.append(float(spearmanr(profiles[:, node], context[:, N_NODES + node]).statistic))
        ac1_rho.append(float(spearmanr(profiles[:, node], context[:, 2 * N_NODES + node]).statistic))
    return {
        "rho_x0_std_by_node": std_rho,
        "rho_x0_ac1_by_node": ac1_rho,
        "mean_rho_x0_std": float(np.mean(std_rho)),
        "mean_rho_x0_ac1": float(np.mean(ac1_rho)),
    }


def parameter_observer(
    train_context: np.ndarray,
    train_profiles: np.ndarray,
    test_context: np.ndarray,
    test_profiles: np.ndarray,
) -> tuple[Ridge, np.ndarray, dict[str, list[float] | float]]:
    """Test whether observed statistics encode the three x0 coordinates."""
    model = Ridge(alpha=10.0).fit(train_context, train_profiles)
    prediction = model.predict(test_context)
    rmse = np.sqrt(np.mean((prediction - test_profiles) ** 2, axis=0))
    rho = [
        float(spearmanr(prediction[:, node], test_profiles[:, node]).statistic)
        for node in range(N_NODES)
    ]
    true_risk = np.max(test_profiles - X0_CRITICAL, axis=1)
    predicted_risk = np.max(prediction - X0_CRITICAL, axis=1)
    return model, prediction, {
        "x0_observer_rmse_by_node": rmse.tolist(),
        "x0_observer_rho_by_node": rho,
        "x0_observer_mean_rmse": float(np.mean(rmse)),
        "risk_coordinate_rho": float(spearmanr(predicted_risk, true_risk).statistic),
    }


def horizon_event_metrics(gt: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, int]]:
    output = {}
    for horizon_s in (20, 40, 60, 120):
        count = min(len(gt), horizon_s * RC_HZ)
        gt_network = first_network_event(first_minimum_events(gt[:count]))
        rc_network = first_network_event(first_minimum_events(prediction[:count]))
        output[str(horizon_s)] = {
            "gt_exits": int(np.isfinite(gt_network).sum()),
            "rc_exits": int(np.isfinite(rc_network).sum()),
            **event_counts(gt_network, rc_network),
        }
    return output


def plot_space(
    figdir: Path,
    train: np.ndarray,
    test: np.ndarray,
    events: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*train.T, c="#9fb6aa", s=30, alpha=0.65, label="training")
    event_mask = np.isfinite(events)
    ax.scatter(*test[~event_mask].T, c="#326c91", s=65, marker="o", label="test: no exit")
    ax.scatter(*test[event_mask].T, c=events[event_mask], cmap="magma_r", s=80, marker="^", label="test: future exit")
    critical = np.linspace(-2.5, -1.9, 2)
    aa, bb = np.meshgrid(critical, critical)
    ax.plot_surface(np.full_like(aa, X0_CRITICAL), aa, bb, alpha=0.035, color="red")
    ax.plot_surface(aa, np.full_like(aa, X0_CRITICAL), bb, alpha=0.035, color="red")
    ax.plot_surface(aa, bb, np.full_like(aa, X0_CRITICAL), alpha=0.035, color="red")
    ax.set(xlabel="$x_{0,1}$", ylabel="$x_{0,2}$", zlabel="$x_{0,3}$",
           title="Sampled three-node bifurcation parameter space")
    ax.legend(fontsize=8)
    fig.savefig(figdir / "parameter_space.png", dpi=180)
    plt.close(fig)


def plot_context(figdir: Path, profiles: np.ndarray, context: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for node in range(N_NODES):
        coordinate = profiles[:, node] - X0_CRITICAL
        axes[0, node].scatter(coordinate, context[:, N_NODES + node], c="#176b3a", s=28, alpha=0.7)
        axes[1, node].scatter(coordinate, context[:, 2 * N_NODES + node], c="#38598b", s=28, alpha=0.7)
        for ax in axes[:, node]:
            ax.axvline(0.0, color="black", ls="--", lw=0.9)
            ax.grid(alpha=0.18)
        axes[0, node].set_title(f"Node {node + 1}")
        axes[1, node].set_xlabel("$x_{0,i}-x_0^c$")
    axes[0, 0].set_ylabel("Observed-prefix standard deviation")
    axes[1, 0].set_ylabel("Observed-prefix lag-1 autocorrelation")
    fig.suptitle("Per-trajectory observable context near the reference boundary")
    fig.savefig(figdir / "context_vs_boundary.png", dpi=180)
    plt.close(fig)


def plot_parameter_observer(
    figdir: Path,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    limits = (-2.53, -1.87)
    for node, ax in enumerate(axes):
        ax.scatter(truth[:, node], prediction[:, node], c="#176b3a", s=45, alpha=0.8)
        ax.plot(limits, limits, color="black", ls="--", lw=0.9)
        ax.axvline(X0_CRITICAL, color="#a33a2b", ls=":", lw=0.9)
        ax.axhline(X0_CRITICAL, color="#a33a2b", ls=":", lw=0.9)
        ax.set(
            xlim=limits,
            ylim=limits,
            xlabel="True x0",
            ylabel="Statistics-predicted x0",
            title=f"Node {node + 1}",
        )
        ax.grid(alpha=0.18)
    fig.suptitle("Held-out parameter coordinates decoded from observed statistics")
    fig.savefig(figdir / "parameter_observer.png", dpi=180)
    plt.close(fig)


def plot_rollout(
    figdir: Path,
    warmup: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray,
    gt_events: np.ndarray,
) -> None:
    finite = np.flatnonzero(np.isfinite(gt_events))
    case = int(finite[0]) if finite.size else 0
    observed_s = len(warmup) / RC_HZ
    warm_time = np.arange(len(warmup)) / RC_HZ
    future_time = observed_s + (np.arange(len(gt)) + 1) / RC_HZ
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout="constrained")
    for node, ax in enumerate(axes):
        ax.plot(warm_time, warmup[:, case, node], color="#777777", lw=0.9, label="observed prefix")
        ax.plot(future_time, gt[:, case, node], color="black", lw=1.0, label="GT")
        ax.plot(future_time, prediction[:, case, node], color="#176b3a", lw=1.0, label="RC")
        ax.axhline(-2.0, color="#a33a2b", ls="--", lw=0.8)
        ax.axvline(observed_s, color="#555555", ls=":", lw=0.9)
        ax.set_ylabel(f"Node {node + 1}")
        ax.grid(alpha=0.18)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Display time (s)")
    fig.suptitle(f"Fixed x0 held-out rollout | test profile {case}")
    fig.savefig(figdir / "rollout_example.png", dpi=180)
    plt.close(fig)


def write_results(path: Path, metrics: dict) -> None:
    path.write_text(
        "# Three-node parameter-space reservoir pilot\n\n"
        "Each trajectory uses one fixed x0 vector, while all three coordinates vary "
        "across the sampled cohort. A 1,500-unit monolithic reservoir receives three "
        "signals and nine observed-prefix statistics. No x0 coordinate is an RC input.\n\n"
        f"- Teacher-forced normalized RMSE: {metrics['teacher_rmse_normalized']:.6f}\n"
        f"- Autonomous normalized RMSE: {metrics['autonomous_rmse_normalized']:.6f}\n"
        f"- GT / RC future network exits: {metrics['gt_network_exits']} / {metrics['rc_network_exits']} "
        f"of {metrics['test_profiles']}\n"
        f"- Hits / misses / false alarms / correct non-exits: {metrics['hits']} / "
        f"{metrics['misses']} / {metrics['false_alarms']} / {metrics['correct_nonexits']}\n"
        f"- Mean rho(x0, observed-prefix std): {metrics['mean_rho_x0_std']:.4f}\n"
        f"- Mean rho(x0, observed-prefix ac1): {metrics['mean_rho_x0_ac1']:.4f}\n"
        f"- Statistics-to-x0 observer RMSE by node: "
        f"{', '.join(f'{value:.4f}' for value in metrics['x0_observer_rmse_by_node'])}\n"
        f"- Statistics-to-x0 observer rho by node: "
        f"{', '.join(f'{value:.4f}' for value in metrics['x0_observer_rho_by_node'])}\n\n"
        "Only profiles without an operational event during the 40-second observation "
        "prefix are evaluated. The source threshold is a synthetic project endpoint, "
        "not a clinical seizure label.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "three_node_parameter_space_rc")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "figures" / "RC" / "three_node_parameter_space_rc")
    parser.add_argument("--train-profiles", type=int, default=24)
    parser.add_argument("--test-profiles", type=int, default=12)
    parser.add_argument("--duration", type=float, default=160.0)
    parser.add_argument("--observation", type=float, default=40.0)
    parser.add_argument("--reservoir-size", type=int, default=1500)
    parser.add_argument("--readout-size", type=int, default=1500)
    parser.add_argument("--spectral-radius", type=float, default=0.20)
    parser.add_argument("--leakage-rate", type=float, default=0.05)
    parser.add_argument("--input-scaling", type=float, default=0.08)
    parser.add_argument("--static-scaling", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--target-mode", choices=("delta", "next"), default="delta")
    parser.add_argument("--seed", type=int, default=260911)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    if args.observation >= args.duration:
        raise ValueError("observation must be shorter than duration")

    train_profiles, train_raw, train_context, train_full_events = make_cohort(
        args.train_profiles, args.seed, args.duration, args.observation
    )
    test_profiles, test_raw, test_context, test_full_events = make_cohort(
        args.test_profiles, args.seed + 1000, args.duration, args.observation
    )
    dynamic_center, dynamic_scale = fit_normalizer(train_raw)
    context_center, context_scale = fit_normalizer(train_context)
    train_n = normalize(train_raw, dynamic_center, dynamic_scale)
    test_n = normalize(test_raw, dynamic_center, dynamic_scale)
    train_context_n = normalize(train_context, context_center, context_scale)
    test_context_n = normalize(test_context, context_center, context_scale)

    rc_config = RCParams(
        n_reservoir=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        leakage_rate=args.leakage_rate,
        input_scaling=args.input_scaling,
        static_scaling=args.static_scaling,
        ridge=args.ridge,
        density=0.003,
        innovation_scaling=0.0,
        seed=args.seed + 2000,
    )
    reservoir_class = ResidualMultiOutputStaticRC if args.target_mode == "delta" else MultiOutputStaticRC
    reservoir = reservoir_class(
        rc_config,
        dynamic_dim=N_NODES,
        static_dim=len(CONTEXT_NAMES),
        readout_stride=4,
        readout_size=args.readout_size,
    )
    reservoir.fit(
        [train_n[:, index] for index in range(args.train_profiles)],
        [train_context_n[index] for index in range(args.train_profiles)],
        warmup=5 * RC_HZ,
    )
    observation_samples = int(round(args.observation * RC_HZ))
    teacher = teacher_forced(reservoir, test_n, test_context_n, observation_samples)
    teacher_rmse = float(
        np.sqrt(np.nanmean((teacher[observation_samples + 1 :] - test_n[observation_samples + 1 :]) ** 2))
    )
    prediction_n = reservoir.rollout_switch_batch(
        test_n[:observation_samples],
        test_context_n,
        test_context_n,
        n_steps=len(test_n) - observation_samples,
        seed=args.seed + 3000,
    )
    prediction = prediction_n * dynamic_scale + dynamic_center
    gt_future = test_raw[observation_samples:]
    autonomous_rmse = float(np.sqrt(np.mean(((prediction - gt_future) / dynamic_scale) ** 2)))
    gt_events = first_minimum_events(gt_future)
    rc_events = first_minimum_events(prediction)
    gt_network = first_network_event(gt_events)
    rc_network = first_network_event(rc_events)
    associations = context_associations(
        np.concatenate([train_profiles, test_profiles]),
        np.concatenate([train_context, test_context]),
    )
    observer, observed_x0, observer_metrics = parameter_observer(
        train_context_n,
        train_profiles,
        test_context_n,
        test_profiles,
    )
    metrics = {
        "teacher_rmse_normalized": teacher_rmse,
        "autonomous_rmse_normalized": autonomous_rmse,
        "train_profiles": args.train_profiles,
        "test_profiles": args.test_profiles,
        "gt_network_exits": int(np.isfinite(gt_network).sum()),
        "rc_network_exits": int(np.isfinite(rc_network).sum()),
        **event_counts(gt_network, rc_network),
        **associations,
        **observer_metrics,
        "event_metrics_by_horizon_s": horizon_event_metrics(gt_future, prediction),
    }
    protocol = {
        "status": "complete",
        "date": "2026-09-10",
        "nodes": N_NODES,
        "parameter_space": "fixed x0 vector per trajectory; every x0_i varies across profiles",
        "x0_range": [-2.50, -1.90],
        "x0_input_to_rc": False,
        "dynamic_inputs": ["source_1", "source_2", "source_3"],
        "trajectory_context": list(CONTEXT_NAMES),
        "context_source": "first 40 display seconds only",
        "event_free_observation_filter": True,
        "rc_hz": RC_HZ,
        "reservoir": asdict(rc_config),
        "readout_size": int(reservoir.readout_size),
        "readout_target": args.target_mode,
        "duration_s": args.duration,
        "observation_s": args.observation,
        "connectivity": three_node_connectivity().tolist(),
        "critical_reference": X0_CRITICAL,
        "event_endpoint": "first interior local minimum at or below -2 in any node",
        "seed": args.seed,
    }
    save_json(args.out_dir / "protocol.json", protocol)
    save_json(args.out_dir / "summary.json", metrics)
    np.savez_compressed(
        args.out_dir / "training.npz",
        profiles=train_profiles,
        signals=train_raw,
        context=train_context,
        full_events=train_full_events,
    )
    np.savez_compressed(
        args.out_dir / "validation.npz",
        profiles=test_profiles,
        signals=test_raw,
        context=test_context,
        full_events=test_full_events,
        prediction=prediction,
        teacher_forced_normalized=teacher,
        observed_x0=observed_x0,
        gt_events=gt_events,
        rc_events=rc_events,
    )
    np.savez_compressed(
        args.out_dir / "model.npz",
        a_data=reservoir.a.data,
        a_indices=reservoir.a.indices,
        a_indptr=reservoir.a.indptr,
        a_shape=reservoir.a.shape,
        w_in=reservoir.w_in,
        w_static=reservoir.w_static,
        w_out=reservoir.w_out,
        residuals=reservoir.residuals,
        readout_indices=reservoir.readout_indices,
        dynamic_center=dynamic_center,
        dynamic_scale=dynamic_scale,
        context_center=context_center,
        context_scale=context_scale,
        observer_coef=observer.coef_,
        observer_intercept=observer.intercept_,
    )
    with (args.out_dir / "profile_metrics.csv").open("w", newline="") as handle:
        fields = ["profile", "gt_event_s", "rc_event_s"] + [f"x0_{i + 1}" for i in range(3)] + list(CONTEXT_NAMES)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(args.test_profiles):
            writer.writerow(
                {
                    "profile": index,
                    "gt_event_s": gt_network[index],
                    "rc_event_s": rc_network[index],
                    **{f"x0_{node + 1}": test_profiles[index, node] for node in range(3)},
                    **{name: test_context[index, column] for column, name in enumerate(CONTEXT_NAMES)},
                }
            )
    write_results(args.out_dir / "RESULTS.md", metrics)
    plot_space(args.fig_dir, train_profiles, test_profiles, gt_network)
    plot_context(figdir=args.fig_dir, profiles=np.concatenate([train_profiles, test_profiles]), context=np.concatenate([train_context, test_context]))
    plot_parameter_observer(args.fig_dir, test_profiles, observed_x0)
    plot_rollout(args.fig_dir, test_raw[:observation_samples], gt_future, prediction, gt_network)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
