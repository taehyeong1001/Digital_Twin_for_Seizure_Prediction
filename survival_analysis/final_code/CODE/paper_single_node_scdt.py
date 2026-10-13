#!/usr/bin/env python3
"""SCDT-style main experiment with single-node Epileptor EEG dynamics.

Long precritical trajectories define a low-dimensional statistical coordinate.
The reservoir learns the corresponding family of dynamics. At test time the
coordinate is interpolated or extrapolated, the reservoir is warmed on a safe
branch, and a long autonomous rollout is scored by the same collapse criterion
as the ground-truth Epileptor.
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
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from paper_epileptor_core import (
    PaperEpileptorConfig,
    X0_CRITICAL,
    default_initial_state,
    detect_local_minimum_onset,
    source_proxy,
    stochastic_heun_step,
)
from paper_single_node_feature_discovery import FEATURE_CATALOG, extract_features
from paper_single_node_scrc import (
    RCParams,
    SparseStaticRC,
    expected_improvement,
    fit_normalizer,
    normalize,
    simulate_batch,
    static_from_x0,
)


TRAIN_X0 = np.asarray([-2.30, -2.27, -2.24, -2.21, -2.18, -2.15])
# Balanced across the empirical survival transition. This prevents a trivial
# always-surviving reservoir from winning the Bayesian objective.
VALIDATION_X0 = np.asarray(
    [-2.255, -2.195, -2.125, -2.085, -2.065, -2.045, -2.025, -1.985]
)
ROLE_X0 = {
    "inter": -2.2125,
    "pre": -2.1125,
    "post": -2.0375,
}

# Offset-sensitive locations and redundant transforms are excluded from the
# primary observation coordinate. They remain available in the full catalog.
STATIC_CANDIDATES = [
    "std",
    "mad",
    "iqr",
    "skewness",
    "kurtosis",
    "peak_to_peak",
    "line_length",
    "zero_crossing_rate",
    "ac_lag1",
    "ac_250ms",
    "hjorth_mobility",
    "hjorth_complexity",
    "envelope_std",
    "envelope_p95",
    "dominant_frequency",
    "spectral_centroid",
    "spectral_entropy",
    "relative_power_0p1_1",
    "relative_power_1_4",
    "relative_power_4_8",
    "relative_power_8_16",
    "relative_power_16_30",
    "permutation_entropy",
    "late_outlier_fraction",
    "burst_rate",
    "max_abs_robust_z",
]


@dataclass
class SwitchCohort:
    x0: np.ndarray
    warmups_raw: np.ndarray
    futures_raw: np.ndarray
    gt_events: np.ndarray


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def simulate_x0_switch_batch(
    target_x0: np.ndarray,
    base_x0: float,
    warmup_s: float,
    rollout_s: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a stable branch followed by a fixed-x0 regime switch."""
    targets = np.asarray(target_x0, dtype=float)
    config = PaperEpileptorConfig(
        n_nodes=len(targets),
        duration_s=warmup_s + rollout_s,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    states = default_initial_state(config.n_nodes, rng)
    connectivity = np.zeros((config.n_nodes, config.n_nodes))
    warmup_steps = int(round(warmup_s * config.sampling_hz))
    rollout_steps = int(round(rollout_s * config.sampling_hz))
    warmup_proxy = np.empty((warmup_steps, config.n_nodes))
    future_proxy = np.empty((rollout_steps, config.n_nodes))
    base = np.full(config.n_nodes, base_x0)
    for index in range(warmup_steps):
        states = stochastic_heun_step(states, base, connectivity, config, rng)
        warmup_proxy[index] = source_proxy(states)
    for index in range(rollout_steps):
        states = stochastic_heun_step(states, targets, connectivity, config, rng)
        future_proxy[index] = source_proxy(states)
    return warmup_proxy, future_proxy


def event_time(signal: np.ndarray, sample_hz: float) -> float | None:
    time_s = np.arange(len(signal), dtype=float) / sample_hz
    return detect_local_minimum_onset(time_s, signal)


def event_times(signals: np.ndarray, sample_hz: float) -> np.ndarray:
    values = []
    for index in range(signals.shape[1]):
        onset = event_time(signals[:, index], sample_hz)
        values.append(np.nan if onset is None else onset)
    return np.asarray(values)


def stable_long_trajectories(
    x0: float,
    duration_s: float,
    requested: int,
    seed: int,
    rc_stride: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Return fixed-x0 trajectories that remain on the precritical branch."""
    batch_size = max(requested * 3, requested + 2)
    time_s, proxy = simulate_batch(np.full(batch_size, x0), duration_s, seed)
    sampled = proxy[::rc_stride]
    selected: list[np.ndarray] = []
    selected_indices: list[int] = []
    for index in range(batch_size):
        if detect_local_minimum_onset(time_s, proxy[:, index]) is None:
            selected.append(sampled[:, index].copy())
            selected_indices.append(index)
        if len(selected) == requested:
            break
    if len(selected) < requested:
        raise RuntimeError(
            f"Only {len(selected)} stable trajectories at x0={x0}; "
            f"requested {requested}."
        )
    return selected, selected_indices


def build_long_feature_table(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    sample_hz = 256.0 / args.rc_stride
    for x0_index, x0 in enumerate(TRAIN_X0):
        trajectories, indices = stable_long_trajectories(
            float(x0),
            args.long_trajectory_s,
            args.feature_seeds,
            args.feature_seed_base + 1000 * x0_index,
            args.rc_stride,
        )
        for replicate, (signal, source_index) in enumerate(zip(trajectories, indices)):
            rows.append(
                {
                    "x0": float(x0),
                    "replicate": replicate,
                    "source_index": source_index,
                    **extract_features(signal, sample_hz),
                }
            )
        print(
            f"feature calibration x0={x0:.3f}: "
            f"{len(trajectories)} stable long trajectories",
            flush=True,
        )
    return pd.DataFrame(rows)


def feature_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = frame.groupby("x0")
    x0 = np.asarray(sorted(frame["x0"].unique()))
    for name in STATIC_CANDIDATES:
        means = grouped[name].mean().reindex(x0).to_numpy(dtype=float)
        within_std = grouped[name].std().reindex(x0).fillna(0.0).to_numpy(dtype=float)
        if not np.all(np.isfinite(means)) or np.ptp(means) <= 1e-12:
            continue
        slope, intercept = np.polyfit(x0, means, 1)
        fitted = slope * x0 + intercept
        total = np.sum((means - np.mean(means)) ** 2)
        r2 = 1.0 - np.sum((means - fitted) ** 2) / max(total, 1e-15)
        rho = float(spearmanr(x0, means).statistic)
        direction = np.sign(slope) if slope != 0 else 1.0
        monotonicity = float(np.mean(direction * np.diff(means) >= 0.0))
        pooled_noise = float(np.sqrt(np.mean(within_std**2)))
        signal_to_noise = float(np.ptp(means) / max(pooled_noise, 1e-12))
        loo_errors = []
        scale = max(float(np.ptp(means)), 1e-12)
        for held_out in range(len(x0)):
            keep = np.arange(len(x0)) != held_out
            coefficients = np.polyfit(x0[keep], means[keep], 1)
            prediction = np.polyval(coefficients, x0[held_out])
            loo_errors.append(abs(prediction - means[held_out]) / scale)
        loo_nrmse = float(np.sqrt(np.mean(np.square(loo_errors))))
        score = (
            0.28 * abs(rho)
            + 0.25 * np.clip(r2, 0.0, 1.0)
            + 0.20 * monotonicity
            + 0.17 * np.tanh(signal_to_noise / 4.0)
            + 0.10 * (1.0 - min(loo_nrmse, 1.0))
        )
        rows.append(
            {
                "feature": name,
                "category": FEATURE_CATALOG[name][0],
                "score": score,
                "spearman_rho": rho,
                "linear_r2": r2,
                "monotonic_fraction": monotonicity,
                "range_to_within_std": signal_to_noise,
                "loo_linear_nrmse": loo_nrmse,
                "slope": slope,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def select_nonredundant_features(
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
        raise RuntimeError("Could not select enough nonredundant static features")
    return selected


def fit_static_mapping(
    feature_frame: pd.DataFrame,
    selected: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = feature_frame.groupby("x0")[selected].mean().reindex(TRAIN_X0)
    observed = grouped.to_numpy(dtype=float)
    coefficients = np.vstack(
        [np.polyfit(TRAIN_X0, observed[:, index], 1) for index in range(len(selected))]
    )
    fitted = static_from_x0(TRAIN_X0, coefficients)
    return observed, fitted, coefficients


def build_training_sequences(
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[float], list[dict]]:
    sequences: list[np.ndarray] = []
    coordinates: list[float] = []
    rows: list[dict] = []
    for x0_index, x0 in enumerate(TRAIN_X0):
        trajectories, source_indices = stable_long_trajectories(
            float(x0),
            args.long_trajectory_s,
            args.train_seeds,
            args.train_seed_base + 1000 * x0_index,
            args.rc_stride,
        )
        for replicate, (signal, source_index) in enumerate(zip(trajectories, source_indices)):
            sequences.append(signal)
            coordinates.append(float(x0))
            rows.append(
                {
                    "x0": float(x0),
                    "replicate": replicate,
                    "source_index": source_index,
                    "samples": len(signal),
                }
            )
    return sequences, coordinates, rows


def build_switch_cohort(
    x0_values: np.ndarray,
    trajectories_per_x0: int,
    base_x0: float,
    warmup_s: float,
    rollout_s: float,
    rc_stride: int,
    seed_base: int,
) -> SwitchCohort:
    sample_hz = 256.0 / rc_stride
    warmups: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    coordinates: list[float] = []
    events: list[float] = []
    for x0_index, x0 in enumerate(x0_values):
        warmup_raw, future_raw = simulate_x0_switch_batch(
            np.full(trajectories_per_x0, x0),
            base_x0,
            warmup_s,
            rollout_s,
            seed_base + 1000 * x0_index,
        )
        warmup = warmup_raw[::rc_stride]
        future = future_raw[::rc_stride]
        onset = event_times(future, sample_hz)
        for trajectory in range(trajectories_per_x0):
            warmups.append(warmup[:, trajectory].copy())
            futures.append(future[:, trajectory].copy())
            coordinates.append(float(x0))
            events.append(onset[trajectory])
        print(
            f"switch cohort x0={x0:.4f}: GT survival={np.mean(~np.isfinite(onset)):.2f}",
            flush=True,
        )
    return SwitchCohort(
        x0=np.asarray(coordinates),
        warmups_raw=np.column_stack(warmups),
        futures_raw=np.column_stack(futures),
        gt_events=np.asarray(events),
    )


def grouped_survival(x0: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(sorted(np.unique(x0)))
    survival = np.asarray(
        [np.mean(~np.isfinite(events[x0 == value])) for value in coordinates]
    )
    return coordinates, survival


def parameters_from_scdt_unit(unit: np.ndarray, seed: int) -> RCParams:
    """Map BO coordinates to bounds suited to long static extrapolation."""
    bounds = np.asarray(
        [
            [0.35, 1.35],
            [0.01, 0.35],
            [0.03, 1.20],
            [0.05, 3.00],
            [-8.0, -1.5],
            [0.015, 0.15],
            [0.00, 1.20],
        ]
    )
    values = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    return RCParams(
        spectral_radius=float(values[0]),
        leakage_rate=float(values[1]),
        input_scaling=float(values[2]),
        static_scaling=float(values[3]),
        ridge=float(10.0 ** values[4]),
        density=float(values[5]),
        innovation_scaling=float(values[6]),
        seed=seed,
    )


def evaluate_rc(
    rc: SparseStaticRC,
    cohort: SwitchCohort,
    coefficients: np.ndarray,
    dynamic_center: float,
    dynamic_scale: float,
    static_center: np.ndarray,
    static_scale: np.ndarray,
    base_x0: float,
    sample_hz: float,
    seed: int,
    ramp_steps: int,
) -> dict:
    warmups = normalize(cohort.warmups_raw, dynamic_center, dynamic_scale)
    target_raw = static_from_x0(cohort.x0, coefficients)
    target_static = normalize(target_raw, static_center, static_scale)
    base_raw = static_from_x0([base_x0], coefficients)[0]
    base_static = normalize(base_raw, static_center, static_scale)
    warmup_static = np.repeat(base_static[None, :], len(cohort.x0), axis=0)
    prediction_norm = rc.rollout_switch_batch(
        warmups,
        warmup_static,
        target_static,
        cohort.futures_raw.shape[0],
        seed,
        ramp_steps,
    )
    predictions = prediction_norm * dynamic_scale + dynamic_center
    rc_events = event_times(predictions, sample_hz)
    x0, gt_survival = grouped_survival(cohort.x0, cohort.gt_events)
    _, rc_survival = grouped_survival(cohort.x0, rc_events)
    survival_mae = float(np.mean(np.abs(gt_survival - rc_survival)))
    transition_weight = np.abs(gt_survival - 0.5) * 2.0
    transition_weight = np.maximum(transition_weight, 0.25)
    false_collapse = float(
        np.average(np.maximum(gt_survival - rc_survival, 0.0), weights=transition_weight)
    )
    missed_collapse = float(
        np.average(np.maximum(rc_survival - gt_survival, 0.0), weights=transition_weight)
    )
    monotonic_penalty = float(np.mean(np.maximum(np.diff(rc_survival), 0.0)))

    stable = np.flatnonzero(~np.isfinite(cohort.gt_events))
    short = min(predictions.shape[0], int(round(10.0 * sample_hz)))
    errors = []
    distribution_errors = []
    for index in stable:
        gt = cohort.futures_raw[:short, index]
        pred = predictions[:short, index]
        errors.append(
            np.sqrt(np.mean((pred - gt) ** 2)) / (np.std(gt) + 1e-9)
        )
        distribution_errors.append(
            abs(np.log((np.std(pred) + 1e-8) / (np.std(gt) + 1e-8)))
        )
    short_nrmse = float(np.median(errors)) if errors else 5.0
    distribution_error = (
        float(np.median(distribution_errors)) if distribution_errors else 5.0
    )
    score = (
        6.0 * false_collapse
        + 12.0 * missed_collapse
        + 1.5 * monotonic_penalty
        + 0.25 * min(short_nrmse, 8.0)
        + 0.35 * min(distribution_error, 8.0)
    )
    return {
        "score": score,
        "survival_mae": survival_mae,
        "false_collapse": false_collapse,
        "missed_collapse": missed_collapse,
        "monotonic_penalty": monotonic_penalty,
        "short_nrmse": short_nrmse,
        "distribution_error": distribution_error,
        "predictions": predictions,
        "rc_events": rc_events,
        "x0": x0,
        "gt_survival": gt_survival,
        "rc_survival": rc_survival,
    }


def optimize_rc(
    args: argparse.Namespace,
    train_sequences: list[np.ndarray],
    train_statics: list[np.ndarray],
    validation: SwitchCohort,
    coefficients: np.ndarray,
    dynamic_center: float,
    dynamic_scale: float,
    static_center: np.ndarray,
    static_scale: np.ndarray,
    selected: list[str],
    sample_hz: float,
) -> tuple[RCParams, list[dict]]:
    rng = np.random.default_rng(args.seed + 31000)
    unit_points: list[np.ndarray] = []
    scores: list[float] = []
    rows: list[dict] = []
    initial = min(8, max(5, args.bo_calls // 3))
    for iteration in range(args.bo_calls):
        if iteration < initial:
            unit = rng.random(7)
        else:
            gp = GaussianProcessRegressor(
                kernel=Matern(length_scale=np.ones(7), nu=2.5) + WhiteKernel(1e-5),
                normalize_y=True,
                random_state=args.seed,
                n_restarts_optimizer=1,
            )
            gp.fit(np.asarray(unit_points), np.asarray(scores))
            candidates = rng.random((500, 7))
            acquisition = expected_improvement(candidates, gp, min(scores))
            unit = candidates[int(np.argmax(acquisition))]
        params = replace(
            parameters_from_scdt_unit(unit, args.seed + 40000 + iteration),
            n_reservoir=args.reservoir_nodes,
        )
        try:
            rc = SparseStaticRC(params, static_dim=len(selected))
            rc.fit(train_sequences, train_statics, args.fit_warmup)
            result = evaluate_rc(
                rc,
                validation,
                coefficients,
                dynamic_center,
                dynamic_scale,
                static_center,
                static_scale,
                args.base_x0,
                sample_hz,
                args.seed + 50000 + iteration,
                args.ramp_steps,
            )
        except Exception as error:
            result = {
                "score": 1e6,
                "survival_mae": 1.0,
                "false_collapse": 1.0,
                "missed_collapse": 1.0,
                "monotonic_penalty": 1.0,
                "short_nrmse": 10.0,
                "distribution_error": 10.0,
            }
            print(f"BO candidate failed: {error}", flush=True)
        unit_points.append(unit)
        scores.append(float(result["score"]))
        row = {
            "iteration": iteration + 1,
            "score": result["score"],
            "survival_mae": result["survival_mae"],
            "false_collapse": result["false_collapse"],
            "missed_collapse": result["missed_collapse"],
            "monotonic_penalty": result["monotonic_penalty"],
            "short_nrmse": result["short_nrmse"],
            "distribution_error": result["distribution_error"],
            **asdict(params),
        }
        rows.append(row)
        print(
            f"BO {iteration + 1:02d}/{args.bo_calls}: "
            f"score={row['score']:.3f}, survival MAE={row['survival_mae']:.3f}, "
            f"false/missed={row['false_collapse']:.2f}/{row['missed_collapse']:.2f}, "
            f"short NRMSE={row['short_nrmse']:.3f}",
            flush=True,
        )
    best = int(np.argmin(scores))
    return replace(
        parameters_from_scdt_unit(unit_points[best], args.seed + 40000 + best),
        n_reservoir=args.reservoir_nodes,
    ), rows


def plot_static_space(
    output: Path,
    frame: pd.DataFrame,
    selected: list[str],
    coefficients: np.ndarray,
    x0_min: float,
    x0_max: float,
) -> None:
    fig, axes = plt.subplots(len(selected), 1, figsize=(9.5, 3.7 * len(selected)), squeeze=False)
    grid = np.linspace(x0_min, x0_max, 300)
    fitted = static_from_x0(grid, coefficients)
    grouped = frame.groupby("x0")
    for feature_index, (axis, name) in enumerate(zip(axes[:, 0], selected)):
        axis.scatter(frame["x0"], frame[name], s=24, alpha=0.35, label="long trajectories")
        means = grouped[name].mean().reindex(TRAIN_X0).to_numpy()
        stds = grouped[name].std().reindex(TRAIN_X0).to_numpy()
        axis.errorbar(TRAIN_X0, means, yerr=stds, marker="o", capsize=3, lw=0, label="train mean +/- SD")
        axis.plot(grid, fitted[:, feature_index], lw=2.0, label="precritical linear coordinate")
        axis.axvline(X0_CRITICAL, color="#d62728", ls="--", lw=1.3, label="critical x0" if feature_index == 0 else None)
        for role, value in ROLE_X0.items():
            axis.axvline(value, color={"inter": "#3182bd", "pre": "#fdae6b", "post": "#de2d26"}[role], ls=":", lw=1.1)
        axis.set_ylabel(name)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("fixed x0")
    fig.suptitle("Long-trajectory statistical coordinate", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=230)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_role_trajectories(
    output: Path,
    role_cohort: SwitchCohort,
    predictions: np.ndarray,
    sample_hz: float,
) -> None:
    roles = list(ROLE_X0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.8), sharex=True, sharey=True)
    time_s = np.arange(predictions.shape[0]) / sample_hz
    for column, role in enumerate(roles):
        index = np.flatnonzero(np.isclose(role_cohort.x0, ROLE_X0[role]))[0]
        gt = role_cohort.futures_raw[:, index]
        rc = predictions[:, index]
        axes[0, column].plot(time_s, gt, color="#111111", lw=0.8)
        axes[1, column].plot(time_s, rc, color="#2166ac", lw=0.8)
        for row in range(2):
            axes[row, column].axhline(-2.0, color="#d62728", ls="--", lw=1.0)
            axes[row, column].grid(alpha=0.2)
        axes[0, column].set_title(f"GT {role}  (x0={ROLE_X0[role]:.4f})", loc="left", fontweight="bold")
        axes[1, column].set_title(f"RC {role}", loc="left", fontweight="bold")
        axes[1, column].set_xlabel("time after static switch (s)")
    axes[0, 0].set_ylabel("EEG proxy")
    axes[1, 0].set_ylabel("EEG proxy")
    fig.suptitle("SCDT-style inter / precritical / postcritical rollout", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=230)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_survival(output: Path, result: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(result["x0"], result["gt_survival"], marker="o", lw=1.8, label="GT")
    ax.plot(result["x0"], result["rc_survival"], marker="s", lw=1.8, label="SCDT-RC")
    ax.axvline(X0_CRITICAL, color="#d62728", ls="--", lw=1.3, label=r"$x_0^C=-2.1$")
    ax.set_xlabel("fixed target x0")
    ax.set_ylabel("survival probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"GT vs RC survival rate (MAE={result['survival_mae']:.3f})")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=230)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--long-trajectory-s", type=float, default=160.0)
    parser.add_argument("--warmup-s", type=float, default=20.0)
    parser.add_argument("--rollout-s", type=float, default=140.0)
    parser.add_argument("--rc-stride", type=int, default=8)
    parser.add_argument("--feature-seeds", type=int, default=5)
    parser.add_argument("--train-seeds", type=int, default=1)
    parser.add_argument("--validation-seeds", type=int, default=4)
    parser.add_argument("--survival-seeds", type=int, default=10)
    parser.add_argument("--static-dim", type=int, default=2)
    parser.add_argument("--correlation-limit", type=float, default=0.90)
    parser.add_argument("--fit-warmup", type=int, default=160)
    parser.add_argument("--ramp-steps", type=int, default=0)
    parser.add_argument("--reservoir-nodes", type=int, default=280)
    parser.add_argument("--bo-calls", type=int, default=24)
    parser.add_argument("--survival-x0", type=int, default=26)
    parser.add_argument("--x0-min", type=float, default=-2.2925)
    parser.add_argument("--x0-max", type=float, default=-1.9275)
    parser.add_argument("--base-x0", type=float, default=-2.18)
    parser.add_argument("--seed", type=int, default=120831)
    parser.add_argument("--feature-seed-base", type=int, default=220831)
    parser.add_argument("--train-seed-base", type=int, default=320831)
    parser.add_argument("--validation-seed-base", type=int, default=420831)
    parser.add_argument("--test-seed-base", type=int, default=520831)
    args = parser.parse_args()

    if np.any(TRAIN_X0 >= X0_CRITICAL):
        raise RuntimeError("Every training x0 must be strictly precritical")
    if args.base_x0 >= X0_CRITICAL:
        raise RuntimeError("Warmup branch must be strictly precritical")
    seed_bases = [
        args.feature_seed_base,
        args.train_seed_base,
        args.validation_seed_base,
        args.test_seed_base,
    ]
    if len(set(seed_bases)) != len(seed_bases):
        raise RuntimeError("Simulation seed bases must be distinct")

    data_dir = args.out_dir / "data" / "paper_single_node_scdt"
    figure_dir = args.out_dir / "figures" / "RC" / "paper_single_node_scdt"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sample_hz = 256.0 / args.rc_stride

    feature_frame = build_long_feature_table(args)
    ranking = feature_ranking(feature_frame)
    selected = select_nonredundant_features(
        ranking,
        feature_frame,
        args.static_dim,
        args.correlation_limit,
    )
    observed_static, fitted_static, coefficients = fit_static_mapping(feature_frame, selected)
    print(f"Selected long-trajectory static features: {selected}", flush=True)

    raw_sequences, train_coordinates, training_rows = build_training_sequences(args)
    dynamic_values = np.concatenate(raw_sequences)
    dynamic_center = float(np.mean(dynamic_values))
    dynamic_scale = float(np.std(dynamic_values))
    train_sequences = [normalize(signal, dynamic_center, dynamic_scale) for signal in raw_sequences]
    train_static_raw = static_from_x0(train_coordinates, coefficients)
    static_center, static_scale = fit_normalizer(static_from_x0(TRAIN_X0, coefficients))
    train_statics = [row for row in normalize(train_static_raw, static_center, static_scale)]

    validation = build_switch_cohort(
        VALIDATION_X0,
        args.validation_seeds,
        args.base_x0,
        args.warmup_s,
        args.rollout_s,
        args.rc_stride,
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
        selected,
        sample_hz,
    )
    rc = SparseStaticRC(params, static_dim=len(selected))
    rc.fit(train_sequences, train_statics, args.fit_warmup)

    test_x0 = np.linspace(args.x0_min, args.x0_max, args.survival_x0)
    test = build_switch_cohort(
        test_x0,
        args.survival_seeds,
        args.base_x0,
        args.warmup_s,
        args.rollout_s,
        args.rc_stride,
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
        args.base_x0,
        sample_hz,
        args.seed + 90000,
        args.ramp_steps,
    )
    role_cohort = build_switch_cohort(
        np.asarray(list(ROLE_X0.values())),
        1,
        args.base_x0,
        args.warmup_s,
        args.rollout_s,
        args.rc_stride,
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
        args.base_x0,
        sample_hz,
        args.seed + 100000,
        args.ramp_steps,
    )

    feature_frame.to_csv(data_dir / "long_trajectory_features.csv", index=False)
    ranking.to_csv(data_dir / "feature_ranking.csv", index=False)
    write_rows(data_dir / "training_trajectories.csv", training_rows)
    write_rows(data_dir / "bo_history.csv", bo_rows)
    survival_rows = [
        {
            "x0": x0,
            "gt_survival": gt,
            "rc_survival": rc_value,
        }
        for x0, gt, rc_value in zip(
            test_result["x0"],
            test_result["gt_survival"],
            test_result["rc_survival"],
        )
    ]
    write_rows(data_dir / "survival_rate.csv", survival_rows)
    selected_protocol = {
        "experiment": "SCDT-style long-trajectory statistical coordinate",
        "clinical_short_window_claim": False,
        "training_x0": TRAIN_X0.tolist(),
        "validation_x0": VALIDATION_X0.tolist(),
        "representative_roles": ROLE_X0,
        "selected_features": selected,
        "feature_coefficients": coefficients.tolist(),
        "observed_training_static": observed_static.tolist(),
        "fitted_training_static": fitted_static.tolist(),
        "dynamic_center": dynamic_center,
        "dynamic_scale": dynamic_scale,
        "static_center": static_center.tolist(),
        "static_scale": static_scale.tolist(),
        "best_rc_parameters": asdict(params),
        "best_validation_metrics": min(bo_rows, key=lambda row: row["score"]),
        "test_survival_mae": test_result["survival_mae"],
        "arguments": vars(args) | {"out_dir": str(args.out_dir)},
    }
    (data_dir / "protocol.json").write_text(
        json.dumps(selected_protocol, indent=2), encoding="utf-8"
    )

    plot_static_space(
        figure_dir / "static_space",
        feature_frame,
        selected,
        coefficients,
        args.x0_min,
        args.x0_max,
    )
    plot_role_trajectories(
        figure_dir / "gt_rc_inter_pre_post",
        role_cohort,
        role_result["predictions"],
        sample_hz,
    )
    plot_survival(figure_dir / "survival_rate", test_result)

    print("\nSCDT-style main result")
    print(f"  selected static features: {selected}")
    print(f"  test survival MAE: {test_result['survival_mae']:.4f}")
    print(f"  best RC parameters: {asdict(params)}")
    print(f"  outputs: {figure_dir}")


if __name__ == "__main__":
    main()
