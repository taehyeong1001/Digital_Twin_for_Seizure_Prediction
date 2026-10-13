#!/usr/bin/env python3
"""SCRC rollout and survival for the paper-based single-node Epileptor."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from paper_epileptor_core import (
    PaperEpileptorConfig,
    X0_CRITICAL,
    detect_local_minimum_onset,
    simulate_fixed_x0,
    source_proxy,
)


@dataclass(frozen=True)
class RCParams:
    n_reservoir: int = 220
    spectral_radius: float = 0.7677407747126472
    leakage_rate: float = 0.2812239177110785
    input_scaling: float = 0.5711791129660125
    static_scaling: float = 0.14013485850236362
    ridge: float = 0.0006011541771080097
    density: float = 0.06515474997949441
    innovation_scaling: float = 0.9755515182485607
    seed: int = 49104


@dataclass(frozen=True)
class FeatureManifold:
    center: np.ndarray
    scale: np.ndarray
    inverse_covariance: np.ndarray
    threshold: float
    window_samples: int
    step_samples: int
    sustain_windows: int


class SparseStaticRC:
    def __init__(self, params: RCParams, static_dim: int = 2):
        self.params = params
        self.static_dim = int(static_dim)
        if self.static_dim < 1:
            raise ValueError("static_dim must be positive")
        rng = np.random.default_rng(params.seed)
        self.w_in = (
            rng.uniform(-1.0, 1.0, (params.n_reservoir, 1))
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
                        )
                    )
                )
            )
        except Exception:
            radius = float(
                np.max(np.abs(np.linalg.eigvals(matrix.toarray())))
            )
        self.a = matrix * (params.spectral_radius / max(radius, 1e-9))
        self.w_out: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.clip_low = -5.0
        self.clip_high = 5.0

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
    def augment(
        reservoir: np.ndarray,
        dynamic: np.ndarray,
    ) -> np.ndarray:
        transformed = reservoir.copy()
        transformed[1::2] = transformed[1::2] ** 2
        if transformed.ndim == 1:
            return np.r_[transformed, dynamic, 1.0]
        return np.vstack(
            [
                transformed,
                dynamic,
                np.ones((1, transformed.shape[1])),
            ]
        )

    def fit(
        self,
        sequences: list[np.ndarray],
        statics: list[np.ndarray],
        warmup: int,
    ) -> None:
        all_values = np.concatenate(sequences)
        span = float(np.ptp(all_values))
        self.clip_low = float(np.min(all_values) - 2.5 * span)
        self.clip_high = float(np.max(all_values) + 1.2 * span)
        n_columns = sum(max(0, len(sequence) - 1 - warmup) for sequence in sequences)
        augmented = np.empty(
            (self.params.n_reservoir + 2, n_columns),
            dtype=float,
        )
        targets = np.empty((1, n_columns), dtype=float)
        column = 0
        for sequence, static in zip(sequences, statics):
            reservoir = np.zeros(self.params.n_reservoir)
            for index in range(len(sequence) - 1):
                reservoir = self.update(
                    reservoir,
                    np.asarray([sequence[index]]),
                    static,
                )
                if index >= warmup:
                    augmented[:, column] = self.augment(
                        reservoir,
                        np.asarray([sequence[index]]),
                    )
                    targets[0, column] = sequence[index + 1]
                    column += 1
        gram = augmented @ augmented.T
        self.w_out = targets @ augmented.T @ np.linalg.solve(
            gram + self.params.ridge * np.eye(gram.shape[0]),
            np.eye(gram.shape[0]),
        )
        residuals = (targets - self.w_out @ augmented).ravel()
        lower, upper = np.quantile(residuals, [0.0025, 0.9975])
        self.residuals = np.clip(residuals, lower, upper)

    def rollout(
        self,
        warmup: np.ndarray,
        static: np.ndarray,
        n_steps: int,
        seed: int = 0,
    ) -> np.ndarray:
        prediction = self.rollout_batch(
            warmup[:, None],
            static[None, :],
            n_steps,
            seed,
        )
        return prediction[:, 0]

    def rollout_batch(
        self,
        warmup: np.ndarray,
        statics: np.ndarray,
        n_steps: int,
        seed: int = 0,
    ) -> np.ndarray:
        return self.rollout_switch_batch(
            warmup,
            statics,
            statics,
            n_steps,
            seed,
        )

    def rollout_switch_batch(
        self,
        warmup: np.ndarray,
        warmup_statics: np.ndarray,
        target_statics: np.ndarray,
        n_steps: int,
        seed: int = 0,
        ramp_steps: int = 0,
    ) -> np.ndarray:
        """Warm on one regime and autonomously roll out under another."""
        if self.w_out is None or self.residuals is None:
            raise RuntimeError("RC must be fitted before rollout.")
        if warmup.ndim != 2:
            raise ValueError("warmup must have shape (time, batch).")
        batch = warmup.shape[1]
        reservoir = np.zeros((self.params.n_reservoir, batch))
        warmup_static_matrix = warmup_statics.T
        target_static_matrix = target_statics.T
        for index in range(len(warmup)):
            reservoir = self.update(
                reservoir,
                warmup[index][None, :],
                warmup_static_matrix,
            )
        current = warmup[-1][None, :]
        prediction = np.empty((n_steps, batch), dtype=float)
        rng = np.random.default_rng(seed)
        for index in range(n_steps):
            if ramp_steps > 0 and index < ramp_steps:
                fraction = (index + 1) / ramp_steps
                static_matrix = (
                    (1.0 - fraction) * warmup_static_matrix
                    + fraction * target_static_matrix
                )
            else:
                static_matrix = target_static_matrix
            reservoir = self.update(
                reservoir,
                current,
                static_matrix,
            )
            current = self.w_out @ self.augment(reservoir, current)
            innovations = rng.choice(self.residuals, size=batch)
            current += self.params.innovation_scaling * innovations[None, :]
            current = np.clip(current, self.clip_low, self.clip_high)
            prediction[index] = current[0]
        return prediction


def parse_x0_list(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def simulate_batch(
    x0_values: np.ndarray,
    duration_s: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_trajectories = len(x0_values)
    config = PaperEpileptorConfig(
        n_nodes=n_trajectories,
        duration_s=duration_s,
        seed=seed,
    )
    time_s, states = simulate_fixed_x0(
        config,
        np.asarray(x0_values, dtype=float),
        np.zeros((n_trajectories, n_trajectories)),
    )
    return time_s, source_proxy(states)


def onset_for_signal(
    time_s: np.ndarray,
    signal: np.ndarray,
) -> float | None:
    return detect_local_minimum_onset(time_s, signal)


def static_before_onset(
    time_s: np.ndarray,
    signal: np.ndarray,
    onset: float | None,
    minimum_end_s: float,
) -> np.ndarray:
    if onset is None:
        end = len(signal)
    else:
        end = int(np.searchsorted(time_s, onset))
    minimum_end = int(np.searchsorted(time_s, minimum_end_s))
    end = max(16, minimum_end, end)
    segment = signal[: min(end, len(signal))]
    return np.asarray([np.std(segment), np.var(segment)], dtype=float)


def fit_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return center, scale


def normalize(
    values: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (values - center) / scale


def fit_static_trend(
    x0_values: np.ndarray,
    static_values: np.ndarray,
) -> np.ndarray:
    """Fit the precritical x0-to-static calibration used for extrapolation."""
    return np.vstack(
        [
            np.polyfit(x0_values, static_values[:, feature], 1)
            for feature in range(static_values.shape[1])
        ]
    )


def static_from_x0(
    x0_values: np.ndarray | list[float],
    coefficients: np.ndarray,
) -> np.ndarray:
    x0 = np.atleast_1d(np.asarray(x0_values, dtype=float))
    return np.column_stack(
        [
            np.polyval(coefficients[feature], x0)
            for feature in range(coefficients.shape[0])
        ]
    )


def detect_prediction_onset(
    prediction: np.ndarray,
    sample_hz: float,
) -> float | None:
    relative_time = np.arange(len(prediction), dtype=float) / sample_hz
    return detect_local_minimum_onset(relative_time, prediction)


def rolling_std_var(
    signal: np.ndarray,
    sample_hz: float,
    window_s: float = 10.0,
    step_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    window = max(8, int(round(window_s * sample_hz)))
    step = max(1, int(round(step_s * sample_hz)))
    if len(signal) < window:
        return np.empty(0), np.empty((0, 2))
    starts = np.arange(0, len(signal) - window + 1, step)
    features = np.empty((len(starts), 2), dtype=float)
    for index, start in enumerate(starts):
        segment = signal[start : start + window]
        features[index] = [np.std(segment), np.var(segment)]
    end_times = (starts + window) / sample_hz
    return end_times, features


def fit_feature_manifold(
    predictions: np.ndarray,
    sample_hz: float,
    window_s: float = 10.0,
    step_s: float = 1.0,
    sustain_windows: int = 2,
) -> FeatureManifold:
    all_features = []
    for index in range(predictions.shape[1]):
        _, features = rolling_std_var(
            predictions[:, index],
            sample_hz,
            window_s,
            step_s,
        )
        all_features.append(features)
    features = np.vstack(all_features)
    center = np.median(features, axis=0)
    scale = 1.4826 * np.median(
        np.abs(features - center),
        axis=0,
    )
    fallback = np.std(features, axis=0)
    scale = np.where(scale > 1e-8, scale, np.maximum(fallback, 1e-6))
    standardized = (features - center) / scale
    covariance = np.cov(standardized, rowvar=False)
    covariance += 0.10 * np.eye(covariance.shape[0])
    inverse_covariance = np.linalg.inv(covariance)
    distances = np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            standardized,
            inverse_covariance,
            standardized,
        )
    )
    threshold = max(3.0, float(np.quantile(distances, 0.995)))
    return FeatureManifold(
        center=center,
        scale=scale,
        inverse_covariance=inverse_covariance,
        threshold=threshold,
        window_samples=max(8, int(round(window_s * sample_hz))),
        step_samples=max(1, int(round(step_s * sample_hz))),
        sustain_windows=sustain_windows,
    )


def feature_risk(
    prediction: np.ndarray,
    manifold: FeatureManifold,
    sample_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    end_times, features = rolling_std_var(
        prediction,
        sample_hz,
        manifold.window_samples / sample_hz,
        manifold.step_samples / sample_hz,
    )
    standardized = (features - manifold.center) / manifold.scale
    distances = np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            standardized,
            manifold.inverse_covariance,
            standardized,
        )
    )
    return end_times, distances


def detect_feature_escape(
    prediction: np.ndarray,
    manifold: FeatureManifold,
    sample_hz: float,
) -> float | None:
    end_times, distances = feature_risk(
        prediction,
        manifold,
        sample_hz,
    )
    run = 0
    for index, value in enumerate(distances > manifold.threshold):
        run = run + 1 if value else 0
        if run >= manifold.sustain_windows:
            first = index - manifold.sustain_windows + 1
            return float(end_times[first])
    return None


def fit_and_evaluate(
    params: RCParams,
    train_sequences: list[np.ndarray],
    train_statics: list[np.ndarray],
    normal_warmup: np.ndarray,
    normal_static: np.ndarray,
    normal_gt: np.ndarray,
    target_warmup: np.ndarray,
    target_static: np.ndarray,
    target_gt: np.ndarray,
    fit_warmup: int,
    seed: int = 0,
) -> tuple[SparseStaticRC, np.ndarray, np.ndarray]:
    rc = SparseStaticRC(params)
    rc.fit(train_sequences, train_statics, fit_warmup)
    normal_prediction = rc.rollout(
        normal_warmup,
        normal_static,
        len(normal_gt),
        seed + 1,
    )
    target_prediction = rc.rollout(
        target_warmup,
        target_static,
        len(target_gt),
        seed + 2,
    )
    return rc, normal_prediction, target_prediction


def nrmse(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((prediction - ground_truth) ** 2))
        / (np.std(ground_truth) + 1e-9)
    )


def rollout_score_batch(
    normal_predictions: np.ndarray,
    normal_ground_truth: np.ndarray,
    target_predictions: np.ndarray,
    target_ground_truth: np.ndarray,
    sample_hz: float,
) -> tuple[float, float, float]:
    short_samples = min(
        normal_ground_truth.shape[0],
        int(round(5.0 * sample_hz)),
    )
    normal_errors = []
    amplitude_penalties = []
    quantile_penalties = []
    for index in range(normal_predictions.shape[1]):
        prediction = normal_predictions[:, index]
        ground_truth = normal_ground_truth[:, index]
        normal_errors.append(
            nrmse(
                prediction[:short_samples],
                ground_truth[:short_samples],
            )
        )
        amplitude_penalties.append(
            abs(
                np.log(
                    max(
                        np.std(prediction)
                        / (np.std(ground_truth) + 1e-9),
                        1e-6,
                    )
                )
            )
        )
        gt_quantiles = np.quantile(ground_truth, [0.1, 0.5, 0.9])
        rc_quantiles = np.quantile(prediction, [0.1, 0.5, 0.9])
        quantile_penalties.append(
            float(
                np.mean(np.abs(rc_quantiles - gt_quantiles))
                / (np.std(ground_truth) + 1e-9)
            )
        )

    calibration_indices = np.arange(0, normal_predictions.shape[1], 2)
    validation_indices = np.arange(1, normal_predictions.shape[1], 2)
    if len(validation_indices) == 0:
        validation_indices = calibration_indices
    feature_manifold = fit_feature_manifold(
        normal_predictions[:, calibration_indices],
        sample_hz,
    )
    normal_events = [
        detect_feature_escape(
            normal_predictions[:, index],
            feature_manifold,
            sample_hz,
        )
        for index in validation_indices
    ]
    target_events = []
    timing_penalties = []
    for index in range(target_predictions.shape[1]):
        gt_event = detect_prediction_onset(
            target_ground_truth[:, index],
            sample_hz,
        )
        rc_event = detect_feature_escape(
            target_predictions[:, index],
            feature_manifold,
            sample_hz,
        )
        target_events.append(rc_event)
        if rc_event is not None and gt_event is not None:
            timing_penalties.append(
                min(
                    max(0.0, rc_event - gt_event)
                    / max(gt_event, 1.0),
                    2.0,
                )
            )

    normal_escape_rate = float(
        np.mean([event is not None for event in normal_events])
    )
    target_escape_rate = float(
        np.mean([event is not None for event in target_events])
    )
    event_penalty = 40.0 * (1.0 - target_escape_rate)
    timing_penalty = (
        float(np.mean(timing_penalties))
        if timing_penalties
        else 0.0
    )
    target_range_penalty = max(
        0.0,
        float(np.max(np.abs(target_predictions))) - 5.0,
    )
    score = float(
        0.15 * np.mean(normal_errors)
        + 60.0 * normal_escape_rate
        + event_penalty
        + timing_penalty
        + 2.2 * np.mean(amplitude_penalties)
        + 1.5 * np.mean(quantile_penalties)
        + 0.2 * target_range_penalty
    )
    return score, normal_escape_rate, target_escape_rate


def expected_improvement(
    candidates: np.ndarray,
    gp: GaussianProcessRegressor,
    best: float,
) -> np.ndarray:
    mean, std = gp.predict(candidates, return_std=True)
    std = np.maximum(std, 1e-9)
    improvement = best - mean
    z_value = improvement / std
    return improvement * norm.cdf(z_value) + std * norm.pdf(z_value)


def parameters_from_unit(unit: np.ndarray, seed: int) -> RCParams:
    bounds = np.asarray(
        [
            [0.45, 1.15],
            [0.025, 0.30],
            [0.05, 0.80],
            [0.02, 1.20],
            [-7.0, -2.0],
            [0.02, 0.10],
            [0.10, 1.60],
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


def bayesian_optimize(
    args: argparse.Namespace,
    train_sequences: list[np.ndarray],
    train_statics: list[np.ndarray],
    normal_warmups: np.ndarray,
    normal_statics: np.ndarray,
    normal_ground_truth: np.ndarray,
    target_warmups: np.ndarray,
    target_statics: np.ndarray,
    target_ground_truth: np.ndarray,
    sample_hz: float,
) -> tuple[RCParams, list[dict[str, float]]]:
    rng = np.random.default_rng(args.seed + 31000)
    unit_points: list[np.ndarray] = []
    scores: list[float] = []
    rows: list[dict[str, float]] = []
    n_initial = min(10, max(6, args.bo_calls // 3))
    for iteration in range(args.bo_calls):
        if iteration < n_initial:
            unit = rng.random(7)
        else:
            gp = GaussianProcessRegressor(
                kernel=(
                    Matern(length_scale=np.ones(7), nu=2.5)
                    + WhiteKernel(noise_level=1e-5)
                ),
                normalize_y=True,
                random_state=args.seed,
                n_restarts_optimizer=1,
            )
            gp.fit(np.asarray(unit_points), np.asarray(scores))
            candidates = rng.random((600, 7))
            acquisition = expected_improvement(
                candidates,
                gp,
                min(scores),
            )
            unit = candidates[int(np.argmax(acquisition))]
        params = parameters_from_unit(
            unit,
            args.seed + 40000 + iteration,
        )
        try:
            candidate_rc = SparseStaticRC(params)
            candidate_rc.fit(
                train_sequences,
                train_statics,
                args.fit_warmup,
            )
            normal_predictions = candidate_rc.rollout_batch(
                normal_warmups,
                normal_statics,
                normal_ground_truth.shape[0],
                args.seed + 60000 + iteration,
            )
            target_predictions = candidate_rc.rollout_batch(
                target_warmups,
                target_statics,
                target_ground_truth.shape[0],
                args.seed + 70000 + iteration,
            )
            (
                score,
                normal_escape_rate,
                target_escape_rate,
            ) = rollout_score_batch(
                normal_predictions,
                normal_ground_truth,
                target_predictions,
                target_ground_truth,
                sample_hz,
            )
        except Exception:
            score = 1.0e6
            normal_escape_rate = 1.0
            target_escape_rate = 0.0
        unit_points.append(unit)
        scores.append(score)
        row = {
            "iteration": float(iteration + 1),
            "score": score,
            "normal_escape_rate": normal_escape_rate,
            "target_escape_rate": target_escape_rate,
            "spectral_radius": params.spectral_radius,
            "leakage_rate": params.leakage_rate,
            "input_scaling": params.input_scaling,
            "static_scaling": params.static_scaling,
            "ridge": params.ridge,
            "density": params.density,
            "innovation_scaling": params.innovation_scaling,
        }
        rows.append(row)
        print(
            f"BO {iteration + 1:02d}/{args.bo_calls}: "
            f"score={score:.4f}, sr={params.spectral_radius:.3f}, "
            f"leak={params.leakage_rate:.3f}, "
            f"static={params.static_scaling:.3f}, "
            f"innovation={params.innovation_scaling:.3f}, "
            f"normal_escape={normal_escape_rate:.2f}, "
            f"target_escape={target_escape_rate:.2f}"
        )
    best = int(np.argmin(scores))
    return parameters_from_unit(
        unit_points[best],
        args.seed + 40000 + best,
    ), rows


def plot_rollout_context(
    output: Path,
    train_x0: list[float],
    train_static_observed: np.ndarray,
    train_static_fitted: np.ndarray,
    target_time: np.ndarray,
    target_signal: np.ndarray,
    target_onset: float | None,
    rollout_time: np.ndarray,
    target_gt: np.ndarray,
    target_prediction: np.ndarray,
    warmup_s: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.0))
    axes[0].plot(
        train_x0,
        train_static_observed[:, 0],
        marker="o",
        ls="none",
        alpha=0.75,
        label="observed mean std",
    )
    axes[0].plot(
        train_x0,
        train_static_fitted[:, 0],
        lw=1.6,
        label="fitted std coordinate",
    )
    axes[0].plot(
        train_x0,
        train_static_observed[:, 1],
        marker="s",
        ls="none",
        alpha=0.75,
        label="observed mean var",
    )
    axes[0].plot(
        train_x0,
        train_static_fitted[:, 1],
        lw=1.6,
        label="fitted var coordinate",
    )
    axes[0].set_title(
        "Precritical x0-to-static calibration (five trajectories per x0)"
    )
    axes[0].set_xlabel("fixed x0")
    axes[0].set_ylabel("raw static value")
    axes[0].legend()

    axes[1].plot(target_time, target_signal, color="#202020", lw=0.75)
    axes[1].axvspan(
        0.0,
        warmup_s,
        color="#78a6c8",
        alpha=0.24,
        label="observed warmup",
    )
    if target_onset is not None:
        axes[1].axvline(
            target_onset,
            color="#111111",
            lw=1.7,
            ls="--",
            label="GT collapse",
        )
    axes[1].set_title("Paper-based target GT context")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel(r"$x_2-x_1$")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        rollout_time,
        target_gt,
        color="#111111",
        lw=0.8,
        label="GT",
    )
    axes[2].plot(
        rollout_time,
        target_prediction,
        color="#d34a32",
        lw=0.9,
        label="SCRC",
    )
    axes[2].axhline(
        -2.0,
        color="#d62728",
        lw=1.0,
        ls=":",
        label="GT collapse threshold",
    )
    axes[2].set_title(
        "Autonomous rollout with extrapolated precritical std/var context"
    )
    axes[2].set_xlabel("time after warmup (s)")
    axes[2].set_ylabel(r"$x_2-x_1$")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=230)
    plt.close(fig)


def plot_2x2(
    output: Path,
    time_s: np.ndarray,
    normal_gt: np.ndarray,
    normal_prediction: np.ndarray,
    target_gt: np.ndarray,
    target_prediction: np.ndarray,
) -> None:
    panels = [
        ("(a1) GT normal", normal_gt),
        ("(a2) GT ictal target", target_gt),
        ("(b1) RC normal", normal_prediction),
        ("(b2) RC ictal context", target_prediction),
    ]
    y_min = min(float(np.min(signal)) for _, signal in panels)
    y_max = max(float(np.max(signal)) for _, signal in panels)
    padding = 0.08 * max(y_max - y_min, 1e-9)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharex=True, sharey=True)
    for panel_index, (ax, (title, signal)) in enumerate(
        zip(axes.flat, panels)
    ):
        ax.plot(time_s, signal, color="#0645f5", lw=0.85)
        if panel_index < 2:
            ax.axhline(-2.0, color="#d62728", lw=1.2, ls="--")
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.grid(alpha=0.20)
    axes[0, 0].set_ylabel(r"$x_2-x_1$")
    axes[1, 0].set_ylabel(r"$x_2-x_1$")
    axes[1, 0].set_xlabel("time after warmup (s)")
    axes[1, 1].set_xlabel("time after warmup (s)")
    fig.suptitle("Paper-based single-node GT vs SCRC rollout", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=230)
    plt.close(fig)


def plot_feature_risk(
    output: Path,
    normal_prediction: np.ndarray,
    target_prediction: np.ndarray,
    manifold: FeatureManifold,
    sample_hz: float,
) -> None:
    normal_time, normal_risk = feature_risk(
        normal_prediction,
        manifold,
        sample_hz,
    )
    target_time, target_risk = feature_risk(
        target_prediction,
        manifold,
        sample_hz,
    )
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.plot(normal_time, normal_risk, lw=1.5, label="RC normal risk")
    ax.plot(target_time, target_risk, lw=1.5, label="RC ictal-context risk")
    ax.axhline(
        manifold.threshold,
        color="#d62728",
        lw=1.3,
        ls="--",
        label="normal-manifold threshold",
    )
    ax.set_xlabel("time after warmup (s)")
    ax.set_ylabel("rolling std/var Mahalanobis risk")
    ax.set_title("SCRC statistical-manifold escape")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=230)
    plt.close(fig)


def delay_embedding(
    signal: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    return signal[:-lag], signal[lag:]


def plot_attractors(
    output: Path,
    normal_gt: np.ndarray,
    normal_prediction: np.ndarray,
    target_gt: np.ndarray,
    target_prediction: np.ndarray,
    lag: int,
) -> None:
    panels = [
        ("(a1) GT normal", normal_gt),
        ("(a2) GT ictal target", target_gt),
        ("(b1) RC normal", normal_prediction),
        ("(b2) RC ictal context", target_prediction),
    ]
    embedded = [
        (title, *delay_embedding(signal, lag))
        for title, signal in panels
    ]
    limits = np.concatenate(
        [np.r_[x_value, y_value] for _, x_value, y_value in embedded]
    )
    lower, upper = float(np.min(limits)), float(np.max(limits))
    padding = 0.08 * max(upper - lower, 1e-9)
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 8.2), sharex=True, sharey=True)
    for ax, (title, x_value, y_value) in zip(axes.flat, embedded):
        ax.plot(x_value, y_value, color="#0645f5", lw=0.65, alpha=0.82)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xlim(lower - padding, upper + padding)
        ax.set_ylim(lower - padding, upper + padding)
        ax.grid(alpha=0.20)
    axes[0, 0].set_ylabel(r"$y(t+\tau)$")
    axes[1, 0].set_ylabel(r"$y(t+\tau)$")
    axes[1, 0].set_xlabel(r"$y(t)$")
    axes[1, 1].set_xlabel(r"$y(t)$")
    fig.suptitle(f"GT vs SCRC delay embeddings, lag={lag} samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=230)
    plt.close(fig)


def survival_scan(
    args: argparse.Namespace,
    rc: SparseStaticRC,
    dynamic_center: float,
    dynamic_scale: float,
    static_center: np.ndarray,
    static_scale: np.ndarray,
    static_coefficients: np.ndarray,
    sample_hz: float,
    feature_manifold: FeatureManifold,
) -> list[dict[str, float]]:
    x0_values = np.linspace(args.x0_min, args.x0_max, args.survival_x0)
    rows: list[dict[str, float]] = []
    warmup_samples = int(round(args.warmup_s * sample_hz))
    rollout_samples = int(round(args.rollout_s * sample_hz))
    for x0_index, x0 in enumerate(x0_values):
        time_s, proxies = simulate_batch(
            np.full(args.survival_seeds, x0),
            args.duration_s,
            args.seed + 50000 + 1000 * x0_index,
        )
        sampled = proxies[:: args.rc_stride]
        onsets = [
            onset_for_signal(time_s, proxies[:, index])
            for index in range(args.survival_seeds)
        ]
        static_raw = np.repeat(
            static_from_x0([x0], static_coefficients),
            args.survival_seeds,
            axis=0,
        )
        statics = normalize(static_raw, static_center, static_scale)
        warmup = normalize(
            sampled[:warmup_samples],
            np.asarray(dynamic_center),
            np.asarray(dynamic_scale),
        )
        prediction_norm = rc.rollout_batch(
            warmup,
            statics,
            rollout_samples,
            args.seed + 90000 + 1000 * x0_index,
        )
        predictions = (
            prediction_norm * dynamic_scale + dynamic_center
        )
        horizon_end = args.warmup_s + args.rollout_s
        gt_survived = [
            onset is None or onset > horizon_end
            for onset in onsets
        ]
        rc_onsets = [
            detect_feature_escape(
                prediction_norm[:, index],
                feature_manifold,
                sample_hz,
            )
            for index in range(args.survival_seeds)
        ]
        rc_survived = [onset is None for onset in rc_onsets]
        row = {
            "x0": float(x0),
            "gt_survival": float(np.mean(gt_survived)),
            "rc_survival": float(np.mean(rc_survived)),
            "gt_event_count": float(
                args.survival_seeds - np.sum(gt_survived)
            ),
            "rc_event_count": float(
                args.survival_seeds - np.sum(rc_survived)
            ),
        }
        rows.append(row)
        print(
            f"survival x0={x0:.4f}: "
            f"GT={row['gt_survival']:.2f}, "
            f"RC={row['rc_survival']:.2f}"
        )
    return rows


def plot_survival(
    output: Path,
    rows: list[dict[str, float]],
) -> None:
    x0 = np.asarray([row["x0"] for row in rows])
    gt = np.asarray([row["gt_survival"] for row in rows])
    rc = np.asarray([row["rc_survival"] for row in rows])
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(x0, gt, marker="o", lw=1.8, label="GT survival")
    ax.plot(x0, rc, marker="s", lw=1.8, label="SCRC survival")
    ax.axvline(
        X0_CRITICAL,
        color="#d62728",
        lw=1.2,
        ls="--",
        label=r"$x_0^C=-2.1$",
    )
    ax.set_xlabel("fixed x0")
    ax.set_ylabel("survival probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "GT collapse survival vs SCRC statistical-manifold survival"
    )
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=230)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument(
        "--train-x0",
        type=str,
        default="-2.300,-2.260,-2.220,-2.180,-2.140,-2.100,-2.065",
    )
    parser.add_argument("--normal-x0", type=float, default=-2.10)
    parser.add_argument("--target-x0", type=float, default=-2.024)
    parser.add_argument("--train-segment-s", type=float, default=90.0)
    parser.add_argument("--warmup-s", type=float, default=40.0)
    parser.add_argument("--rollout-s", type=float, default=120.0)
    parser.add_argument("--rc-stride", type=int, default=4)
    parser.add_argument("--fit-warmup", type=int, default=160)
    parser.add_argument("--attractor-lag", type=int, default=8)
    parser.add_argument("--x0-min", type=float, default=-2.30)
    parser.add_argument("--x0-max", type=float, default=-1.90)
    parser.add_argument("--survival-x0", type=int, default=30)
    parser.add_argument("--survival-seeds", type=int, default=10)
    parser.add_argument("--static-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--bo-calls", type=int, default=36)
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args()

    if args.warmup_s + args.rollout_s > args.duration_s:
        raise ValueError("warmup-s + rollout-s must not exceed duration-s.")

    figure_dir = args.out_dir / "figures" / "RC" / "paper_single_node"
    data_dir = args.out_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_x0 = parse_x0_list(args.train_x0)
    train_time, train_proxy = simulate_batch(
        np.asarray(train_x0),
        args.duration_s,
        args.seed,
    )
    normal_time, normal_proxy_batch = simulate_batch(
        np.asarray([args.normal_x0]),
        args.duration_s,
        args.seed + 1000,
    )
    target_time, target_proxy_batch = simulate_batch(
        np.asarray([args.target_x0]),
        args.duration_s,
        args.seed + 2000,
    )
    normal_proxy = normal_proxy_batch[:, 0]
    target_proxy = target_proxy_batch[:, 0]
    normal_onset = onset_for_signal(normal_time, normal_proxy)
    target_onset = onset_for_signal(target_time, target_proxy)

    sample_hz = 256.0 / args.rc_stride
    train_samples = int(round(args.train_segment_s * sample_hz))
    warmup_samples = int(round(args.warmup_s * sample_hz))
    rollout_samples = int(round(args.rollout_s * sample_hz))
    train_sampled = train_proxy[:: args.rc_stride][:train_samples]
    normal_sampled = normal_proxy[:: args.rc_stride]
    target_sampled = target_proxy[:: args.rc_stride]

    train_sequences_raw = [
        train_sampled[:, index]
        for index in range(len(train_x0))
    ]
    train_static_observed_rows = []
    for index, x0 in enumerate(train_x0):
        calibration_time, calibration_proxy = simulate_batch(
            np.full(args.static_seeds, x0),
            args.duration_s,
            args.seed + 10000 + 1000 * index,
        )
        calibration_onsets = [
            onset_for_signal(
                calibration_time,
                calibration_proxy[:, seed_index],
            )
            for seed_index in range(args.static_seeds)
        ]
        calibration_static = np.vstack(
            [
                static_before_onset(
                    calibration_time,
                    calibration_proxy[:, seed_index],
                    calibration_onsets[seed_index],
                    args.warmup_s,
                )
                for seed_index in range(args.static_seeds)
            ]
        )
        train_static_observed_rows.append(
            np.mean(calibration_static, axis=0)
        )
    train_static_observed = np.vstack(train_static_observed_rows)
    static_coefficients = fit_static_trend(
        np.asarray(train_x0),
        train_static_observed,
    )
    train_static_raw = static_from_x0(train_x0, static_coefficients)
    normal_static_observed = static_before_onset(
        normal_time,
        normal_proxy,
        normal_onset,
        args.warmup_s,
    )
    target_static_observed = static_before_onset(
        target_time,
        target_proxy,
        target_onset,
        args.warmup_s,
    )
    normal_static_raw = static_from_x0(
        [args.normal_x0],
        static_coefficients,
    )[0]
    target_static_raw = static_from_x0(
        [args.target_x0],
        static_coefficients,
    )[0]

    train_concat = np.concatenate(train_sequences_raw)
    dynamic_center = float(np.mean(train_concat))
    dynamic_scale = float(np.std(train_concat))
    train_sequences = [
        (sequence - dynamic_center) / dynamic_scale
        for sequence in train_sequences_raw
    ]
    static_center, static_scale = fit_normalizer(train_static_raw)
    train_statics = [
        normalize(value, static_center, static_scale)
        for value in train_static_raw
    ]
    normal_static = normalize(
        normal_static_raw,
        static_center,
        static_scale,
    )
    target_static = normalize(
        target_static_raw,
        static_center,
        static_scale,
    )

    normal_warmup = (
        normal_sampled[:warmup_samples] - dynamic_center
    ) / dynamic_scale
    target_warmup = (
        target_sampled[:warmup_samples] - dynamic_center
    ) / dynamic_scale
    normal_gt = normal_sampled[
        warmup_samples : warmup_samples + rollout_samples
    ]
    target_gt = target_sampled[
        warmup_samples : warmup_samples + rollout_samples
    ]

    bo_normal_x0_all = np.linspace(-2.300, -2.065, 18)
    bo_target_x0_all = np.linspace(-2.050, -1.950, 8)
    bo_normal_time, bo_normal_proxy_all = simulate_batch(
        bo_normal_x0_all,
        args.duration_s,
        args.seed + 3000,
    )
    bo_target_time, bo_target_proxy_all = simulate_batch(
        bo_target_x0_all,
        args.duration_s,
        args.seed + 4000,
    )
    horizon_end = args.warmup_s + args.rollout_s
    bo_normal_onsets = [
        onset_for_signal(bo_normal_time, bo_normal_proxy_all[:, index])
        for index in range(len(bo_normal_x0_all))
    ]
    bo_target_onsets = [
        onset_for_signal(bo_target_time, bo_target_proxy_all[:, index])
        for index in range(len(bo_target_x0_all))
    ]
    bo_normal_keep = np.asarray(
        [
            onset is None or onset > horizon_end
            for onset in bo_normal_onsets
        ]
    )
    bo_target_keep = np.asarray(
        [
            onset is not None
            and args.warmup_s < onset <= horizon_end
            for onset in bo_target_onsets
        ]
    )
    if not np.any(bo_normal_keep):
        bo_normal_keep = np.ones_like(bo_normal_keep, dtype=bool)
    if not np.any(bo_target_keep):
        bo_target_keep = np.ones_like(bo_target_keep, dtype=bool)
    bo_normal_x0 = bo_normal_x0_all[bo_normal_keep]
    bo_target_x0 = bo_target_x0_all[bo_target_keep]
    bo_normal_sampled = bo_normal_proxy_all[:: args.rc_stride][
        :, bo_normal_keep
    ]
    bo_target_sampled = bo_target_proxy_all[:: args.rc_stride][
        :, bo_target_keep
    ]
    bo_normal_warmups = (
        bo_normal_sampled[:warmup_samples] - dynamic_center
    ) / dynamic_scale
    bo_target_warmups = (
        bo_target_sampled[:warmup_samples] - dynamic_center
    ) / dynamic_scale
    bo_normal_ground_truth = bo_normal_sampled[
        warmup_samples : warmup_samples + rollout_samples
    ]
    bo_target_ground_truth = bo_target_sampled[
        warmup_samples : warmup_samples + rollout_samples
    ]
    bo_normal_statics = normalize(
        static_from_x0(bo_normal_x0, static_coefficients),
        static_center,
        static_scale,
    )
    bo_target_statics = normalize(
        static_from_x0(bo_target_x0, static_coefficients),
        static_center,
        static_scale,
    )
    print(f"BO normal validation x0: {bo_normal_x0.tolist()}")
    print(f"BO target validation x0: {bo_target_x0.tolist()}")

    params = RCParams()
    bo_rows: list[dict[str, float]] = []
    if not args.no_optimize:
        params, bo_rows = bayesian_optimize(
            args,
            train_sequences,
            train_statics,
            bo_normal_warmups,
            bo_normal_statics,
            bo_normal_ground_truth,
            bo_target_warmups,
            bo_target_statics,
            bo_target_ground_truth,
            sample_hz,
        )
        bo_path = data_dir / "paper_single_node_scrc_bo.csv"
        with bo_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(bo_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(bo_rows)

    rc, normal_prediction_norm, target_prediction_norm = fit_and_evaluate(
        params,
        train_sequences,
        train_statics,
        normal_warmup,
        normal_static,
        normal_gt,
        target_warmup,
        target_static,
        target_gt,
        args.fit_warmup,
        args.seed + 70000,
    )
    normal_prediction = (
        normal_prediction_norm * dynamic_scale + dynamic_center
    )
    target_prediction = (
        target_prediction_norm * dynamic_scale + dynamic_center
    )
    rollout_time = np.arange(rollout_samples) / sample_hz
    normal_calibration_predictions = rc.rollout_batch(
        bo_normal_warmups,
        bo_normal_statics,
        rollout_samples,
        args.seed + 81000,
    )
    feature_manifold = fit_feature_manifold(
        normal_calibration_predictions,
        sample_hz,
    )

    survival_rows = survival_scan(
        args,
        rc,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        static_coefficients,
        sample_hz,
        feature_manifold,
    )

    rollout_path = figure_dir / "paper_single_node_scrc_rollout.png"
    grid_path = figure_dir / "paper_single_node_scrc_2x2.png"
    attractor_path = figure_dir / "paper_single_node_scrc_attractor.png"
    feature_risk_path = (
        figure_dir / "paper_single_node_scrc_feature_risk.png"
    )
    survival_path = figure_dir / "paper_single_node_scrc_survival.png"
    plot_rollout_context(
        rollout_path,
        train_x0,
        train_static_observed,
        train_static_raw,
        target_time,
        target_proxy,
        target_onset,
        rollout_time,
        target_gt,
        target_prediction,
        args.warmup_s,
    )
    plot_2x2(
        grid_path,
        rollout_time,
        normal_gt,
        normal_prediction,
        target_gt,
        target_prediction,
    )
    plot_feature_risk(
        feature_risk_path,
        normal_prediction_norm,
        target_prediction_norm,
        feature_manifold,
        sample_hz,
    )
    plot_attractors(
        attractor_path,
        normal_gt,
        normal_prediction,
        target_gt,
        target_prediction,
        args.attractor_lag,
    )
    plot_survival(survival_path, survival_rows)

    summary_rows = []
    for index, x0 in enumerate(train_x0):
        summary_rows.append(
            {
                "label": f"train {index + 1}",
                "x0": x0,
                "onset_s": "",
                "std": train_static_raw[index, 0],
                "var": train_static_raw[index, 1],
                "observed_std": train_static_observed[index, 0],
                "observed_var": train_static_observed[index, 1],
            }
        )
    summary_rows.extend(
        [
            {
                "label": "normal validation",
                "x0": args.normal_x0,
                "onset_s": "" if normal_onset is None else normal_onset,
                "std": normal_static_raw[0],
                "var": normal_static_raw[1],
                "observed_std": normal_static_observed[0],
                "observed_var": normal_static_observed[1],
            },
            {
                "label": "ictal target",
                "x0": args.target_x0,
                "onset_s": "" if target_onset is None else target_onset,
                "std": target_static_raw[0],
                "var": target_static_raw[1],
                "observed_std": target_static_observed[0],
                "observed_var": target_static_observed[1],
            },
        ]
    )
    summary_path = data_dir / "paper_single_node_scrc_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    survival_csv = data_dir / "paper_single_node_scrc_survival.csv"
    with survival_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(survival_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(survival_rows)

    np.savez_compressed(
        data_dir / "paper_single_node_scrc_results.npz",
        rollout_time_s=rollout_time,
        normal_gt=normal_gt,
        normal_prediction=normal_prediction,
        target_gt=target_gt,
        target_prediction=target_prediction,
        target_onset_s=np.asarray(
            np.nan if target_onset is None else target_onset
        ),
        train_x0=np.asarray(train_x0),
        normal_x0=np.asarray(args.normal_x0),
        target_x0=np.asarray(args.target_x0),
        train_static_raw=train_static_raw,
        train_static_observed=train_static_observed,
        normal_static_raw=normal_static_raw,
        normal_static_observed=normal_static_observed,
        target_static_raw=target_static_raw,
        target_static_observed=target_static_observed,
        static_coefficients=static_coefficients,
        bo_normal_x0=bo_normal_x0,
        bo_target_x0=bo_target_x0,
        dynamic_center=np.asarray(dynamic_center),
        dynamic_scale=np.asarray(dynamic_scale),
        static_center=static_center,
        static_scale=static_scale,
        feature_manifold_center=feature_manifold.center,
        feature_manifold_scale=feature_manifold.scale,
        feature_manifold_inverse_covariance=(
            feature_manifold.inverse_covariance
        ),
        feature_manifold_threshold=np.asarray(
            feature_manifold.threshold
        ),
        rc_parameters=np.asarray(
            [
                params.spectral_radius,
                params.leakage_rate,
                params.input_scaling,
                params.static_scaling,
                params.ridge,
                params.density,
                params.innovation_scaling,
            ]
        ),
    )
    gt_target_event = detect_prediction_onset(target_gt, sample_hz)
    rc_target_event = detect_feature_escape(
        target_prediction_norm,
        feature_manifold,
        sample_hz,
    )
    print(f"Normal onset: {normal_onset}")
    print(f"Target onset: {target_onset}")
    print(f"GT target event after warmup: {gt_target_event}")
    print(f"RC target event after warmup: {rc_target_event}")
    print(f"Best RC parameters: {params}")
    print(f"Saved {grid_path}")
    print(f"Saved {attractor_path}")
    print(f"Saved {feature_risk_path}")
    print(f"Saved {survival_path}")


if __name__ == "__main__":
    main()
