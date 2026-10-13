#!/usr/bin/env python3
"""Strict single-node SCDT experiment.

The hidden Epileptor parameter x0 is used only to generate synthetic records and
to align a validation axis.  The reservoir sees one dynamic channel and two
statistics measured directly from each normal training record: standard
deviation and permutation entropy.  Hyperparameters are selected only by
normal-record statistic reproduction.  The statistical path is then swept once
to estimate the 50% normal-survival threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.neighbors import NearestNeighbors

from paper_epileptor_core import (
    PaperEpileptorConfig,
    default_initial_state,
    source_proxy,
    stochastic_heun_step,
)
from paper_single_node_scrc import RCParams, SparseStaticRC, expected_improvement


TRAIN_X0 = np.asarray([-2.30, -2.27, -2.24, -2.21, -2.18, -2.15])
CALIBRATION_X0 = np.asarray([-2.30, -2.26, -2.22, -2.18, -2.14, -2.10, -2.08, -2.07, -2.065])
VALIDATION_X0 = np.asarray([-2.285, -2.255, -2.225, -2.195, -2.165, -2.135])
GT_X0 = np.unique(
    np.r_[  # Coarse normal/event plateaus and fine transition grid.
        np.round(np.arange(-2.30, -2.0899, 0.01), 6),
        np.round(np.arange(-2.080, -2.0399, 0.001), 6),
        np.round(np.arange(-2.03, -1.8999, 0.01), 6),
        np.round(np.arange(-1.88, -1.6999, 0.02), 6),
    ]
)

EVENT_THRESHOLD = -0.5
EVENT_BLANK_S = 5.0

MORPHOLOGY_FEATURES = ("rolling_std",)


@dataclass
class MorphologyReference:
    std_floor: float
    training_min: float
    training_q005: float
    window_samples: int
    step_samples: int
    sustain_windows: int


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def permutation_entropy(signal: np.ndarray, order: int = 3) -> float:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < order + 1:
        return float("nan")
    patterns: dict[tuple[int, ...], int] = {}
    for start in range(len(signal) - order + 1):
        key = tuple(np.argsort(signal[start : start + order], kind="mergesort"))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.asarray(list(patterns.values()), dtype=float)
    probability = counts / counts.sum()
    return float(-np.sum(probability * np.log(probability + 1e-15)) / np.log(math.factorial(order)))


def statistics(signal: np.ndarray, sample_hz: float) -> np.ndarray:
    # PE is evaluated at about 16 Hz, matching the prior single-node analysis.
    stride = max(1, int(round(sample_hz / 16.0)))
    return np.asarray([np.std(signal), permutation_entropy(signal[::stride])], dtype=float)


def permutation_entropy_fast(signal: np.ndarray) -> float:
    """Vectorized order-3 permutation entropy for rolling morphology windows."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 4:
        return float("nan")
    windows = np.lib.stride_tricks.sliding_window_view(signal, 3)
    order = np.argsort(windows, axis=1, kind="stable")
    code = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
    counts = np.bincount(code, minlength=27).astype(float)
    probability = counts[counts > 0] / counts.sum()
    return float(-np.sum(probability * np.log(probability)) / np.log(6.0))


def morphology_vector(segment: np.ndarray, sample_hz: float, peak_prominence: float) -> np.ndarray:
    median = float(np.median(segment))
    centered = segment - median
    peaks, _ = find_peaks(segment, prominence=peak_prominence)
    pe_stride = max(1, int(round(sample_hz / 16.0)))
    return np.asarray(
        [
            median,
            np.std(segment),
            np.quantile(segment, 0.995) - median,
            np.mean(np.abs(np.diff(segment))),
            len(peaks) / (len(segment) / sample_hz),
            permutation_entropy_fast(segment[::pe_stride]),
        ],
        dtype=float,
    )


def rolling_morphology(
    signal: np.ndarray,
    sample_hz: float,
    window_samples: int,
    step_samples: int,
    peak_prominence: float,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(signal) - window_samples + 1, step_samples)
    features = np.asarray(
        [
            morphology_vector(signal[start : start + window_samples], sample_hz, peak_prominence)
            for start in starts
        ]
    )
    end_times = (starts + window_samples) / sample_hz
    return end_times, features


def fit_morphology_reference(
    training_signals: list[np.ndarray],
    sample_hz: float,
    window_s: float = 10.0,
    step_s: float = 1.0,
    sustain_windows: int = 5,
) -> MorphologyReference:
    """Learn a conservative low-activity floor from normal training only."""
    window_samples = int(round(window_s * sample_hz))
    step_samples = int(round(step_s * sample_hz))
    all_std = []
    for signal in training_signals:
        starts = np.arange(0, len(signal) - window_samples + 1, step_samples)
        all_std.extend(float(np.std(signal[start : start + window_samples])) for start in starts)
    values = np.asarray(all_std, dtype=float)
    training_min = float(np.min(values))
    return MorphologyReference(
        # The observed training minimum is conservative: no training window is
        # labelled abnormal, and the five-window sustain rule rejects brief dips.
        std_floor=training_min,
        training_min=training_min,
        training_q005=float(np.quantile(values, 0.005)),
        window_samples=window_samples,
        step_samples=step_samples,
        sustain_windows=sustain_windows,
    )


def morphology_event_mask(
    signals: np.ndarray,
    sample_hz: float,
    reference: MorphologyReference,
) -> np.ndarray:
    output = np.zeros(signals.shape[1], dtype=bool)
    for column in range(signals.shape[1]):
        signal = signals[:, column]
        starts = np.arange(
            0, len(signal) - reference.window_samples + 1, reference.step_samples
        )
        end_times = (starts + reference.window_samples) / sample_hz
        rolling_std = np.asarray(
            [np.std(signal[start : start + reference.window_samples]) for start in starts]
        )
        outside = (rolling_std < reference.std_floor) & (end_times >= EVENT_BLANK_S)
        run = 0
        for value in outside:
            run = run + 1 if value else 0
            if run >= reference.sustain_windows:
                output[column] = True
                break
    return output


def event_mask(
    signals: np.ndarray,
    sample_hz: float,
    morphology_reference: Optional[MorphologyReference] = None,
) -> np.ndarray:
    """Composite normal departure: trough crossing or sustained morphology escape."""
    start = int(round(EVENT_BLANK_S * sample_hz))
    trough = np.any(signals[start:] <= EVENT_THRESHOLD, axis=0)
    if morphology_reference is None:
        return trough
    return trough | morphology_event_mask(signals, sample_hz, morphology_reference)


def simulate_fixed_batch(x0: float, count: int, duration_s: float, seed: int) -> np.ndarray:
    config = PaperEpileptorConfig(n_nodes=count, duration_s=duration_s, seed=seed)
    rng = np.random.default_rng(seed)
    state = default_initial_state(count, rng)
    x0s = np.full(count, x0)
    connectivity = np.zeros((count, count))
    steps = int(round(duration_s * config.sampling_hz)) + 1
    output = np.empty((steps, count), dtype=float)
    output[0] = source_proxy(state)
    for index in range(1, steps):
        state = stochastic_heun_step(state, x0s, connectivity, config, rng)
        output[index] = source_proxy(state)
    return output


def simulate_switch_batch(
    target_x0: float,
    count: int,
    base_x0: float,
    warmup_s: float,
    rollout_s: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    config = PaperEpileptorConfig(n_nodes=count, duration_s=warmup_s + rollout_s, seed=seed)
    rng = np.random.default_rng(seed)
    state = default_initial_state(count, rng)
    connectivity = np.zeros((count, count))
    warmup_steps = int(round(warmup_s * config.sampling_hz))
    rollout_steps = int(round(rollout_s * config.sampling_hz))
    warmup = np.empty((warmup_steps, count), dtype=float)
    future = np.empty((rollout_steps, count), dtype=float)
    for index in range(warmup_steps):
        state = stochastic_heun_step(
            state, np.full(count, base_x0), connectivity, config, rng
        )
        warmup[index] = source_proxy(state)
    for index in range(rollout_steps):
        state = stochastic_heun_step(
            state, np.full(count, target_x0), connectivity, config, rng
        )
        future[index] = source_proxy(state)
    return warmup, future


def normal_record(x0: float, duration_s: float, stride: int, seed: int) -> np.ndarray:
    """Generate one fixed-x0 record and enforce the preregistered normal rule."""
    for attempt in range(20):
        raw = simulate_fixed_batch(x0, 1, duration_s, seed + attempt * 100003)[:, 0]
        sampled = raw[::stride]
        if not event_mask(sampled[:, None], 256.0 / stride)[0]:
            return sampled
    raise RuntimeError(f"Could not obtain a normal record at x0={x0}")


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    denominator = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum((y - fitted) ** 2) / max(denominator, 1e-15)
    return float(slope), float(intercept), float(r2)


def crossing(x: np.ndarray, survival: np.ndarray, level: float = 0.5) -> float:
    """Descending crossing after the maximum-survival normal branch."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(survival, dtype=float)
    peak = int(np.argmax(y))
    hits = np.flatnonzero((np.arange(len(y)) > peak) & (y <= level))
    if not len(hits):
        return float("nan")
    right = int(hits[0])
    left = right - 1
    if y[left] == y[right]:
        return float(0.5 * (x[left] + x[right]))
    return float(x[left] + (level - y[left]) * (x[right] - x[left]) / (y[right] - y[left]))


def wilson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def params_from_unit(unit: np.ndarray, n_reservoir: int, seed: int) -> RCParams:
    bounds = np.asarray(
        [
            [0.35, 1.25],   # spectral radius
            [0.02, 0.35],   # leakage
            [0.05, 1.10],   # dynamic input
            [0.02, 1.20],   # static input
            [-7.5, -2.0],   # log10 ridge
            [0.02, 0.12],   # density
            [0.00, 1.10],   # residual innovation
        ]
    )
    value = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    return RCParams(
        n_reservoir=n_reservoir,
        spectral_radius=float(value[0]),
        leakage_rate=float(value[1]),
        input_scaling=float(value[2]),
        static_scaling=float(value[3]),
        ridge=float(10.0 ** value[4]),
        density=float(value[5]),
        innovation_scaling=float(value[6]),
        seed=seed,
    )


def starter_units() -> list[np.ndarray]:
    """Deterministic, generic RC starting points; selection remains normal-only."""
    bounds = np.asarray(
        [
            [0.35, 1.25], [0.02, 0.35], [0.05, 1.10], [0.02, 1.20],
            [-7.5, -2.0], [0.02, 0.12], [0.00, 1.10],
        ]
    )
    physical = [
        [0.77, 0.28, 0.57, 0.14, math.log10(6.0e-4), 0.065, 0.98],
        [0.80, 0.10, 0.40, 0.10, -4.0, 0.050, 0.50],
        [0.60, 0.20, 0.50, 0.20, -4.0, 0.060, 0.80],
        [1.00, 0.08, 0.25, 0.08, -5.0, 0.040, 0.25],
    ]
    return [np.clip((np.asarray(row) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]), 0.0, 1.0) for row in physical]


def build_rc(
    params: RCParams,
    sequences_norm: list[np.ndarray],
    statics_norm: np.ndarray,
    fit_warmup: int,
) -> SparseStaticRC:
    rc = SparseStaticRC(params, static_dim=2)
    rc.fit(sequences_norm, [row for row in statics_norm], fit_warmup)
    return rc


def normal_validation_score(
    rc: SparseStaticRC,
    validation: list[np.ndarray],
    validation_stat_raw: np.ndarray,
    train_sequences_norm: list[np.ndarray],
    train_stat_norm: np.ndarray,
    dynamic_center: float,
    dynamic_scale: float,
    static_center: np.ndarray,
    static_scale: np.ndarray,
    sample_hz: float,
    warmup_samples: int,
    rollout_samples: int,
    seed: int,
    morphology_reference: MorphologyReference,
    repeats: int = 3,
) -> dict:
    predictions: list[np.ndarray] = []
    target_stats: list[np.ndarray] = []
    generated_stats: list[np.ndarray] = []
    false_events = 0
    for index, (target_signal, target_raw) in enumerate(zip(validation, validation_stat_raw)):
        nearest = int(np.argmin(np.abs(train_stat_norm[:, 0] - ((target_raw - static_center) / static_scale)[0])))
        warmup = np.repeat(train_sequences_norm[nearest][-warmup_samples:, None], repeats, axis=1)
        warmup_static = np.repeat(train_stat_norm[nearest][None, :], repeats, axis=0)
        target_static = np.repeat(((target_raw - static_center) / static_scale)[None, :], repeats, axis=0)
        pred_norm = rc.rollout_switch_batch(
            warmup,
            warmup_static,
            target_static,
            rollout_samples,
            seed + index,
        )
        pred = pred_norm * dynamic_scale + dynamic_center
        generated_repeats = np.asarray([statistics(pred[:, repeat], sample_hz) for repeat in range(repeats)])
        generated = np.mean(generated_repeats, axis=0)
        predictions.append(pred[:, 0])
        target_stats.append(target_raw)
        generated_stats.append(generated)
        false_events += int(np.sum(event_mask(pred, sample_hz, morphology_reference)))
    target_stats_array = np.asarray(target_stats)
    generated_stats_array = np.asarray(generated_stats)
    scales = np.maximum(np.ptp(validation_stat_raw, axis=0), np.asarray([0.03, 0.01]))
    normalized_error = np.abs(generated_stats_array - target_stats_array) / scales
    stat_mae = float(np.mean(np.clip(normalized_error, 0.0, 10.0)))
    false_rate = false_events / (len(validation) * repeats)
    # Normal-only objective: measured statistic reproduction and no false departure.
    score = stat_mae + 8.0 * false_rate
    return {
        "score": score,
        "stat_mae": stat_mae,
        "false_departure_rate": false_rate,
        "target_stats": target_stats_array,
        "generated_stats": generated_stats_array,
        "predictions": predictions,
    }


def optimize_normal_only(
    args: argparse.Namespace,
    sequences_norm: list[np.ndarray],
    train_stat_norm: np.ndarray,
    validation: list[np.ndarray],
    validation_stat_raw: np.ndarray,
    dynamic_center: float,
    dynamic_scale: float,
    static_center: np.ndarray,
    static_scale: np.ndarray,
    sample_hz: float,
    morphology_reference: MorphologyReference,
) -> tuple[RCParams, list[dict]]:
    rng = np.random.default_rng(args.seed + 7000)
    units: list[np.ndarray] = []
    scores: list[float] = []
    rows: list[dict] = []
    starters = starter_units()
    initial = min(10, max(6, args.bo_calls // 3))
    for iteration in range(args.bo_calls):
        if iteration < len(starters):
            unit = starters[iteration]
        elif iteration < initial:
            unit = rng.random(7)
        else:
            gp = GaussianProcessRegressor(
                kernel=Matern(length_scale=np.ones(7), nu=2.5) + WhiteKernel(1e-5),
                normalize_y=True,
                random_state=args.seed,
                n_restarts_optimizer=1,
            )
            gp.fit(np.asarray(units), np.asarray(scores))
            candidates = rng.random((600, 7))
            unit = candidates[int(np.argmax(expected_improvement(candidates, gp, min(scores))))]
        # Same random reservoir and validation innovations for every candidate.
        params = params_from_unit(unit, args.reservoir_nodes, args.reservoir_seed)
        try:
            rc = build_rc(params, sequences_norm, train_stat_norm, args.fit_warmup)
            result = normal_validation_score(
                rc,
                validation,
                validation_stat_raw,
                sequences_norm,
                train_stat_norm,
                dynamic_center,
                dynamic_scale,
                static_center,
                static_scale,
                sample_hz,
                args.warmup_samples,
                args.validation_rollout_samples,
                args.seed + 8000,
                morphology_reference,
            )
        except Exception as error:
            print(f"BO candidate failed: {error}", flush=True)
            result = {"score": 1e6, "stat_mae": 1e6, "false_departure_rate": 1.0}
        units.append(unit)
        scores.append(float(result["score"]))
        row = {
            "iteration": iteration + 1,
            "normal_only_score": result["score"],
            "statistic_mae_scaled": result["stat_mae"],
            "false_departure_rate": result["false_departure_rate"],
            **asdict(params),
        }
        rows.append(row)
        print(
            f"normal-only BO {iteration + 1:02d}/{args.bo_calls}: "
            f"score={row['normal_only_score']:.4f}, "
            f"stat={row['statistic_mae_scaled']:.4f}, "
            f"false={row['false_departure_rate']:.3f}",
            flush=True,
        )
    best = int(np.argmin(scores))
    return params_from_unit(units[best], args.reservoir_nodes, args.reservoir_seed), rows


def sweep_rate(
    rc: SparseStaticRC,
    sigma_grid: np.ndarray,
    path_slope: float,
    path_intercept: float,
    sequences_norm: list[np.ndarray],
    train_stat_norm: np.ndarray,
    dynamic_center: float,
    dynamic_scale: float,
    static_center: np.ndarray,
    static_scale: np.ndarray,
    sample_hz: float,
    warmup_samples: int,
    rollout_samples: int,
    ensemble: int,
    seed_base: int,
    morphology_reference: MorphologyReference,
    condition_independent: bool = False,
) -> tuple[list[dict], dict[float, np.ndarray]]:
    rows: list[dict] = []
    examples: dict[float, np.ndarray] = {}
    for grid_index, sigma in enumerate(sigma_grid):
        raw = np.asarray([sigma, path_slope * sigma + path_intercept])
        target = (raw - static_center) / static_scale
        indices = np.arange(ensemble) % len(sequences_norm)
        warmups = np.column_stack([sequences_norm[i][-warmup_samples:] for i in indices])
        warmup_statics = train_stat_norm[indices]
        targets = np.repeat(target[None, :], ensemble, axis=0)
        if condition_independent:
            warmup_statics = np.zeros_like(warmup_statics)
            targets = np.zeros_like(targets)
        pred_norm = rc.rollout_switch_batch(
            warmups,
            warmup_statics,
            targets,
            rollout_samples,
            seed_base + grid_index,
        )
        pred = pred_norm * dynamic_scale + dynamic_center
        events = event_mask(pred, sample_hz, morphology_reference)
        survivors = int(np.sum(~events))
        low, high = wilson(survivors, ensemble)
        generated = np.asarray([statistics(pred[:, j], sample_hz) for j in range(ensemble)])
        rows.append(
            {
                "sigma_input": float(sigma),
                "permutation_entropy_input": float(raw[1]),
                "survivors": survivors,
                "ensemble": ensemble,
                "survival_rate": survivors / ensemble,
                "wilson_low": low,
                "wilson_high": high,
                "generated_sigma_mean": float(np.mean(generated[:, 0])),
                "generated_sigma_sd": float(np.std(generated[:, 0])),
                "generated_pe_mean": float(np.mean(generated[:, 1])),
                "generated_pe_sd": float(np.std(generated[:, 1])),
            }
        )
        if grid_index in {0, len(sigma_grid) // 2, len(sigma_grid) - 1}:
            examples[float(sigma)] = pred[:, : min(3, ensemble)]
        print(
            f"SCDT sweep {grid_index + 1:02d}/{len(sigma_grid)}: "
            f"sigma={sigma:.4f}, survival={survivors}/{ensemble}",
            flush=True,
        )
    return rows, examples


def bootstrap_crossing(sigma: np.ndarray, survivors: np.ndarray, total: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        sampled = rng.binomial(total, survivors / total) / total
        value = crossing(sigma, sampled)
        if np.isfinite(value):
            values.append(value)
    if len(values) < 100:
        return float("nan"), float("nan")
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "figure.facecolor": "white",
        }
    )


def plot_clean_survival(
    figure_path: Path,
    sweep_rows: list[dict],
    gt_rows: list[dict],
    title: str = "Single-node normal survival",
) -> None:
    """Only the two requested survival curves; retain full tables separately."""
    configure_style()
    sigma = np.asarray([float(row["sigma_input"]) for row in sweep_rows])
    rc_rate = np.asarray([float(row["survival_rate"]) for row in sweep_rows])
    gt_sigma = np.asarray([float(row["sigma_coordinate"]) for row in gt_rows])
    gt_rate = np.asarray([float(row["survival_rate"]) for row in gt_rows])
    gt_order = np.argsort(gt_sigma)
    fig, ax = plt.subplots(figsize=(9.5, 5.1))
    ax.plot(gt_sigma[gt_order], gt_rate[gt_order], color="#202020", lw=2.2, label="GT")
    ax.plot(sigma, rc_rate, color="#2166ac", lw=2.2, marker="o", ms=4, label="SCDT")
    ax.set(
        xlabel=r"statistical coordinate $\sigma$",
        ylabel="normal survival rate",
        ylim=(-0.04, 1.04),
        title=title,
    )
    ax.xaxis.set_major_locator(MaxNLocator(8))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=240)
    plt.close(fig)


def make_figures(
    figure_dir: Path,
    train_stats: np.ndarray,
    calibration_rows: list[dict],
    validation_stats: np.ndarray,
    validation_result: dict,
    sigma_path: tuple[float, float, float],
    sigma_x0: tuple[float, float, float],
    pe_x0: tuple[float, float, float],
    sweep_rows: list[dict],
    control_rows: list[dict],
    gt_rows: list[dict],
    thresholds: dict,
    examples: dict[float, np.ndarray],
    sample_hz: float,
) -> None:
    configure_style()
    calibration_x0 = np.asarray([r["x0"] for r in calibration_rows])
    calibration_sigma = np.asarray([r["sigma_mean"] for r in calibration_rows])
    calibration_sigma_sd = np.asarray([r["sigma_sd"] for r in calibration_rows])
    calibration_pe = np.asarray([r["pe_mean"] for r in calibration_rows])
    calibration_pe_sd = np.asarray([r["pe_sd"] for r in calibration_rows])
    grid_x0 = np.linspace(calibration_x0.min(), calibration_x0.max(), 300)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(calibration_x0, calibration_sigma, yerr=calibration_sigma_sd, fmt="o", capsize=3, label="held-out normal records")
    axes[0].plot(grid_x0, sigma_x0[0] * grid_x0 + sigma_x0[1], lw=2, label=f"linear fit ($R^2$={sigma_x0[2]:.3f})")
    axes[0].set(xlabel=r"hidden $x_0$ (validation only)", ylabel=r"measured $\sigma$", title="Parameter–statistic relation")
    axes[0].legend(fontsize=8)
    axes[1].errorbar(calibration_x0, calibration_pe, yerr=calibration_pe_sd, fmt="o", capsize=3, color="#7b3294", label="held-out normal records")
    axes[1].plot(grid_x0, pe_x0[0] * grid_x0 + pe_x0[1], lw=2, color="#c51b7d", label=f"linear fit ($R^2$={pe_x0[2]:.3f})")
    axes[1].set(xlabel=r"hidden $x_0$ (validation only)", ylabel="permutation entropy", title="Parameter–statistic relation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "01_parameter_statistic_relations.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.scatter(train_stats[:, 0], train_stats[:, 1], s=65, color="#2166ac", label="measured training records", zorder=3)
    sigma_grid = np.linspace(min(r["sigma_input"] for r in sweep_rows), max(r["sigma_input"] for r in sweep_rows), 300)
    ax.plot(sigma_grid, sigma_path[0] * sigma_grid + sigma_path[1], color="#b2182b", lw=2, label=f"direct path ($R^2$={sigma_path[2]:.3f})")
    for index, point in enumerate(train_stats):
        ax.annotate(str(index + 1), point, xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axvspan(train_stats[:, 0].min(), train_stats[:, 0].max(), color="#2166ac", alpha=0.08, label="training range")
    ax.set(xlabel=r"leading statistic $\sigma$", ylabel="permutation entropy", title="SCDT conditioning path measured directly from normal records")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "02_statistical_conditioning_path.png", dpi=240)
    plt.close(fig)

    target = validation_result["target_stats"]
    generated = validation_result["generated_stats"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    labels = [(0, r"$\sigma$"), (1, "permutation entropy")]
    for axis, (idx, label) in zip(axes, labels):
        low = min(target[:, idx].min(), generated[:, idx].min())
        high = max(target[:, idx].max(), generated[:, idx].max())
        axis.scatter(target[:, idx], generated[:, idx], s=55, color="#1b7837")
        axis.plot([low, high], [low, high], "--", color="0.35")
        axis.set(xlabel=f"measured normal {label}", ylabel=f"RC reproduced {label}", title="Held-out normal validation")
    fig.suptitle(f"Normal-only model selection: scaled statistic MAE={validation_result['stat_mae']:.3f}, false departure={validation_result['false_departure_rate']:.3f}")
    fig.tight_layout()
    fig.savefig(figure_dir / "03_normal_validation.png", dpi=240)
    plt.close(fig)

    plot_clean_survival(figure_dir / "04_rate_threshold.png", sweep_rows, gt_rows)
    s_sigma = np.asarray([r["sigma_input"] for r in sweep_rows])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True, sharey=True)
    for axis, (sigma, signals) in zip(axes, sorted(examples.items())):
        time_s = np.arange(len(signals)) / sample_hz
        for idx in range(signals.shape[1]):
            axis.plot(time_s, signals[:, idx], lw=0.7, alpha=0.8)
        axis.axhline(EVENT_THRESHOLD, color="#b2182b", ls="--", lw=1)
        rate = next(r["survival_rate"] for r in sweep_rows if np.isclose(r["sigma_input"], sigma))
        axis.set_title(f"RC autonomous trajectories: sigma={sigma:.4f}, survival={rate:.2f}", loc="left")
        axis.set_ylabel("proxy")
    axes[-1].set_xlabel("autonomous time (s)")
    fig.tight_layout()
    fig.savefig(figure_dir / "05_rc_trajectory_examples.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    gen_sigma = np.asarray([r["generated_sigma_mean"] for r in sweep_rows])
    gen_sigma_sd = np.asarray([r["generated_sigma_sd"] for r in sweep_rows])
    gen_pe = np.asarray([r["generated_pe_mean"] for r in sweep_rows])
    gen_pe_sd = np.asarray([r["generated_pe_sd"] for r in sweep_rows])
    pe_input = np.asarray([r["permutation_entropy_input"] for r in sweep_rows])
    axes[0].errorbar(s_sigma, gen_sigma, yerr=gen_sigma_sd, fmt="o-", ms=3, capsize=2)
    axes[0].plot(s_sigma, s_sigma, "--", color="0.35")
    axes[0].set(xlabel=r"input $\sigma$", ylabel=r"generated $\sigma$", title="Statistic reproduction along sweep")
    axes[1].errorbar(pe_input, gen_pe, yerr=gen_pe_sd, fmt="o-", ms=3, capsize=2, color="#7b3294")
    axes[1].plot(pe_input, pe_input, "--", color="0.35")
    axes[1].set(xlabel="input permutation entropy", ylabel="generated permutation entropy", title="Statistic reproduction along sweep")
    fig.tight_layout()
    fig.savefig(figure_dir / "06_sweep_statistic_reproduction.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(".."))
    parser.add_argument("--duration-s", type=float, default=160.0)
    parser.add_argument("--warmup-s", type=float, default=20.0)
    parser.add_argument("--rollout-s", type=float, default=140.0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--training-replicates", type=int, default=3)
    parser.add_argument("--calibration-replicates", type=int, default=5)
    parser.add_argument("--gt-ensemble", type=int, default=30)
    parser.add_argument("--rc-ensemble", type=int, default=30)
    parser.add_argument("--reservoir-nodes", type=int, default=280)
    parser.add_argument("--bo-calls", type=int, default=24)
    parser.add_argument("--sweep-points", type=int, default=35)
    parser.add_argument("--fit-warmup", type=int, default=160)
    parser.add_argument("--reservoir-seed", type=int, default=91017)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    out = args.out_dir.resolve()
    data_dir = out / "data"
    figure_dir = out / "figures"
    result_dir = out / "results"
    for directory in (data_dir, figure_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)
    sample_hz = 256.0 / args.stride
    args.warmup_samples = int(round(args.warmup_s * sample_hz))
    args.validation_rollout_samples = int(round(60.0 * sample_hz))
    rollout_samples = int(round(args.rollout_s * sample_hz))

    print("1/6 generating disjoint full-length normal training records", flush=True)
    training: list[np.ndarray] = []
    train_rows: list[dict] = []
    for index, x0 in enumerate(TRAIN_X0):
        for replicate in range(args.training_replicates):
            signal = normal_record(
                float(x0), args.duration_s, args.stride,
                args.seed + 1000 + index * 1000 + replicate,
            )
            stat = statistics(signal, sample_hz)
            training.append(signal)
            train_rows.append(
                {
                    "record": len(train_rows) + 1,
                    "x0_validation_only": float(x0),
                    "replicate": replicate,
                    "sigma": stat[0],
                    "permutation_entropy": stat[1],
                    "samples": len(signal),
                }
            )
    train_stats = np.asarray([[r["sigma"], r["permutation_entropy"]] for r in train_rows])
    path_slope, path_intercept, path_r2 = fit_line(train_stats[:, 0], train_stats[:, 1])

    dynamic_values = np.concatenate(training)
    dynamic_center = float(np.mean(dynamic_values))
    dynamic_scale = float(np.std(dynamic_values))
    static_center = np.mean(train_stats, axis=0)
    static_scale = np.std(train_stats, axis=0)
    static_scale = np.where(static_scale > 1e-9, static_scale, 1.0)
    sequences_norm = [(s - dynamic_center) / dynamic_scale for s in training]
    train_stat_norm = (train_stats - static_center) / static_scale
    morphology_reference = fit_morphology_reference(training, sample_hz)

    print("2/6 generating held-out normal calibration and validation records", flush=True)
    calibration_rows: list[dict] = []
    for x_index, x0 in enumerate(CALIBRATION_X0):
        values = []
        for replicate in range(args.calibration_replicates):
            signal = normal_record(float(x0), args.duration_s, args.stride, args.seed + 100000 + x_index * 1000 + replicate)
            values.append(statistics(signal, sample_hz))
        array = np.asarray(values)
        calibration_rows.append({"x0": float(x0), "replicates": len(array), "sigma_mean": float(array[:, 0].mean()), "sigma_sd": float(array[:, 0].std()), "pe_mean": float(array[:, 1].mean()), "pe_sd": float(array[:, 1].std())})
    sigma_x0 = fit_line(np.asarray([r["x0"] for r in calibration_rows]), np.asarray([r["sigma_mean"] for r in calibration_rows]))
    pe_x0 = fit_line(np.asarray([r["x0"] for r in calibration_rows]), np.asarray([r["pe_mean"] for r in calibration_rows]))

    validation: list[np.ndarray] = []
    validation_stats = []
    for index, x0 in enumerate(VALIDATION_X0):
        signal = normal_record(float(x0), args.duration_s, args.stride, args.seed + 200000 + index * 1000)
        validation.append(signal)
        validation_stats.append(statistics(signal, sample_hz))
    validation_stats_array = np.asarray(validation_stats)
    heldout_normal_gt_false_departure = float(
        np.mean(event_mask(np.column_stack(validation), sample_hz, morphology_reference))
    )
    print(
        f"training-only low-activity reference: "
        f"10 s std floor={morphology_reference.std_floor:.5f}, "
        f"held-out normal GT false departure={heldout_normal_gt_false_departure:.3f}",
        flush=True,
    )

    print("3/6 normal-only Bayesian hyperparameter search", flush=True)
    best_params, bo_rows = optimize_normal_only(
        args,
        sequences_norm,
        train_stat_norm,
        validation,
        validation_stats_array,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        sample_hz,
        morphology_reference,
    )
    rc = build_rc(best_params, sequences_norm, train_stat_norm, args.fit_warmup)
    validation_result = normal_validation_score(
        rc,
        validation,
        validation_stats_array,
        sequences_norm,
        train_stat_norm,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        sample_hz,
        args.warmup_samples,
        args.validation_rollout_samples,
        args.seed + 8000,
        morphology_reference,
    )

    print("4/6 locked-model statistical sweep", flush=True)
    sigma_min = float(train_stats[:, 0].min())
    sigma_max = min(0.50, float(train_stats[:, 0].max() + 0.11))
    sigma_grid = np.linspace(sigma_min, sigma_max, args.sweep_points)
    sweep_rows, examples = sweep_rate(
        rc, sigma_grid, path_slope, path_intercept, sequences_norm, train_stat_norm,
        dynamic_center, dynamic_scale, static_center, static_scale, sample_hz,
        args.warmup_samples, rollout_samples, args.rc_ensemble, args.seed + 300000,
        morphology_reference,
    )

    # Negative control: the same RC architecture receives no statistical condition.
    zero_statics = np.zeros_like(train_stat_norm)
    control_rc = build_rc(best_params, sequences_norm, zero_statics, args.fit_warmup)
    control_rows, _ = sweep_rate(
        control_rc, sigma_grid, path_slope, path_intercept, sequences_norm, zero_statics,
        dynamic_center, dynamic_scale, static_center, static_scale, sample_hz,
        args.warmup_samples, rollout_samples, args.rc_ensemble, args.seed + 400000,
        morphology_reference,
        condition_independent=True,
    )

    print("5/6 fresh held-out GT boundary cohort", flush=True)
    gt_rows: list[dict] = []
    for index, x0 in enumerate(GT_X0):
        _, future = simulate_switch_batch(float(x0), args.gt_ensemble, -2.18, args.warmup_s, args.rollout_s, args.seed + 500000 + index * 1000)
        sampled = future[:: args.stride]
        survival = float(np.mean(~event_mask(sampled, sample_hz, morphology_reference)))
        sigma_coordinate = sigma_x0[0] * x0 + sigma_x0[1]
        gt_rows.append({"x0": float(x0), "sigma_coordinate": float(sigma_coordinate), "survivors": int(round(survival * args.gt_ensemble)), "ensemble": args.gt_ensemble, "survival_rate": survival})
        print(f"GT {index + 1:02d}/{len(GT_X0)} x0={x0:.4f}: survival={survival:.3f}", flush=True)

    s_sigma = np.asarray([r["sigma_input"] for r in sweep_rows])
    s_surv = np.asarray([r["survival_rate"] for r in sweep_rows])
    survivors = np.asarray([r["survivors"] for r in sweep_rows])
    rc_threshold = crossing(s_sigma, s_surv)
    rc_ci = (
        bootstrap_crossing(s_sigma, survivors, args.rc_ensemble, args.seed + 600000)
        if np.isfinite(rc_threshold)
        else (float("nan"), float("nan"))
    )
    gt_x0_values = np.asarray([r["x0"] for r in gt_rows])
    gt_survival = np.asarray([r["survival_rate"] for r in gt_rows])
    gt_x0_threshold = crossing(gt_x0_values, gt_survival)
    gt_sigma_threshold = sigma_x0[0] * gt_x0_threshold + sigma_x0[1]
    thresholds = {
        "rc_sigma_50": rc_threshold,
        "rc_sigma_50_bootstrap_95_low": rc_ci[0],
        "rc_sigma_50_bootstrap_95_high": rc_ci[1],
        "rc_aligned_x0_50_validation_only": (rc_threshold - sigma_x0[1]) / sigma_x0[0] if np.isfinite(rc_threshold) else float("nan"),
        "gt_x0_50": gt_x0_threshold,
        "gt_sigma_50": gt_sigma_threshold,
        "sigma_threshold_error": rc_threshold - gt_sigma_threshold if np.isfinite(rc_threshold) else float("nan"),
    }

    print("6/6 writing audit tables and figures", flush=True)
    write_csv(data_dir / "training_records.csv", train_rows)
    write_csv(data_dir / "calibration_records.csv", calibration_rows)
    write_csv(data_dir / "bo_normal_only.csv", bo_rows)
    write_csv(data_dir / "scdt_rate_sweep.csv", sweep_rows)
    write_csv(data_dir / "condition_independent_control.csv", control_rows)
    write_csv(data_dir / "gt_boundary.csv", gt_rows)
    np.savez_compressed(
        data_dir / "training_signals.npz",
        signals=np.asarray(training),
        x0_validation_only=np.asarray([row["x0_validation_only"] for row in train_rows]),
        replicate=np.asarray([row["replicate"] for row in train_rows]),
        statistics=train_stats,
        sample_hz=sample_hz,
    )
    np.savez_compressed(
        data_dir / "morphology_reference.npz",
        feature_names=np.asarray(MORPHOLOGY_FEATURES),
        std_floor=morphology_reference.std_floor,
        training_min=morphology_reference.training_min,
        training_q005=morphology_reference.training_q005,
        window_samples=morphology_reference.window_samples,
        step_samples=morphology_reference.step_samples,
        sustain_windows=morphology_reference.sustain_windows,
    )
    make_figures(
        figure_dir, train_stats, calibration_rows, validation_stats_array,
        validation_result, (path_slope, path_intercept, path_r2), sigma_x0,
        pe_x0, sweep_rows, control_rows, gt_rows, thresholds, examples, sample_hz,
    )
    protocol = {
        "experiment": "strict single-node SCDT",
        "separation_rules": {
            "reservoir_inputs": ["dynamic proxy x2-x1", "measured sigma", "measured permutation entropy"],
            "x0_use": "synthetic data generation and validation-axis alignment only",
            "training": "normal fixed-x0 records only; no postcritical/event trajectory",
            "record_definition": "each 160 s fixed-x0 trajectory is one intact record; no window segmentation",
            "hyperparameter_selection": "held-out normal statistic reproduction only",
            "final_threshold_test": "executed after model locking",
            "warmup": "last 20 s of training records only; no target GT warmup",
        },
        "event_rule": (
            f"proxy <= {EVENT_THRESHOLD} after {EVENT_BLANK_S} s blanking OR "
            "10 s rolling std below the normal-training minimum for 5 consecutive 1 s steps"
        ),
        "morphology_reference": {
            "features": list(MORPHOLOGY_FEATURES),
            "window_s": morphology_reference.window_samples / sample_hz,
            "step_s": morphology_reference.step_samples / sample_hz,
            "sustain_windows": morphology_reference.sustain_windows,
            "std_floor": morphology_reference.std_floor,
            "training_min": morphology_reference.training_min,
            "training_q005": morphology_reference.training_q005,
            "heldout_normal_gt_false_departure": heldout_normal_gt_false_departure,
        },
        "training_x0_validation_only": TRAIN_X0.tolist(),
        "calibration_x0_validation_only": CALIBRATION_X0.tolist(),
        "normal_validation_x0_validation_only": VALIDATION_X0.tolist(),
        "gt_test_x0": GT_X0.tolist(),
        "statistical_path": {"leading": "sigma", "secondary": "permutation_entropy", "slope": path_slope, "intercept": path_intercept, "r2": path_r2},
        "x0_sigma_alignment_validation_only": {"slope": sigma_x0[0], "intercept": sigma_x0[1], "r2": sigma_x0[2]},
        "x0_pe_alignment_validation_only": {"slope": pe_x0[0], "intercept": pe_x0[1], "r2": pe_x0[2]},
        "best_rc_parameters": asdict(best_params),
        "normal_validation": {"scaled_statistic_mae": validation_result["stat_mae"], "false_departure_rate": validation_result["false_departure_rate"]},
        "thresholds": thresholds,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    with (result_dir / "protocol.json").open("w") as handle:
        json.dump(protocol, handle, indent=2, allow_nan=True)
    with (result_dir / "summary.json").open("w") as handle:
        json.dump({"thresholds": thresholds, "normal_validation": protocol["normal_validation"], "best_rc_parameters": asdict(best_params)}, handle, indent=2, allow_nan=True)
    print(json.dumps(thresholds, indent=2), flush=True)


if __name__ == "__main__":
    main()
