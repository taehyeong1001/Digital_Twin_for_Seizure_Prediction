#!/usr/bin/env python3
"""Three-node SCDT with a calibrated statistical coordinate and survival BO."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import Ridge

from experiment_three_node_bifurcation_rc import (
    N_NODES,
    RC_HZ,
    latin_hypercube,
    normalize,
    save_json,
    simulate_paths,
    three_node_connectivity,
)
from experiment_three_node_parameter_space_rc import (
    CONTEXT_NAMES,
    ResidualMultiOutputStaticRC,
    fit_normalizer,
    trajectory_context,
)
from paper_epileptor_core import PaperEpileptorConfig, X0_CRITICAL, source_proxy
from paper_multinode_coupled_scdt import (
    MultiOutputStaticRC,
    initial_state_batch,
    network_heun_step_batch,
)
from paper_single_node_scrc import RCParams


ROOT = Path(__file__).resolve().parents[1]
BASE_X0 = -2.30
DEFAULT_EVENT_THRESHOLD = -0.5
DEFAULT_EVENT_BLANKING_S = 5.0
MORPHOLOGY_SCALES = {
    "normal_mean_mae": 0.15,
    "normal_log_std_mae": 0.30,
    "normal_ac1_mae": 0.03,
    "normal_spike_rate_mae": 0.02,
    "normal_psd_l1": 0.80,
}


def first_threshold_crossing(
    signals: np.ndarray,
    threshold: float,
    start_s: float = 0.0,
) -> np.ndarray:
    """Return the first time any node reaches the normal-departure threshold."""
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 3:
        raise ValueError("signals must have shape (time, records, nodes)")
    start = int(round(start_s * RC_HZ))
    if start < 0 or start >= len(signals):
        raise ValueError("start_s must fall inside the signal")
    crossed = np.any(signals[start:] <= threshold, axis=2)
    event = (np.argmax(crossed, axis=0).astype(float) + start) / RC_HZ
    event[~np.any(crossed, axis=0)] = np.nan
    return event


def parameter_from_unit(unit: np.ndarray, seed: int) -> tuple[RCParams, str]:
    unit = np.asarray(unit, dtype=float)
    if unit.shape != (7,) or np.any(unit < 0.0) or np.any(unit > 1.0):
        raise ValueError("SCDT BO coordinate must have shape (7,) inside [0, 1]")
    params = RCParams(
        n_reservoir=1500,
        spectral_radius=0.30 + 2.20 * unit[0],
        leakage_rate=10 ** (-2.0 + np.log10(70.0) * unit[1]),
        input_scaling=10 ** (-1.7 + 1.6 * unit[2]),
        static_scaling=10 ** (-2.5 + 2.2 * unit[3]),
        ridge=10 ** (-3.0 + 4.5 * unit[4]),
        density=0.003,
        innovation_scaling=0.60 * unit[5],
        seed=seed,
    )
    mode = "next" if unit[6] < 0.5 else "delta"
    return params, mode


def build_reservoir(params: RCParams, mode: str) -> MultiOutputStaticRC:
    cls = ResidualMultiOutputStaticRC if mode == "delta" else MultiOutputStaticRC
    return cls(
        params,
        dynamic_dim=N_NODES,
        static_dim=len(CONTEXT_NAMES),
        readout_stride=4,
        readout_size=1500,
    )


def make_stable_training(
    count: int,
    seed: int,
    duration_s: float,
    event_threshold: float = DEFAULT_EVENT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select full event-free trajectories from the precritical 3-D cube."""
    candidate_count = count * 2
    profiles = latin_hypercube(
        candidate_count, -2.50, X0_CRITICAL - 0.04, np.random.default_rng(seed)
    )
    signals, _ = simulate_paths(
        profiles,
        profiles,
        duration_s=duration_s,
        hold_s=0.0,
        seed=seed + 1,
    )
    events = first_threshold_crossing(signals, event_threshold)
    eligible = np.flatnonzero(~np.isfinite(events))
    if len(eligible) < count:
        raise RuntimeError(f"Only {len(eligible)} event-free training records; need {count}")
    order = eligible[np.argsort(np.max(profiles[eligible], axis=1))]
    chosen = order[np.linspace(0, len(order) - 1, count).round().astype(int)]
    selected_signals = signals[:, chosen]
    context = trajectory_context(selected_signals)
    return profiles[chosen], selected_signals, context


def fit_statistical_coordinate(
    profiles: np.ndarray,
    observed_context: np.ndarray,
) -> tuple[Ridge, np.ndarray, dict]:
    """Calibrate the SCDT x0-to-statistics coordinate on stable trajectories."""
    model = Ridge(alpha=1.0e-4).fit(profiles, observed_context)
    fitted = model.predict(profiles)
    rho = [
        float(spearmanr(observed_context[:, index], fitted[:, index]).statistic)
        for index in range(observed_context.shape[1])
    ]
    return model, fitted, {
        "context_fit_rmse": float(np.sqrt(np.mean((observed_context - fitted) ** 2))),
        "context_fit_rho": rho,
    }


def target_profiles(count: int, seed: int) -> np.ndarray:
    """Construct profiles on an equally spaced maximum-risk coordinate."""
    rng = np.random.default_rng(seed)
    maximum = X0_CRITICAL + np.linspace(-0.08, 0.20, count)
    profiles = np.empty((count, N_NODES), dtype=float)
    peak_nodes = np.tile(rng.permutation(N_NODES), int(np.ceil(count / N_NODES)))[:count]
    for index, (peak, peak_node) in enumerate(zip(maximum, peak_nodes)):
        gaps = rng.uniform(0.025, 0.10, size=N_NODES)
        profiles[index] = np.maximum(peak - gaps, -2.28)
        profiles[index, peak_node] = peak
    if np.any(np.isclose(profiles, BASE_X0)):
        raise AssertionError("Every target x0 coordinate must change from base")
    risk = np.max(profiles - X0_CRITICAL, axis=1)
    if not np.allclose(np.diff(risk), np.diff(risk)[0]):
        raise AssertionError("Projected target risk must be equally spaced")
    return profiles


def simulate_switch_cohort(
    targets: np.ndarray,
    seeds: tuple[int, ...],
    warmup_s: float,
    future_s: float,
    coupling: float = 0.7,
    noise_variance: float = 0.0025,
    event_threshold: float = DEFAULT_EVENT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """State-continuous base-to-target Epileptor switches for SCDT validation."""
    targets = np.asarray(targets, dtype=float)
    all_warmups, all_futures, all_profiles, all_seed_ids = [], [], [], []
    stride = 8
    for seed in seeds:
        config = PaperEpileptorConfig(
            n_nodes=N_NODES,
            duration_s=warmup_s + future_s,
            global_coupling=coupling,
            noise_variance=noise_variance,
            seed=seed,
        )
        rng = np.random.default_rng(seed)
        states = initial_state_batch(len(targets), N_NODES, rng)
        connectivity = three_node_connectivity()

        def segment(profiles: np.ndarray, duration_s: float) -> np.ndarray:
            nonlocal states
            steps = int(round(duration_s * config.sampling_hz))
            records = np.empty((steps // stride, len(targets), N_NODES), dtype=float)
            output = 0
            for index in range(steps):
                states = network_heun_step_batch(states, profiles, connectivity, config, rng)
                if index % stride == 0:
                    records[output] = source_proxy(states)
                    output += 1
            return records[:output]

        base = np.full_like(targets, BASE_X0)
        warmup = segment(base, warmup_s)
        future = segment(targets, future_s)
        warm_events = first_threshold_crossing(warmup, event_threshold)
        if np.any(np.isfinite(warm_events)):
            raise RuntimeError(f"Base warmup event at seed {seed}; seed replacement is forbidden")
        all_warmups.append(warmup)
        all_futures.append(future)
        all_profiles.append(targets)
        all_seed_ids.append(np.full(len(targets), seed, dtype=int))
    return (
        np.concatenate(all_warmups, axis=1),
        np.concatenate(all_futures, axis=1),
        np.concatenate(all_profiles, axis=0),
        np.concatenate(all_seed_ids),
    )


def grouped_survival(
    profile_count: int,
    replicate_count: int,
    events: np.ndarray,
) -> np.ndarray:
    event_matrix = np.isfinite(events).reshape(replicate_count, profile_count)
    return 1.0 - event_matrix.mean(axis=0)


def event_metrics(gt_events: np.ndarray, rc_events: np.ndarray) -> dict:
    gt = np.isfinite(gt_events)
    rc = np.isfinite(rc_events)
    hits = int(np.sum(gt & rc))
    misses = int(np.sum(gt & ~rc))
    false_alarms = int(np.sum(~gt & rc))
    correct_nonexits = int(np.sum(~gt & ~rc))
    miss_rate = misses / max(int(np.sum(gt)), 1)
    false_rate = false_alarms / max(int(np.sum(~gt)), 1)
    return {
        "gt_exits": int(np.sum(gt)),
        "rc_exits": int(np.sum(rc)),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_nonexits": correct_nonexits,
        "miss_rate": float(miss_rate),
        "false_alarm_rate": float(false_rate),
        "risk_error": float(0.65 * miss_rate + 0.35 * false_rate),
    }


def normal_morphology_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    stable: np.ndarray,
) -> dict:
    """Compare distributional morphology on matched GT-stable records."""
    gt = np.asarray(ground_truth[:, stable], dtype=float)
    rc = np.asarray(prediction[:, stable], dtype=float)
    if gt.shape[1] == 0:
        return {
            **{name: 10.0 * scale for name, scale in MORPHOLOGY_SCALES.items()},
            "morphology_score": 10.0,
            "morphology_violation": 9.0,
        }

    def features(signals: np.ndarray) -> tuple[np.ndarray, ...]:
        mean = np.mean(signals, axis=0)
        std = np.std(signals, axis=0)
        left = signals[:-1] - np.mean(signals[:-1], axis=0, keepdims=True)
        right = signals[1:] - np.mean(signals[1:], axis=0, keepdims=True)
        denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
        ac1 = np.sum(left * right, axis=0) / np.maximum(denominator, 1e-12)
        spike_rate = np.mean(signals > 1.5, axis=0)
        centered = signals - np.mean(signals, axis=0, keepdims=True)
        psd = np.abs(np.fft.rfft(centered, axis=0)) ** 2
        psd /= np.maximum(np.sum(psd, axis=0, keepdims=True), 1e-12)
        return mean, std, ac1, spike_rate, psd

    gt_mean, gt_std, gt_ac1, gt_spike, gt_psd = features(gt)
    rc_mean, rc_std, rc_ac1, rc_spike, rc_psd = features(rc)
    raw = {
        "normal_mean_mae": float(np.mean(np.abs(gt_mean - rc_mean))),
        "normal_log_std_mae": float(
            np.mean(np.abs(np.log((rc_std + 1e-8) / (gt_std + 1e-8))))
        ),
        "normal_ac1_mae": float(np.mean(np.abs(gt_ac1 - rc_ac1))),
        "normal_spike_rate_mae": float(np.mean(np.abs(gt_spike - rc_spike))),
        "normal_psd_l1": float(np.mean(np.sum(np.abs(gt_psd - rc_psd), axis=0))),
    }
    ratios = np.asarray([raw[name] / scale for name, scale in MORPHOLOGY_SCALES.items()])
    raw["morphology_score"] = float(np.mean(ratios))
    raw["morphology_violation"] = float(np.mean(np.maximum(ratios - 1.0, 0.0)))
    return raw


def evaluate(
    reservoir: MultiOutputStaticRC,
    warmup_raw: np.ndarray,
    future_raw: np.ndarray,
    profiles: np.ndarray,
    coordinate_model: Ridge,
    dynamic_center: np.ndarray,
    dynamic_scale: np.ndarray,
    static_center: np.ndarray,
    static_scale: np.ndarray,
    profile_count: int,
    replicate_count: int,
    seed: int,
    event_threshold: float = DEFAULT_EVENT_THRESHOLD,
    event_blanking_s: float = DEFAULT_EVENT_BLANKING_S,
) -> tuple[dict, np.ndarray, np.ndarray]:
    warmup = normalize(warmup_raw, dynamic_center, dynamic_scale)
    target_raw = coordinate_model.predict(profiles)
    target_static = normalize(target_raw, static_center, static_scale)
    base_raw = coordinate_model.predict(np.full((1, N_NODES), BASE_X0))[0]
    base_static = normalize(base_raw, static_center, static_scale)
    warmup_static = np.repeat(base_static[None, :], len(profiles), axis=0)
    prediction_n = reservoir.rollout_switch_batch(
        warmup,
        warmup_static,
        target_static,
        n_steps=len(future_raw),
        seed=seed,
    )
    prediction = prediction_n * dynamic_scale + dynamic_center
    gt_events = first_threshold_crossing(future_raw, event_threshold, event_blanking_s)
    rc_events = first_threshold_crossing(prediction, event_threshold, event_blanking_s)
    metrics = event_metrics(gt_events, rc_events)
    gt_survival = grouped_survival(profile_count, replicate_count, gt_events)
    rc_survival = grouped_survival(profile_count, replicate_count, rc_events)
    metrics["survival_mae"] = float(np.mean(np.abs(gt_survival - rc_survival)))
    gt_positive = np.isfinite(gt_events)
    score_start = int(round(event_blanking_s * RC_HZ))
    rc_minimum = np.min(prediction[score_start:], axis=(0, 2))
    margins = np.where(
        gt_positive,
        np.maximum(rc_minimum - event_threshold, 0.0),
        np.maximum(event_threshold - rc_minimum, 0.0),
    )
    metrics["threshold_margin"] = float(np.mean(np.minimum(margins / 2.0, 5.0)))
    stable = ~gt_positive
    short = min(len(future_raw), 20 * RC_HZ)
    if np.any(stable):
        metrics["stable_20s_nrmse"] = float(
            np.sqrt(
                np.mean(
                    ((prediction[:short, stable] - future_raw[:short, stable]) / dynamic_scale)
                    ** 2
                )
            )
        )
    else:
        metrics["stable_20s_nrmse"] = 5.0
    metrics.update(normal_morphology_metrics(future_raw, prediction, stable))
    low = reservoir.clip_low * dynamic_scale + dynamic_center
    high = reservoir.clip_high * dynamic_scale + dynamic_center
    at_clip = (prediction <= low + 1e-9) | (prediction >= high - 1e-9)
    metrics["clipping_fraction"] = float(np.mean(at_clip))
    metrics["survival_objective"] = float(
        2.0 * metrics["survival_mae"]
        + 2.0 * metrics["risk_error"]
        + 0.30 * metrics["threshold_margin"]
    )
    metrics["departure_objective"] = float(
        metrics["survival_mae"] + metrics["risk_error"]
    )
    metrics["morphology_objective"] = float(
        metrics["morphology_score"]
        + 50.0 * metrics["clipping_fraction"]
    )
    metrics["stage2_objective"] = float(
        metrics["morphology_score"]
        + 4.0 * metrics["departure_objective"]
        + 50.0 * metrics["clipping_fraction"]
    )
    metrics["score"] = metrics["stage2_objective"]
    metrics["gt_survival"] = gt_survival.tolist()
    metrics["rc_survival"] = rc_survival.tolist()
    return metrics, prediction, rc_events


def expected_improvement(candidates: np.ndarray, gp: GaussianProcessRegressor, best: float) -> np.ndarray:
    mean, std = gp.predict(candidates, return_std=True)
    improvement = best - mean
    z = improvement / np.maximum(std, 1e-12)
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def write_history(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_bo(figdir: Path, records: list[dict], morphology_trials: int) -> None:
    morphology = np.asarray([row["morphology_objective"] for row in records])
    joint = np.asarray([row["stage2_objective"] for row in records])
    trial = np.arange(1, len(records) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
    for ax, score, label in (
        (axes[0], morphology, "Morphology objective"),
        (axes[1], joint, "Morphology-constrained survival objective"),
    ):
        ax.plot(trial, score, "o-", alpha=0.75, label="trial")
        ax.plot(trial, np.minimum.accumulate(score), lw=2, color="#176b3a", label="best")
        ax.axvline(morphology_trials + 0.5, color="#8B2E1E", ls=":")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_title("Three-node SCDT two-stage Bayesian optimization")
    axes[-1].set_xlabel("SCDT BO trial")
    fig.savefig(figdir / "bo_trace.png", dpi=180)
    plt.close(fig)


def plot_survival(
    figdir: Path,
    targets: np.ndarray,
    metrics: dict,
    analysis_label: str = "Locked",
) -> None:
    x0 = np.max(targets, axis=1)
    order = np.argsort(x0)
    gt = np.asarray(metrics["gt_survival"])[order]
    rc = np.asarray(metrics["rc_survival"])[order]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x0[order], gt, "o-", color="black", label="GT")
    ax.plot(x0[order], rc, "s--", color="#176b3a", label="SCDT-RC")
    ax.axvline(
        X0_CRITICAL,
        color="#777777",
        ls=":",
        label=f"single-node paper reference ({X0_CRITICAL:.2f})",
    )
    # The literature reference is not the empirical transition point of this
    # coupled, noisy, finite-horizon protocol.  Show the post-hoc GT 50% point
    # separately whenever the sampled survival curve brackets it.
    centered = gt - 0.5
    crossings = np.flatnonzero(centered[:-1] * centered[1:] <= 0.0)
    crossings = [index for index in crossings if gt[index] != gt[index + 1]]
    if crossings:
        index = crossings[0]
        empirical_x50 = x0[order][index] + (
            (0.5 - gt[index])
            * (x0[order][index + 1] - x0[order][index])
            / (gt[index + 1] - gt[index])
        )
        ax.axvline(
            empirical_x50,
            color="#a33a2b",
            ls="--",
            label=f"GT finite-horizon 50% point ({empirical_x50:.3f})",
        )
    rc_centered = rc - 0.5
    rc_crossings = np.flatnonzero(rc_centered[:-1] * rc_centered[1:] <= 0.0)
    rc_crossings = [index for index in rc_crossings if rc[index] != rc[index + 1]]
    if rc_crossings:
        index = rc_crossings[0]
        rc_x50 = x0[order][index] + (
            (0.5 - rc[index])
            * (x0[order][index + 1] - x0[order][index])
            / (rc[index + 1] - rc[index])
        )
        ax.axvline(
            rc_x50,
            color="#176b3a",
            ls="-.",
            label=f"RC finite-horizon 50% point ({rc_x50:.3f})",
        )
    ax.set_xticks(x0[order])
    ax.set_xticklabels([f"{value:.3f}" for value in x0[order]], rotation=45, ha="right", fontsize=8)
    ax.set(xlabel="x0", ylabel="120 s survival fraction", ylim=(-0.05, 1.05), title=f"{analysis_label} survival surface projection | MAE={metrics['survival_mae']:.3f}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "survival_vs_boundary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rollout(
    figdir: Path,
    warmup: np.ndarray,
    future: np.ndarray,
    prediction: np.ndarray,
    profiles: np.ndarray,
    event_threshold: float,
    event_blanking_s: float,
) -> None:
    gt_events = first_threshold_crossing(future, event_threshold, event_blanking_s)
    candidates = np.flatnonzero(np.isfinite(gt_events))
    index = int(candidates[len(candidates) // 2]) if len(candidates) else 0
    time = np.arange(len(warmup) + len(future)) / RC_HZ
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for node, ax in enumerate(axes):
        ax.plot(time[: len(warmup)], warmup[:, index, node], color="gray", lw=0.8, label="base warmup")
        ax.plot(time[len(warmup) :], future[:, index, node], color="black", lw=0.8, label="GT")
        ax.plot(time[len(warmup) :], prediction[:, index, node], color="#176b3a", lw=0.9, label="SCDT-RC")
        ax.axhline(event_threshold, color="#a33a2b", ls="--", lw=0.9)
        ax.axvline(len(warmup) / RC_HZ, color="black", ls=":", lw=0.9)
        ax.axvspan(
            len(warmup) / RC_HZ,
            len(warmup) / RC_HZ + event_blanking_s,
            color="#d9d9d9",
            alpha=0.35,
        )
        ax.set(ylabel=f"Node {node + 1}")
        ax.grid(alpha=0.15)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Display time (s)")
    fig.suptitle(f"SCDT switch rollout | target x0={np.round(profiles[index], 3)}")
    fig.savefig(figdir / "rollout_example.png", dpi=180)
    plt.close(fig)


def plot_gt_rc_grid(
    figdir: Path,
    warmup: np.ndarray,
    future: np.ndarray,
    prediction: np.ndarray,
    profiles: np.ndarray,
    event_threshold: float,
    event_blanking_s: float,
) -> None:
    """Compare matched normal and event trajectories as a GT/RC 2x2 grid."""
    gt_events = first_threshold_crossing(future, event_threshold, event_blanking_s)
    rc_events = first_threshold_crossing(prediction, event_threshold, event_blanking_s)
    risk = np.max(profiles - X0_CRITICAL, axis=1)

    normal_candidates = np.flatnonzero(~np.isfinite(gt_events) & ~np.isfinite(rc_events))
    hit_candidates = np.flatnonzero(np.isfinite(gt_events) & np.isfinite(rc_events))
    event_candidates = hit_candidates
    if len(event_candidates) == 0:
        event_candidates = np.flatnonzero(np.isfinite(gt_events))
    if len(normal_candidates) == 0 or len(event_candidates) == 0:
        return
    normal_index = int(normal_candidates[np.argmax(risk[normal_candidates])])

    start = int(round(event_blanking_s * RC_HZ))
    below = np.any(prediction[start:] <= event_threshold, axis=2)

    def longest_run(values: np.ndarray) -> int:
        changes = np.diff(np.r_[False, values, False].astype(int))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        return int(np.max(ends - starts, initial=0))

    run_lengths = np.asarray([longest_run(below[:, index]) for index in event_candidates])
    if np.max(run_lengths, initial=0) > 0:
        event_index = int(event_candidates[np.argmax(run_lengths)])
    else:
        event_index = int(event_candidates[np.argmax(risk[event_candidates])])

    display_time = np.arange(len(warmup) + len(future)) / RC_HZ
    switch_time = len(warmup) / RC_HZ
    colors = ("#276FBF", "#D1495B", "#2E8B57")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    rows = ((normal_index, "Normal"), (event_index, "Departure / seizure"))
    columns = ((future, "GT"), (prediction, "SCDT-RC"))
    for row, (index, row_name) in enumerate(rows):
        for column, (post_switch, model_name) in enumerate(columns):
            ax = axes[row, column]
            for node, color in enumerate(colors):
                series = np.concatenate((warmup[:, index, node], post_switch[:, index, node]))
                ax.plot(display_time, series, color=color, lw=0.85, label=f"Node {node + 1}")
            ax.axhline(event_threshold, color="#8B2E1E", ls="--", lw=1.0)
            ax.axvline(switch_time, color="black", ls=":", lw=1.0)
            ax.axvspan(
                switch_time,
                switch_time + event_blanking_s,
                color="#d9d9d9",
                alpha=0.35,
            )
            event_time = gt_events[index] if column == 0 else rc_events[index]
            event_text = (
                f"first <= {event_threshold:g}: +{event_time:.1f} s"
                if np.isfinite(event_time)
                else f"no point <= {event_threshold:g}"
            )
            ax.text(
                0.985,
                0.04,
                event_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
            )
            ax.set_title(f"{row_name} — {model_name}")
            ax.grid(alpha=0.15)
        axes[row, 0].set_ylabel(
            f"Signal | x0={np.array2string(profiles[index], precision=3)}"
        )
    axes[0, 0].legend(loc="upper right", ncol=3)
    axes[1, 0].set_xlabel("Display time (s)")
    axes[1, 1].set_xlabel("Display time (s)")
    fig.suptitle(
        "Matched GT and SCDT-RC trajectories | dashed: departure threshold; shaded: 5 s settling",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(figdir / "gt_rc_2x2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morphology-trials", type=int, default=40)
    parser.add_argument("--survival-trials", type=int, default=60)
    parser.add_argument("--train-profiles", type=int, default=32)
    parser.add_argument("--validation-profiles", type=int, default=12)
    parser.add_argument("--validation-seeds", type=int, default=4)
    parser.add_argument("--test-profiles", type=int, default=16)
    parser.add_argument("--test-seeds", type=int, default=8)
    parser.add_argument("--warmup-s", type=float, default=40.0)
    parser.add_argument("--future-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=676000)
    parser.add_argument("--event-threshold", type=float, default=DEFAULT_EVENT_THRESHOLD)
    parser.add_argument("--event-blanking-s", type=float, default=DEFAULT_EVENT_BLANKING_S)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "three_node_scdt_sp25_lr07_bo100")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "figures" / "RC" / "three_node_scdt_sp25_lr07_bo100")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    train_profiles, train_raw, train_context = make_stable_training(
        args.train_profiles, args.seed, args.warmup_s + args.future_s, args.event_threshold
    )
    coordinate_model, fitted_context, coordinate_metrics = fit_statistical_coordinate(
        train_profiles, train_context
    )
    dynamic_center, dynamic_scale = fit_normalizer(train_raw)
    static_center, static_scale = fit_normalizer(fitted_context)
    train_n = normalize(train_raw, dynamic_center, dynamic_scale)
    train_static_n = normalize(fitted_context, static_center, static_scale)

    validation_targets = target_profiles(args.validation_profiles, args.seed + 1000)
    validation_seed_values = tuple(args.seed + 2000 + index for index in range(args.validation_seeds))
    validation = simulate_switch_cohort(
        validation_targets,
        validation_seed_values,
        args.warmup_s,
        args.future_s,
        event_threshold=args.event_threshold,
    )
    validation_warmup, validation_future, validation_profiles, _ = validation

    rng = np.random.default_rng(args.seed + 3000)
    units: list[np.ndarray] = []
    records: list[dict] = []
    total_trials = args.morphology_trials + args.survival_trials
    initial = min(20, max(12, args.morphology_trials // 2))
    for trial in range(total_trials):
        stage = "morphology" if trial < args.morphology_trials else "survival"
        objective_key = (
            "morphology_objective" if stage == "morphology" else "stage2_objective"
        )
        if trial < initial:
            unit = rng.random(7)
            unit[6] = 0.25 if trial % 2 == 0 else 0.75
        else:
            x = np.vstack(units)
            y = np.asarray([row[objective_key] for row in records])
            gp = GaussianProcessRegressor(
                kernel=Matern(np.ones(7), nu=2.5) + WhiteKernel(1e-5),
                normalize_y=True,
                random_state=args.seed,
                n_restarts_optimizer=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(x, y)
            candidates = rng.random((4096, 7))
            candidates[:1024] = np.clip(
                x[int(np.argmin(y))] + rng.normal(0.0, 0.10, (1024, 7)), 0.0, 1.0
            )
            unit = candidates[int(np.argmax(expected_improvement(candidates, gp, float(np.min(y)))))]
        params, mode = parameter_from_unit(unit, args.seed + 4000)
        reservoir = build_reservoir(params, mode)
        reservoir.fit(
            [train_n[:, index] for index in range(args.train_profiles)],
            [train_static_n[index] for index in range(args.train_profiles)],
            warmup=5 * RC_HZ,
        )
        result, _, _ = evaluate(
            reservoir,
            validation_warmup,
            validation_future,
            validation_profiles,
            coordinate_model,
            dynamic_center,
            dynamic_scale,
            static_center,
            static_scale,
            args.validation_profiles,
            args.validation_seeds,
            args.seed + 5000,
            args.event_threshold,
            args.event_blanking_s,
        )
        row = {
            "trial": trial + 1,
            "stage": stage,
            **{f"unit_{index}": float(value) for index, value in enumerate(unit)},
            "mode": mode,
            **{key: value for key, value in result.items() if key not in ("gt_survival", "rc_survival")},
            "score": result[objective_key],
            **asdict(params),
        }
        units.append(unit)
        records.append(row)
        write_history(args.out_dir / "bo_history.csv", records)
        print(json.dumps(row), flush=True)

    winner_index = int(np.argmin([row["stage2_objective"] for row in records]))
    winner_params, winner_mode = parameter_from_unit(units[winner_index], args.seed + 4000)
    winner = build_reservoir(winner_params, winner_mode)
    winner.fit(
        [train_n[:, index] for index in range(args.train_profiles)],
        [train_static_n[index] for index in range(args.train_profiles)],
        warmup=5 * RC_HZ,
    )

    test_targets = target_profiles(args.test_profiles, args.seed + 6000)
    test_seed_values = tuple(args.seed + 7000 + index for index in range(args.test_seeds))
    test_warmup, test_future, test_profiles_repeated, test_seed_ids = simulate_switch_cohort(
        test_targets,
        test_seed_values,
        args.warmup_s,
        args.future_s,
        event_threshold=args.event_threshold,
    )
    test_result, test_prediction, test_rc_events = evaluate(
        winner,
        test_warmup,
        test_future,
        test_profiles_repeated,
        coordinate_model,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        args.test_profiles,
        args.test_seeds,
        args.seed + 8000,
        args.event_threshold,
        args.event_blanking_s,
    )
    validation_result, validation_prediction, validation_rc_events = evaluate(
        winner,
        validation_warmup,
        validation_future,
        validation_profiles,
        coordinate_model,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        args.validation_profiles,
        args.validation_seeds,
        args.seed + 5000,
        args.event_threshold,
        args.event_blanking_s,
    )
    summary = {
        "status": "complete",
        "selected_trial": winner_index + 1,
        "selected_mode": winner_mode,
        "selected_parameters": asdict(winner_params),
        "bo_stages": {
            "morphology_trials": args.morphology_trials,
            "survival_trials": args.survival_trials,
        },
        "morphology_scales": MORPHOLOGY_SCALES,
        "coordinate_calibration": coordinate_metrics,
        "validation": validation_result,
        "locked_test": test_result,
    }
    save_json(args.out_dir / "summary.json", summary)
    save_json(
        args.out_dir / "protocol.json",
        {
            "date": "2026-09-10",
            "architecture": "three-node SCDT: calibrated x0-to-statistics coordinate, base-to-target static switch, autonomous reservoir rollout",
            "train_profiles": args.train_profiles,
            "validation_profiles_x_seeds": [args.validation_profiles, args.validation_seeds],
            "locked_test_profiles_x_seeds": [args.test_profiles, args.test_seeds],
            "train_x0_cube": [-2.50, X0_CRITICAL - 0.04],
            "target_x0_cube": [-2.28, -1.90],
            "target_design": "equally spaced max_i(x0_i - x0 critical) from -0.08 to 0.20; peak node rotates and all coordinates vary",
            "base_x0": BASE_X0,
            "critical_x0": X0_CRITICAL,
            "event_definition": f"first sample at or below {args.event_threshold} in any node after a {args.event_blanking_s} s post-switch settling interval",
            "event_threshold": args.event_threshold,
            "event_blanking_s": args.event_blanking_s,
            "all_three_target_coordinates_change": True,
            "static_coordinate": list(CONTEXT_NAMES),
            "x0_is_reservoir_input": False,
            "reservoir_nodes": 1500,
            "readout_states": 1500,
            "bo_trials": total_trials,
            "bo_stages": [args.morphology_trials, args.survival_trials],
            "bo_selection": "stage 1 minimizes normal mean/std/ac1/spike-rate/PSD mismatch; stage 2 minimizes morphology score + 4*(survival MAE + event risk error); clipping is a validity penalty",
            "spectral_radius_range": [0.30, 2.50],
            "leakage_rate_range": [0.01, 0.70],
            "selection_endpoints": ["normal trajectory morphology", "normal departure"],
            "morphology_scales": MORPHOLOGY_SCALES,
            "locked_test_used_for_selection": False,
            "seeds": {
                "training": args.seed,
                "validation_profiles": args.seed + 1000,
                "validation_ode": list(validation_seed_values),
                "locked_profiles": args.seed + 6000,
                "locked_ode": list(test_seed_values),
            },
        },
    )
    np.savez_compressed(
        args.out_dir / "training.npz",
        profiles=train_profiles,
        signals=train_raw,
        observed_context=train_context,
        fitted_context=fitted_context,
    )
    np.savez_compressed(
        args.out_dir / "validation.npz",
        targets=validation_targets,
        profiles=validation_profiles,
        warmup=validation_warmup,
        future=validation_future,
        prediction=validation_prediction,
        rc_events=validation_rc_events,
    )
    np.savez_compressed(
        args.out_dir / "locked_test.npz",
        targets=test_targets,
        profiles=test_profiles_repeated,
        seeds=test_seed_ids,
        warmup=test_warmup,
        future=test_future,
        prediction=test_prediction,
        rc_events=test_rc_events,
    )
    np.savez_compressed(
        args.out_dir / "winner_model.npz",
        a_data=winner.a.data,
        a_indices=winner.a.indices,
        a_indptr=winner.a.indptr,
        a_shape=winner.a.shape,
        w_in=winner.w_in,
        w_static=winner.w_static,
        w_out=winner.w_out,
        residuals=winner.residuals,
        readout_indices=winner.readout_indices,
        dynamic_center=dynamic_center,
        dynamic_scale=dynamic_scale,
        static_center=static_center,
        static_scale=static_scale,
        clip_low=winner.clip_low,
        clip_high=winner.clip_high,
        coordinate_coef=coordinate_model.coef_,
        coordinate_intercept=coordinate_model.intercept_,
    )
    plot_bo(args.fig_dir, records, args.morphology_trials)
    plot_survival(args.fig_dir, test_targets, test_result)
    plot_rollout(
        args.fig_dir,
        test_warmup,
        test_future,
        test_prediction,
        test_profiles_repeated,
        args.event_threshold,
        args.event_blanking_s,
    )
    plot_gt_rc_grid(
        args.fig_dir,
        test_warmup,
        test_future,
        test_prediction,
        test_profiles_repeated,
        args.event_threshold,
        args.event_blanking_s,
    )
    (args.out_dir / "RESULTS.md").write_text(
        "# Three-node SCDT validation\n\n"
        "This experiment follows the existing SCDT flow: stable precritical trajectories calibrate an x0-to-statistics coordinate; the reservoir learns the conditioned dynamics; and a base-to-target coordinate switch is evaluated by autonomous survival.\n\n"
        f"- Selected BO trial / mode: {winner_index + 1} / {winner_mode}\n"
        f"- Validation survival MAE: {validation_result['survival_mae']:.4f}\n"
        f"- Validation hits / misses / false alarms: {validation_result['hits']} / {validation_result['misses']} / {validation_result['false_alarms']}\n"
        f"- Locked survival MAE: {test_result['survival_mae']:.4f}\n"
        f"- Locked GT / RC exits: {test_result['gt_exits']} / {test_result['rc_exits']}\n"
        f"- Locked hits / misses / false alarms / correct non-exits: {test_result['hits']} / {test_result['misses']} / {test_result['false_alarms']} / {test_result['correct_nonexits']}\n"
        f"- Locked stable 20 s NRMSE: {test_result['stable_20s_nrmse']:.4f}\n"
        f"- Locked morphology score / violation: {test_result['morphology_score']:.4f} / {test_result['morphology_violation']:.4f}\n"
        f"- Locked normal mean / log-std / AC1 MAE: {test_result['normal_mean_mae']:.4f} / {test_result['normal_log_std_mae']:.4f} / {test_result['normal_ac1_mae']:.4f}\n"
        f"- Locked normal spike-rate MAE / PSD L1: {test_result['normal_spike_rate_mae']:.4f} / {test_result['normal_psd_l1']:.4f}\n"
        f"- Locked clipping fraction: {test_result['clipping_fraction']:.6f}\n\n"
        f"- BO stages: {args.morphology_trials} morphology trials + {args.survival_trials} morphology-constrained survival trials\n"
        "- Spectral radius search: 0.30 to 2.50 (linear)\n"
        "- Leakage-rate search: 0.01 to 0.70 (log)\n"
        f"- Event definition: first point <= {args.event_threshold:.3f} in any node after a {args.event_blanking_s:.1f} s post-switch settling interval\n"
        f"- Validation risk grid: -0.08 to 0.20 in {args.validation_profiles} equally spaced levels (step {0.28 / (args.validation_profiles - 1):.7f})\n"
        f"- Locked risk grid: -0.08 to 0.20 in {args.test_profiles} equally spaced levels (step {0.28 / (args.test_profiles - 1):.7f})\n\n"
        "All three target x0 coordinates change together. x0 defines the synthetic intervention and statistical-coordinate mapping but is never a reservoir input. The locked profiles and ODE seeds are generated only after BO selection.\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
