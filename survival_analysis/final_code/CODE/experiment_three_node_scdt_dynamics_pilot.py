#!/usr/bin/env python3
"""Pilot readout/training changes for normal spikes and sustained seizure drops."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from experiment_three_node_parameter_space_rc import fit_normalizer
from experiment_three_node_scdt import (
    DEFAULT_EVENT_BLANKING_S,
    DEFAULT_EVENT_THRESHOLD,
    RC_HZ,
    build_reservoir,
    evaluate,
    first_threshold_crossing,
    fit_statistical_coordinate,
    normalize,
    simulate_switch_cohort,
)
from paper_single_node_scrc import RCParams


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "three_node_scdt_sp25_lr07_bo100"
OUT = ROOT / "experiments" / "three_node_scdt_dynamics_pilot"
FIG = ROOT / "figures" / "RC" / "three_node_scdt_dynamics_pilot"


@dataclass(frozen=True)
class Variant:
    name: str
    mixed_training: bool
    retain_spikes: bool
    spike_weight: float
    low_weight: float


VARIANTS = (
    Variant("stable_baseline", False, False, 1.0, 1.0),
    Variant("stable_spike_weighted", False, True, 6.0, 1.0),
    Variant("mixed_uniform", True, False, 1.0, 1.0),
    Variant("mixed_rare_2x", True, True, 2.0, 2.0),
    Variant("mixed_spike_low_3x", True, True, 6.0, 3.0),
    Variant("mixed_spike_low_8x", True, True, 6.0, 8.0),
)


def event_training_profiles() -> np.ndarray:
    """Held-out-seed postcritical profiles used only to teach the low attractor."""
    maximum = np.asarray([-1.99, -1.96, -1.93, -1.90])
    rng = np.random.default_rng(681050)
    profiles = np.empty((len(maximum), 3), dtype=float)
    peak_nodes = np.asarray([0, 1, 2, 0])
    for index, (peak, peak_node) in enumerate(zip(maximum, peak_nodes)):
        profiles[index] = peak - rng.uniform(0.025, 0.10, size=3)
        profiles[index, peak_node] = peak
    return profiles


def collect_weighted_readout(
    reservoir,
    normalized_sequences: np.ndarray,
    raw_sequences: np.ndarray,
    statics: np.ndarray,
    warmup: int,
    retain_spikes: bool,
    spike_weight: float,
    low_weight: float,
) -> dict:
    """Fit every fourth sample plus rare spike/low-state samples from skipped phases."""
    all_values = normalized_sequences.reshape(-1, normalized_sequences.shape[-1])
    span = np.ptp(all_values, axis=0)
    reservoir.clip_low = np.min(all_values, axis=0) - 1.0 * span
    reservoir.clip_high = np.max(all_values, axis=0) + 1.0 * span
    batch = normalized_sequences.shape[1]
    state = np.zeros((reservoir.params.n_reservoir, batch))
    static_batch = statics.T
    features, targets, weights = [], [], []
    regular_rows = spike_rows = low_rows = 0
    for index in range(len(normalized_sequences) - 1):
        current = normalized_sequences[index]
        state = reservoir.update(state, current.T, static_batch)
        if index < warmup:
            continue
        raw_current = raw_sequences[index]
        raw_target = raw_sequences[index + 1]
        spike = np.any((raw_current > 1.5) | (raw_target > 1.5), axis=1)
        low = np.any(
            (raw_current <= DEFAULT_EVENT_THRESHOLD)
            | (raw_target <= DEFAULT_EVENT_THRESHOLD),
            axis=1,
        )
        regular = (index - warmup) % 4 == 0
        keep = np.ones(batch, dtype=bool) if regular else (spike | low) & retain_spikes
        if not np.any(keep):
            continue
        x = reservoir.augment(state[:, keep], current[keep].T).T.astype(np.float32)
        y = normalized_sequences[index + 1, keep].astype(np.float32)
        w = np.ones(np.sum(keep), dtype=np.float32)
        w[spike[keep]] *= spike_weight
        w[low[keep]] *= low_weight
        features.append(x)
        targets.append(y)
        weights.append(w)
        regular_rows += int(regular) * int(np.sum(keep))
        spike_rows += int(np.sum(spike[keep]))
        low_rows += int(np.sum(low[keep]))
    x = np.concatenate(features)
    y = np.concatenate(targets)
    w = np.concatenate(weights)
    model = Ridge(
        alpha=reservoir.params.ridge,
        fit_intercept=False,
        solver="cholesky",
    ).fit(x, y, sample_weight=w)
    reservoir.w_out = np.asarray(model.coef_, dtype=float)
    residuals = y - model.predict(x)
    lower = np.quantile(residuals, 0.0025, axis=0)
    upper = np.quantile(residuals, 0.9975, axis=0)
    reservoir.residuals = np.clip(residuals, lower, upper)
    return {
        "fit_rows": int(len(x)),
        "regular_rows": regular_rows,
        "spike_rows": spike_rows,
        "low_rows": low_rows,
    }


def longest_low_run(signals: np.ndarray) -> np.ndarray:
    low = np.any(signals <= DEFAULT_EVENT_THRESHOLD, axis=2)
    result = np.zeros(low.shape[1], dtype=float)
    for record in range(low.shape[1]):
        changes = np.diff(np.r_[False, low[:, record], False].astype(int))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        result[record] = np.max(ends - starts, initial=0) / RC_HZ
    return result


def shape_metrics(gt: np.ndarray, rc: np.ndarray) -> dict:
    start = int(round(DEFAULT_EVENT_BLANKING_S * RC_HZ))
    gt_post = gt[start:]
    rc_post = rc[start:]
    gt_events = np.isfinite(
        first_threshold_crossing(gt, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S)
    )
    gt_q05 = np.quantile(gt_post, 0.05, axis=(0, 2))
    rc_q05 = np.quantile(rc_post, 0.05, axis=(0, 2))
    gt_occupancy = np.mean(np.any(gt_post <= DEFAULT_EVENT_THRESHOLD, axis=2), axis=0)
    rc_occupancy = np.mean(np.any(rc_post <= DEFAULT_EVENT_THRESHOLD, axis=2), axis=0)
    gt_run = longest_low_run(gt_post)
    rc_run = longest_low_run(rc_post)
    gt_sustained = gt_run >= 5.0
    rc_sustained = rc_run >= 5.0
    return {
        "event_q05_mae": float(np.mean(np.abs(gt_q05[gt_events] - rc_q05[gt_events]))),
        "event_occupancy_mae": float(
            np.mean(np.abs(gt_occupancy[gt_events] - rc_occupancy[gt_events]))
        ),
        "event_run_mae_s": float(np.mean(np.abs(gt_run[gt_events] - rc_run[gt_events]))),
        "rc_event_q05_median": float(np.median(rc_q05[gt_events])),
        "rc_event_occupancy_median": float(np.median(rc_occupancy[gt_events])),
        "rc_event_run_median_s": float(np.median(rc_run[gt_events])),
        "sustained_hits": int(np.sum(gt_sustained & rc_sustained)),
        "sustained_misses": int(np.sum(gt_sustained & ~rc_sustained)),
        "sustained_false_alarms": int(np.sum(~gt_sustained & rc_sustained)),
    }


def plot_postcritical_comparison(
    futures: np.ndarray,
    predictions: dict[str, np.ndarray],
    profiles: np.ndarray,
) -> None:
    gt_events = first_threshold_crossing(
        futures, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S
    )
    candidate = np.flatnonzero(np.isfinite(gt_events))
    record = int(candidate[len(candidate) // 2])
    time = np.arange(len(futures)) / RC_HZ
    colors = ("#276FBF", "#D1495B", "#2E8B57")
    rows = [("GT", futures), *[(name, values) for name, values in predictions.items()]]
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 2.25 * len(rows)), sharex=True, sharey=True)
    for ax, (label, values) in zip(axes, rows):
        for node, color in enumerate(colors):
            ax.plot(time, values[:, record, node], color=color, lw=0.8, label=f"Node {node+1}")
        ax.axhline(DEFAULT_EVENT_THRESHOLD, color="#8B2E1E", ls="--", lw=1.0)
        ax.axvspan(0, DEFAULT_EVENT_BLANKING_S, color="#d9d9d9", alpha=0.35)
        ax.set_ylabel(label)
        ax.grid(alpha=0.15)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Time after x0 switch (s)")
    fig.suptitle(
        f"Postcritical trajectory training pilot | x0={np.array2string(profiles[record], precision=3)}"
    )
    fig.tight_layout()
    fig.savefig(FIG / "postcritical_variant_trajectories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((SOURCE / "summary.json").read_text())
    params = RCParams(**source_summary["selected_parameters"])
    with np.load(SOURCE / "training.npz") as data:
        train_profiles = data["profiles"]
        train_raw_full = data["signals"]
        train_context = data["observed_context"]
    coordinate, fitted_context, _ = fit_statistical_coordinate(train_profiles, train_context)
    dynamic_center, dynamic_scale = fit_normalizer(train_raw_full)
    static_center, static_scale = fit_normalizer(fitted_context)
    stable_raw = train_raw_full[-120 * RC_HZ :]
    stable_n = normalize(stable_raw, dynamic_center, dynamic_scale)
    stable_static = normalize(fitted_context, static_center, static_scale)

    event_targets = event_training_profiles()
    event_seeds = (681100, 681101, 681102)
    _, event_raw, event_profiles, _ = simulate_switch_cohort(
        event_targets,
        event_seeds,
        warmup_s=40.0,
        future_s=120.0,
        event_threshold=DEFAULT_EVENT_THRESHOLD,
    )
    event_count = int(
        np.sum(
            np.isfinite(
                first_threshold_crossing(
                    event_raw, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S
                )
            )
        )
    )
    event_n = normalize(event_raw, dynamic_center, dynamic_scale)
    event_static_raw = coordinate.predict(event_profiles)
    event_static = normalize(event_static_raw, static_center, static_scale)

    with np.load(SOURCE / "validation.npz") as data:
        validation = {name: data[name] for name in data.files}
    profile_count = len(validation["targets"])
    replicate_count = validation["profiles"].shape[0] // profile_count
    rows = []
    predictions = {}
    for variant in VARIANTS:
        if variant.mixed_training:
            raw = np.concatenate((stable_raw, event_raw), axis=1)
            normalized = np.concatenate((stable_n, event_n), axis=1)
            statics = np.concatenate((stable_static, event_static), axis=0)
        else:
            raw, normalized, statics = stable_raw, stable_n, stable_static
        reservoir = build_reservoir(params, source_summary["selected_mode"])
        fit_info = collect_weighted_readout(
            reservoir,
            normalized,
            raw,
            statics,
            warmup=5 * RC_HZ,
            retain_spikes=variant.retain_spikes,
            spike_weight=variant.spike_weight,
            low_weight=variant.low_weight,
        )
        metrics, prediction, _ = evaluate(
            reservoir,
            validation["warmup"],
            validation["future"],
            validation["profiles"],
            coordinate,
            dynamic_center,
            dynamic_scale,
            static_center,
            static_scale,
            profile_count,
            replicate_count,
            seed=681500,
            event_threshold=DEFAULT_EVENT_THRESHOLD,
            event_blanking_s=DEFAULT_EVENT_BLANKING_S,
        )
        shape = shape_metrics(validation["future"], prediction)
        row = {
            "variant": variant.name,
            "mixed_training": variant.mixed_training,
            "spike_weight": variant.spike_weight,
            "low_weight": variant.low_weight,
            **fit_info,
            **{key: value for key, value in metrics.items() if key not in ("gt_survival", "rc_survival")},
            **shape,
        }
        rows.append(row)
        predictions[variant.name] = prediction
        print(json.dumps(row), flush=True)
    with (OUT / "pilot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "event_training_profiles_x_seeds": [len(event_targets), len(event_seeds)],
        "event_training_departures": event_count,
        "normalizer_and_coordinate_fit": "stable training only",
        "event_static_input": "stable-coordinate extrapolation; no future-statistic leakage",
        "variants": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_postcritical_comparison(
        validation["future"], predictions, validation["profiles"]
    )


if __name__ == "__main__":
    main()
