#!/usr/bin/env python3
"""Recalibrate the three-node GT boundary before SCDT training and BO."""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from experiment_three_node_bifurcation_rc import N_NODES, latin_hypercube, save_json, simulate_paths
from experiment_three_node_parameter_space_rc import fit_normalizer, trajectory_context
from experiment_three_node_scdt import (
    BASE_X0,
    DEFAULT_EVENT_BLANKING_S,
    DEFAULT_EVENT_THRESHOLD,
    RC_HZ,
    build_reservoir,
    evaluate,
    first_threshold_crossing,
    fit_statistical_coordinate,
    grouped_survival,
    normalize,
    parameter_from_unit,
    plot_gt_rc_grid,
    plot_survival,
    simulate_switch_cohort,
)
from experiment_three_node_scdt_dynamics_bo import objective
from experiment_three_node_scdt_dynamics_pilot import collect_weighted_readout, shape_metrics
from paper_epileptor_core import X0_CRITICAL


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "three_node_scdt_empirical_boundary_bo100"
FIG = ROOT / "figures" / "RC" / "three_node_scdt_empirical_boundary_bo100"
TRIALS = 100
TRAIN_RECORDS = 48
VALIDATION_LEVELS = 12
VALIDATION_SEEDS = 4
TEST_LEVELS = 16
TEST_SEEDS = 8
WARMUP_S = 40.0
FUTURE_S = 120.0
SEED = 730000

# Prior winners are legitimate BO starting points: their locked outcomes were
# obtained before this empirical-boundary dataset existed.  Including them
# prevents a broad seven-dimensional random design from missing the narrow
# low-leakage basin already identified by the preceding experiment.
PRIOR_UNITS = (
    np.asarray([0.08878345, 0.0, 0.21906929, 0.0, 0.83393283, 0.96314822, 0.25]),
    np.asarray([0.08974290, 0.84859099, 0.53031490, 0.13735338, 0.42178364, 0.36356545, 0.75]),
    np.asarray([0.93762854, 0.43542770, 0.20742211, 0.43964033, 0.86049963, 0.82342096, 0.25]),
)


def profiles_from_maximum(maximum: np.ndarray, seed: int, gap=(0.025, 0.10)) -> np.ndarray:
    maximum = np.asarray(maximum, dtype=float)
    rng = np.random.default_rng(seed)
    profiles = np.empty((len(maximum), N_NODES), dtype=float)
    peak_nodes = np.tile(rng.permutation(N_NODES), int(np.ceil(len(maximum) / N_NODES)))[: len(maximum)]
    for index, (peak, node) in enumerate(zip(maximum, peak_nodes)):
        profiles[index] = peak - rng.uniform(gap[0], gap[1], N_NODES)
        profiles[index, node] = peak
    return profiles


def empirical_x50(x0: np.ndarray, survival: np.ndarray) -> float:
    order = np.argsort(x0)
    x = np.asarray(x0)[order]
    y = np.asarray(survival)[order]
    exact = np.flatnonzero(np.isclose(y, 0.5))
    if len(exact):
        return float(x[exact[0]])
    crossings = np.flatnonzero((y[:-1] - 0.5) * (y[1:] - 0.5) < 0.0)
    if not len(crossings):
        raise RuntimeError("Calibration grid does not bracket 50% GT survival")
    index = int(crossings[0])
    return float(x[index] + (0.5 - y[index]) * (x[index + 1] - x[index]) / (y[index + 1] - y[index]))


def calibrate_boundary():
    maxima = np.linspace(-2.10, -1.98, 13)
    targets = profiles_from_maximum(maxima, SEED + 10)
    seeds = tuple(SEED + 100 + index for index in range(8))
    warmup, future, profiles, seed_ids = simulate_switch_cohort(
        targets,
        seeds,
        WARMUP_S,
        FUTURE_S,
        event_threshold=DEFAULT_EVENT_THRESHOLD,
    )
    events = first_threshold_crossing(future, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S)
    survival = grouped_survival(len(targets), len(seeds), events)
    x50 = empirical_x50(maxima, survival)
    return maxima, targets, survival, x50, warmup, future, profiles, seed_ids, events


def make_stable_training(upper: float):
    candidate_count = TRAIN_RECORDS * 3
    # Latin-hypercube coverage is retained, while one third of the candidates
    # deliberately cover the previously missing near-critical stable band.
    broad = latin_hypercube(
        TRAIN_RECORDS * 2,
        -2.50,
        upper,
        np.random.default_rng(SEED + 300),
    )
    near_max = np.linspace(upper - 0.08, upper, TRAIN_RECORDS)
    near = profiles_from_maximum(near_max, SEED + 301, gap=(0.02, 0.08))
    candidates = np.concatenate((broad, near), axis=0)
    if len(candidates) != candidate_count:
        raise AssertionError("unexpected training candidate count")
    signals, _ = simulate_paths(
        candidates,
        candidates,
        duration_s=WARMUP_S + FUTURE_S,
        hold_s=0.0,
        seed=SEED + 302,
    )
    events = first_threshold_crossing(signals, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S)
    eligible = np.flatnonzero(~np.isfinite(events))
    if len(eligible) < TRAIN_RECORDS:
        raise RuntimeError(f"Only {len(eligible)} event-free candidates for {TRAIN_RECORDS} training records")
    # Preserve broad coverage and force twelve event-free examples from the
    # upper 0.06-wide band whenever available.
    near_eligible = eligible[np.max(candidates[eligible], axis=1) >= upper - 0.06]
    near_take = near_eligible[np.linspace(0, len(near_eligible) - 1, min(12, len(near_eligible))).round().astype(int)]
    remaining = np.setdiff1d(eligible, near_take)
    remaining = remaining[np.argsort(np.max(candidates[remaining], axis=1))]
    broad_take = remaining[np.linspace(0, len(remaining) - 1, TRAIN_RECORDS - len(near_take)).round().astype(int)]
    chosen = np.r_[broad_take, near_take]
    chosen = chosen[np.argsort(np.max(candidates[chosen], axis=1))]
    selected = signals[:, chosen]
    return candidates[chosen], selected, trajectory_context(selected), len(eligible), len(near_take)


def expected_improvement(candidates, gp, best):
    mean, std = gp.predict(candidates, return_std=True)
    gain = best - mean
    z = gain / np.maximum(std, 1e-12)
    return gain * norm.cdf(z) + std * norm.pdf(z)


def plot_calibration(maxima, survival, x50):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(maxima, survival, "o-", color="black", label="training-only GT calibration")
    ax.axvline(X0_CRITICAL, color="#777777", ls=":", label=f"single-node reference ({X0_CRITICAL:.2f})")
    ax.axvline(x50, color="#a33a2b", ls="--", label=f"empirical 50% point ({x50:.3f})")
    ax.set(xlabel="max node x0", ylabel="120 s survival fraction", ylim=(-0.05, 1.05), title="Independent three-node GT boundary calibration")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "gt_boundary_calibration.png", dpi=180)
    plt.close(fig)


def plot_bo(rows):
    values = np.asarray([row["objective"] for row in rows])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(rows) + 1), values, "o-", ms=3, alpha=0.65, label="trial")
    ax.plot(np.arange(1, len(rows) + 1), np.minimum.accumulate(values), color="#176b3a", lw=2.5, label="best")
    ax.set(xlabel="BO trial", ylabel="normal + sustained-drop objective", title="Empirical-boundary three-node SCDT BO")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "bo_trace.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    maxima, calibration_targets, calibration_survival, x50, calibration_warmup, calibration_future, calibration_profiles, calibration_seed_ids, calibration_events = calibrate_boundary()
    stable_upper = x50 - 0.012
    train_profiles, stable_full, train_context, eligible_count, near_count = make_stable_training(stable_upper)
    coordinate, fitted_context, coordinate_metrics = fit_statistical_coordinate(train_profiles, train_context)
    dynamic_center, dynamic_scale = fit_normalizer(stable_full)
    static_center, static_scale = fit_normalizer(fitted_context)
    stable_raw = stable_full[-int(FUTURE_S * RC_HZ) :]
    stable_n = normalize(stable_raw, dynamic_center, dynamic_scale)
    stable_static = normalize(fitted_context, static_center, static_scale)

    event_maxima = x50 + np.asarray([0.02, 0.05, 0.08, 0.11])
    event_targets = profiles_from_maximum(event_maxima, SEED + 400)
    event_seed_values = tuple(SEED + 500 + index for index in range(3))
    _, event_raw, event_profiles, event_seed_ids = simulate_switch_cohort(
        event_targets,
        event_seed_values,
        WARMUP_S,
        FUTURE_S,
        event_threshold=DEFAULT_EVENT_THRESHOLD,
    )
    event_count = int(np.sum(np.isfinite(first_threshold_crossing(event_raw, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S))))
    event_n = normalize(event_raw, dynamic_center, dynamic_scale)
    event_static = normalize(coordinate.predict(event_profiles), static_center, static_scale)
    train_raw = np.concatenate((stable_raw, event_raw), axis=1)
    train_n = np.concatenate((stable_n, event_n), axis=1)
    train_static = np.concatenate((stable_static, event_static), axis=0)

    validation_maxima = x50 + np.linspace(-0.10, 0.14, VALIDATION_LEVELS)
    validation_targets = profiles_from_maximum(validation_maxima, SEED + 600)
    validation_seed_values = tuple(SEED + 700 + index for index in range(VALIDATION_SEEDS))
    validation_warmup, validation_future, validation_profiles, validation_seed_ids = simulate_switch_cohort(
        validation_targets,
        validation_seed_values,
        WARMUP_S,
        FUTURE_S,
        event_threshold=DEFAULT_EVENT_THRESHOLD,
    )

    rng = np.random.default_rng(SEED + 800)
    units, rows = [], []
    for trial in range(TRIALS):
        if trial < len(PRIOR_UNITS):
            unit = PRIOR_UNITS[trial].copy()
        elif trial < 12:
            # Resolve the known dynamics basin before global exploration.
            unit = np.clip(
                PRIOR_UNITS[0] + rng.normal(0.0, 0.10, 7),
                0.0,
                1.0,
            )
            unit[6] = 0.25
        elif trial < 20:
            unit = rng.random(7)
            unit[6] = 0.25 if trial % 2 == 0 else 0.75
        else:
            x = np.vstack(units)
            y = np.asarray([row["objective"] for row in rows])
            gp = GaussianProcessRegressor(
                kernel=Matern(np.ones(7), nu=2.5) + WhiteKernel(1e-5),
                normalize_y=True,
                random_state=SEED,
                n_restarts_optimizer=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(x, y)
            candidates = rng.random((4096, 7))
            candidates[:1024] = np.clip(x[int(np.argmin(y))] + rng.normal(0.0, 0.10, (1024, 7)), 0.0, 1.0)
            unit = candidates[int(np.argmax(expected_improvement(candidates, gp, float(np.min(y)))))]
        params, mode = parameter_from_unit(unit, SEED + 900)
        reservoir = build_reservoir(params, mode)
        fit_info = collect_weighted_readout(
            reservoir,
            train_n,
            train_raw,
            train_static,
            warmup=5 * RC_HZ,
            retain_spikes=False,
            spike_weight=1.0,
            low_weight=1.0,
        )
        metrics, prediction, _ = evaluate(
            reservoir,
            validation_warmup,
            validation_future,
            validation_profiles,
            coordinate,
            dynamic_center,
            dynamic_scale,
            static_center,
            static_scale,
            VALIDATION_LEVELS,
            VALIDATION_SEEDS,
            SEED + 1000,
            DEFAULT_EVENT_THRESHOLD,
            DEFAULT_EVENT_BLANKING_S,
        )
        shape = shape_metrics(validation_future, prediction)
        row = {
            "trial": trial + 1,
            "mode": mode,
            **{f"unit_{index}": float(value) for index, value in enumerate(unit)},
            **asdict(params),
            **fit_info,
            **{key: value for key, value in metrics.items() if key not in ("gt_survival", "rc_survival")},
            **shape,
            "objective": objective(metrics, shape),
        }
        units.append(unit)
        rows.append(row)
        with (OUT / "bo_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)

    # Enforce useful validation sensitivity/specificity before minimizing the
    # continuous objective. This prevents selection of a morphology-only model.
    feasible = [
        index for index, row in enumerate(rows)
        if row["sustained_hits"] >= 0.75 * (row["sustained_hits"] + row["sustained_misses"])
        and row["sustained_false_alarms"] <= 0.10 * (row["gt_exits"] + row["correct_nonexits"])
    ]
    selected = min(feasible, key=lambda index: rows[index]["objective"]) if feasible else int(np.argmin([row["objective"] for row in rows]))
    params, mode = parameter_from_unit(units[selected], SEED + 900)
    winner = build_reservoir(params, mode)
    winner_fit = collect_weighted_readout(winner, train_n, train_raw, train_static, 5 * RC_HZ, False, 1.0, 1.0)

    test_maxima = x50 + np.linspace(-0.12, 0.16, TEST_LEVELS)
    test_targets = profiles_from_maximum(test_maxima, SEED + 1100)
    test_seed_values = tuple(SEED + 1200 + index for index in range(TEST_SEEDS))
    test_warmup, test_future, test_profiles, test_seed_ids = simulate_switch_cohort(
        test_targets,
        test_seed_values,
        WARMUP_S,
        FUTURE_S,
        event_threshold=DEFAULT_EVENT_THRESHOLD,
    )
    test_metrics, test_prediction, test_rc_events = evaluate(
        winner,
        test_warmup,
        test_future,
        test_profiles,
        coordinate,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        TEST_LEVELS,
        TEST_SEEDS,
        SEED + 1300,
        DEFAULT_EVENT_THRESHOLD,
        DEFAULT_EVENT_BLANKING_S,
    )
    test_shape = shape_metrics(test_future, test_prediction)
    result = {
        "status": "complete",
        "boundary_calibration": {
            "single_node_reference": X0_CRITICAL,
            "grid": maxima.tolist(),
            "survival": calibration_survival.tolist(),
            "empirical_x50": x50,
            "stable_training_upper": stable_upper,
            "seeds": list(range(SEED + 100, SEED + 108)),
        },
        "training": {
            "stable_records": TRAIN_RECORDS,
            "eligible_candidates": eligible_count,
            "near_boundary_records": near_count,
            "stable_maximum_range": [float(np.min(np.max(train_profiles, axis=1))), float(np.max(np.max(train_profiles, axis=1)))],
            "event_records": int(event_raw.shape[1]),
            "event_departures": event_count,
            "event_maxima": event_maxima.tolist(),
            "normalizer_and_coordinate_fit": "stable training only",
            **winner_fit,
        },
        "coordinate_calibration": coordinate_metrics,
        "selected_trial": selected + 1,
        "selected_from_feasible_set": bool(feasible),
        "feasible_trial_count": len(feasible),
        "selected_mode": mode,
        "selected_parameters": asdict(params),
        "validation": rows[selected],
        "fresh_locked_test": {
            **{key: value for key, value in test_metrics.items() if key not in ("gt_survival", "rc_survival")},
            **test_shape,
            "gt_survival": test_metrics["gt_survival"],
            "rc_survival": test_metrics["rc_survival"],
            "objective": objective(test_metrics, test_shape),
        },
    }
    save_json(OUT / "summary.json", result)
    save_json(
        OUT / "protocol.json",
        {
            "date": "2026-09-10",
            "purpose": "independently calibrate the coupled finite-horizon GT boundary before expanding stable SCDT training",
            "old_locked_test_used": False,
            "empirical_boundary_used_for_design": x50,
            "stable_margin_below_x50": 0.012,
            "bo_trials": TRIALS,
            "reservoir_nodes": 1500,
            "spectral_radius_range": [0.30, 2.50],
            "leakage_rate_range": [0.01, 0.70],
            "input": "three node signals plus nine fitted stable-trajectory statistics; no direct x0",
            "event": "any node <= -0.5 after 5 s; sustained endpoint requires at least 5 s continuous low state",
            "all_seed_families": {
                "boundary": list(range(SEED + 100, SEED + 108)),
                "training": SEED + 302,
                "event_training": list(event_seed_values),
                "validation": list(validation_seed_values),
                "fresh_locked_test": list(test_seed_values),
            },
        },
    )
    np.savez_compressed(OUT / "boundary_calibration.npz", targets=calibration_targets, maxima=maxima, survival=calibration_survival, warmup=calibration_warmup, future=calibration_future, profiles=calibration_profiles, seeds=calibration_seed_ids, events=calibration_events)
    np.savez_compressed(OUT / "training.npz", profiles=train_profiles, stable_signals=stable_full, observed_context=train_context, fitted_context=fitted_context, event_profiles=event_profiles, event_seeds=event_seed_ids, event_signals=event_raw)
    np.savez_compressed(OUT / "validation.npz", targets=validation_targets, profiles=validation_profiles, seeds=validation_seed_ids, warmup=validation_warmup, future=validation_future)
    np.savez_compressed(OUT / "fresh_locked_test.npz", targets=test_targets, profiles=test_profiles, seeds=test_seed_ids, warmup=test_warmup, future=test_future, prediction=test_prediction, rc_events=test_rc_events)
    np.savez_compressed(OUT / "winner_model.npz", a_data=winner.a.data, a_indices=winner.a.indices, a_indptr=winner.a.indptr, a_shape=winner.a.shape, w_in=winner.w_in, w_static=winner.w_static, w_out=winner.w_out, residuals=winner.residuals, readout_indices=winner.readout_indices, dynamic_center=dynamic_center, dynamic_scale=dynamic_scale, static_center=static_center, static_scale=static_scale, clip_low=winner.clip_low, clip_high=winner.clip_high, coordinate_coef=coordinate.coef_, coordinate_intercept=coordinate.intercept_)
    plot_calibration(maxima, calibration_survival, x50)
    plot_bo(rows)
    plot_survival(FIG, test_targets, {**test_metrics, "gt_survival": test_metrics["gt_survival"], "rc_survival": test_metrics["rc_survival"]}, "Fresh locked")
    plot_gt_rc_grid(FIG, test_warmup, test_future, test_prediction, test_profiles, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S)
    (OUT / "RESULTS.md").write_text(
        "# Empirical-boundary three-node SCDT BO\n\n"
        f"Independent GT calibration placed the finite-horizon 50% point at x0={x50:.4f}; the single-node paper reference is {X0_CRITICAL:.2f}. Stable training was expanded through x0={stable_upper:.4f}, with {near_count} deliberately sampled near-boundary records.\n\n"
        f"- Selected BO trial / mode: {selected + 1} / {mode}\n"
        f"- Feasible validation trials: {len(feasible)} / {TRIALS}\n"
        f"- Fresh locked sustained hits / misses / false alarms: {test_shape['sustained_hits']} / {test_shape['sustained_misses']} / {test_shape['sustained_false_alarms']}\n"
        f"- Fresh locked morphology / spike-rate MAE: {test_metrics['morphology_score']:.4f} / {test_metrics['normal_spike_rate_mae']:.4f}\n"
        f"- Fresh locked q05 / occupancy / run: {test_shape['rc_event_q05_median']:.4f} / {test_shape['rc_event_occupancy_median']:.4f} / {test_shape['rc_event_run_median_s']:.2f} s\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
