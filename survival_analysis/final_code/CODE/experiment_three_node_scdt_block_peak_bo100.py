#!/usr/bin/env python3
"""Peak-weighted continuation BO for the 1,500-node block-innovation SCDT."""

from __future__ import annotations

import csv
import gc
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

from experiment_three_node_parameter_space_rc import fit_normalizer
from experiment_three_node_scdt import (
    DEFAULT_EVENT_BLANKING_S,
    DEFAULT_EVENT_THRESHOLD,
    N_NODES,
    RC_HZ,
    evaluate,
    first_threshold_crossing,
    fit_statistical_coordinate,
    normalize,
    plot_gt_rc_grid,
    plot_survival,
    simulate_switch_cohort,
)
from experiment_three_node_scdt_block_innovation import (
    BlockInnovationRC,
    InnovationVariant,
    configure,
    load_locked_model,
    plot_normal_zoom,
    prepare_training as prepare_locked_training,
    profiles_from_maximum,
    sequential_residuals,
)
from experiment_three_node_scdt_dynamics_pilot import collect_weighted_readout, shape_metrics
from paper_single_node_scrc import RCParams


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "three_node_scdt_empirical_boundary_bo100"
PREVIOUS = ROOT / "experiments" / "three_node_scdt_block_bo100"
OUT = ROOT / "experiments" / "three_node_scdt_block_peak_bo100"
FIG = ROOT / "figures" / "RC" / "three_node_scdt_block_peak_bo100"
PREVIOUS_TRIALS = 100
NEW_TRIALS = 100
TRIALS = PREVIOUS_TRIALS + NEW_TRIALS
SEED = 733000
WARMUP_S = 40.0
FUTURE_S = 120.0
VALIDATION_LEVELS = 21
VALIDATION_SEEDS = 4
TEST_LEVELS = 33
TEST_SEEDS = 8


# First point is the block-pilot winner. Remaining prespecified points resolve
# nearby block lengths/scales before the global BO phase.
PRIOR_UNITS = (
    np.asarray([0.23908250, 0.04222604, 0.33411638, 0.12349039, 0.81763129, 4 / 7, 1.0]),
    np.asarray([0.23908250, 0.04222604, 0.33411638, 0.12349039, 0.81763129, 2 / 7, 1.0]),
    np.asarray([0.23908250, 0.04222604, 0.33411638, 0.12349039, 0.81763129, 4 / 7, 0.80]),
    np.asarray([0.23908250, 0.04222604, 0.33411638, 0.12349039, 0.81763129, 4 / 7, 0.60]),
    np.asarray([0.23908250, 0.04222604, 0.33411638, 0.12349039, 0.81763129, 3 / 7, 0.90]),
    np.asarray([0.08878345, 0.0, 0.21906929, 0.0, 0.83393283, 4 / 7, 1.0]),
)


def parameters_from_unit(unit: np.ndarray) -> tuple[RCParams, float]:
    unit = np.asarray(unit, dtype=float)
    if unit.shape != (7,) or np.any((unit < 0.0) | (unit > 1.0)):
        raise ValueError("BO unit must have seven coordinates in [0,1]")
    params = RCParams(
        n_reservoir=1500,
        spectral_radius=0.30 + 2.20 * unit[0],
        leakage_rate=10 ** (-2.0 + np.log10(70.0) * unit[1]),
        input_scaling=10 ** (-1.7 + 1.6 * unit[2]),
        static_scaling=10 ** (-2.5 + 2.2 * unit[3]),
        ridge=10 ** (-3.0 + 4.5 * unit[4]),
        density=0.003,
        innovation_scaling=0.20 + 0.70 * unit[5],
        seed=SEED + 900,
    )
    block_s = 0.125 * 32.0 ** unit[6]
    return params, float(block_s)


def expected_improvement(candidates, gp, best):
    mean, std = gp.predict(candidates, return_std=True)
    gain = best - mean
    z = gain / np.maximum(std, 1e-12)
    return gain * norm.cdf(z) + std * norm.pdf(z)


def curve_nonmonotonicity(survival: list[float]) -> float:
    """Survival should not rise as maximum x0 moves toward the event regime."""
    return float(np.sum(np.maximum(np.diff(np.asarray(survival, dtype=float)), 0.0)))


def peak_metrics(gt: np.ndarray, rc: np.ndarray) -> dict[str, float]:
    """Measure the height of the sparse positive tail on GT-normal records."""
    start = int(round(DEFAULT_EVENT_BLANKING_S * RC_HZ))
    gt_events = np.isfinite(
        first_threshold_crossing(gt, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S)
    )
    normal = ~gt_events
    if not np.any(normal):
        return {"normal_peak_q995_mae": float("inf")}
    gt_q = np.quantile(gt[start:, normal], 0.995, axis=0)
    rc_q = np.quantile(rc[start:, normal], 0.995, axis=0)
    return {"normal_peak_q995_mae": float(np.mean(np.abs(gt_q - rc_q)))}


def peak_objective(metrics: dict, peak_q995_mae: float) -> float:
    records = metrics["hits"] + metrics["misses"] + metrics["false_alarms"] + metrics["correct_nonexits"]
    point_error = (metrics["misses"] + metrics["false_alarms"]) / records
    nonmonotonicity = curve_nonmonotonicity(metrics["rc_survival"])
    return float(
        20.0 * max(metrics["survival_mae"] - 0.05, 0.0)
        + 20.0 * max(point_error - 0.06, 0.0)
        + 10.0 * nonmonotonicity
        + 0.60 * metrics["morphology_score"]
        + 1.00 * metrics["normal_spike_rate_mae"] / 0.02
        + 1.00 * peak_q995_mae / 0.50
        + 50.0 * metrics["clipping_fraction"]
    )


def stored_peak_objective(row: dict) -> float:
    """Score a saved row, using a neutral q99.5 prior for legacy observations."""
    records = row["hits"] + row["misses"] + row["false_alarms"] + row["correct_nonexits"]
    point_error = (row["misses"] + row["false_alarms"]) / records
    peak_q995_mae = row.get("normal_peak_q995_mae", 0.50)
    return float(
        20.0 * max(row["survival_mae"] - 0.05, 0.0)
        + 20.0 * max(point_error - 0.06, 0.0)
        + 10.0 * row["curve_nonmonotonicity"]
        + 0.60 * row["morphology_score"]
        + 1.00 * row["normal_spike_rate_mae"] / 0.02
        + 1.00 * peak_q995_mae / 0.50
        + 50.0 * row["clipping_fraction"]
    )


def prepare_training():
    with np.load(SOURCE / "training.npz") as data:
        profiles = data["profiles"]
        stable_full = data["stable_signals"]
        observed_context = data["observed_context"]
        event_profiles = data["event_profiles"]
        event_raw = data["event_signals"]
    coordinate, fitted_context, coordinate_metrics = fit_statistical_coordinate(profiles, observed_context)
    dynamic_center, dynamic_scale = fit_normalizer(stable_full)
    static_center, static_scale = fit_normalizer(fitted_context)
    stable_raw = stable_full[-int(FUTURE_S * RC_HZ) :]
    stable_n = normalize(stable_raw, dynamic_center, dynamic_scale)
    event_n = normalize(event_raw, dynamic_center, dynamic_scale)
    train_raw = np.concatenate((stable_raw, event_raw), axis=1)
    train_n = np.concatenate((stable_n, event_n), axis=1)
    stable_static = normalize(fitted_context, static_center, static_scale)
    event_static = normalize(coordinate.predict(event_profiles), static_center, static_scale)
    train_static = np.concatenate((stable_static, event_static), axis=0)
    normalizers = dynamic_center, dynamic_scale, static_center, static_scale
    return coordinate, coordinate_metrics, train_raw, train_n, train_static, normalizers, stable_n.shape[1]


def load_or_make_validation():
    path = OUT / "validation.npz"
    if path.exists():
        with np.load(path) as data:
            return {name: data[name] for name in data.files}
    maxima = np.linspace(-2.30, -1.90, VALIDATION_LEVELS)
    targets = profiles_from_maximum(maxima, SEED + 10)
    seeds = tuple(SEED + 100 + index for index in range(VALIDATION_SEEDS))
    warmup, future, profiles, seed_ids = simulate_switch_cohort(
        targets,
        seeds,
        WARMUP_S,
        FUTURE_S,
        event_threshold=DEFAULT_EVENT_THRESHOLD,
    )
    np.savez_compressed(path, targets=targets, maxima=maxima, profiles=profiles, seeds=seed_ids, warmup=warmup, future=future)
    return {"targets": targets, "maxima": maxima, "profiles": profiles, "seeds": seed_ids, "warmup": warmup, "future": future}


def fit_block_model(params, block_s, train_n, train_raw, train_static, stable_count):
    model = BlockInnovationRC(
        params,
        dynamic_dim=N_NODES,
        static_dim=9,
        readout_stride=4,
        readout_size=1500,
    )
    fit_info = collect_weighted_readout(
        model,
        train_n,
        train_raw,
        train_static,
        warmup=5 * RC_HZ,
        retain_spikes=False,
        spike_weight=1.0,
        low_weight=1.0,
    )
    model.residual_sequences = sequential_residuals(model, train_n, train_static, 5 * RC_HZ)
    model.residual_pool_indices = np.arange(stable_count)
    model.innovation_mode = "block"
    model.innovation_block_steps = max(1, int(round(block_s * RC_HZ)))
    return model, fit_info


def evaluate_cohort(model, cohort, coordinate, normalizers, seed):
    dynamic_center, dynamic_scale, static_center, static_scale = normalizers
    profile_count = len(cohort["targets"])
    replicate_count = len(cohort["profiles"]) // profile_count
    return evaluate(
        model,
        cohort["warmup"],
        cohort["future"],
        cohort["profiles"],
        coordinate,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        profile_count,
        replicate_count,
        seed,
        DEFAULT_EVENT_THRESHOLD,
        DEFAULT_EVENT_BLANKING_S,
    )


def plot_bo(rows):
    values = np.asarray([row["objective"] for row in rows])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(rows) + 1), values, "o-", ms=3, alpha=0.6, label="trial")
    ax.plot(np.arange(1, len(rows) + 1), np.minimum.accumulate(values), color="#176b3a", lw=2.5, label="best")
    ax.axvline(PREVIOUS_TRIALS + 0.5, color="0.4", ls="--", lw=1.2, label="peak-weighted continuation")
    ax.set(xlabel="BO observation", ylabel="rate-constrained peak objective", title="Three-node SCDT peak-weighted block BO")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "bo_trace.png", dpi=180)
    plt.close(fig)


def plot_rate_comparison(targets, gt_survival, old_survival, new_survival, figdir: Path = FIG):
    x0 = np.max(targets, axis=1)
    order = np.argsort(x0)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x0[order], np.asarray(gt_survival)[order], "o-", color="black", label="GT")
    ax.plot(x0[order], np.asarray(old_survival)[order], "^:", color="#d07a24", label="rate-weighted BO100")
    ax.plot(x0[order], np.asarray(new_survival)[order], "s--", color="#176b3a", label="peak-weighted BO100")
    ax.set_xticks(x0[order])
    ax.set_xticklabels([f"{value:.3f}" for value in x0[order]], rotation=45, ha="right", fontsize=8)
    ax.set(xlabel="x0", ylabel="120 s survival fraction", ylim=(-0.05, 1.05), title="Rate- versus peak-weighted BO on the fresh locked cohort")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "survival_rate_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    coordinate, coordinate_metrics, train_raw, train_n, train_static, normalizers, stable_count = prepare_training()
    validation = load_or_make_validation()
    dynamic_center, dynamic_scale, static_center, static_scale = normalizers
    history_path = OUT / "bo_history.csv"
    units: list[np.ndarray] = []
    rows: list[dict] = []
    seed_history = history_path if history_path.exists() else PREVIOUS / "bo_history.csv"
    if seed_history.exists():
        with seed_history.open(newline="") as handle:
            saved = list(csv.DictReader(handle))
        for saved_row in saved:
            units.append(np.asarray([float(saved_row[f"unit_{index}"]) for index in range(7)]))
            row = {key: value if key == "mode" else float(value) for key, value in saved_row.items()}
            row.pop("objective", None)
            if row["trial"] <= PREVIOUS_TRIALS:
                # Legacy histories did not record this statistic. Use the
                # measured value from the exact rate-winner re-evaluation.
                row["normal_peak_q995_mae"] = 0.8185681806473207
            row["objective"] = stored_peak_objective(row)
            rows.append(row)
    rng = np.random.default_rng(SEED + 1800)
    random_cache = rng.random((20, 7))
    while len(rows) < TRIALS:
        trial = len(rows)
        if trial == PREVIOUS_TRIALS:
            # Re-evaluate the rate-BO winner so every selectable candidate has
            # the new positive-tail amplitude metric.
            previous_summary = json.loads((PREVIOUS / "summary.json").read_text())
            unit = units[int(previous_summary["selected_trial"]) - 1].copy()
        elif trial < len(PRIOR_UNITS):
            unit = PRIOR_UNITS[trial].copy()
        elif trial < 20:
            unit = random_cache[trial]
        else:
            # Once the continuation has started, fit the acquisition model to
            # rows that contain the measured q99.5 peak term. The preceding
            # BO100 rows have only a neutral prior for that newly added metric.
            x = np.vstack(units[PREVIOUS_TRIALS:])
            raw_y = np.asarray([row["objective"] for row in rows[PREVIOUS_TRIALS:]])
            # The broad box contains catastrophic clipped rollouts. A log
            # response prevents those outliers from dominating the useful
            # low-objective basin identified by the seeded designs.
            y = np.log1p(raw_y)
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
            local_count = 4096 if trial % 5 else 3072
            local_scale = 0.035 if trial < 160 else 0.020
            candidates[:local_count] = np.clip(
                x[int(np.argmin(y))] + rng.normal(0.0, local_scale, (local_count, 7)),
                0.0,
                1.0,
            )
            unit = candidates[int(np.argmax(expected_improvement(candidates, gp, float(np.min(y)))))]
        params, block_s = parameters_from_unit(unit)
        model, fit_info = fit_block_model(params, block_s, train_n, train_raw, train_static, stable_count)
        metrics, prediction, _ = evaluate_cohort(model, validation, coordinate, normalizers, SEED + 1000)
        shape = shape_metrics(validation["future"], prediction)
        peak = peak_metrics(validation["future"], prediction)
        row = {
            "trial": trial + 1,
            "mode": "next+block",
            **{f"unit_{index}": float(value) for index, value in enumerate(unit)},
            **asdict(params),
            "block_s": block_s,
            "block_steps": model.innovation_block_steps,
            **fit_info,
            **{key: value for key, value in metrics.items() if key not in ("gt_survival", "rc_survival")},
            "curve_nonmonotonicity": curve_nonmonotonicity(metrics["rc_survival"]),
            **shape,
            **peak,
            "objective": peak_objective(metrics, peak["normal_peak_q995_mae"]),
        }
        units.append(unit)
        rows.append(row)
        with history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        del model, prediction
        gc.collect()

    selectable = range(PREVIOUS_TRIALS, len(rows))
    rate_winner = rows[PREVIOUS_TRIALS]
    feasible = [
        index for index, row in enumerate(rows)
        if index in selectable
        and row["survival_mae"] <= 0.05
        and row["curve_nonmonotonicity"] <= 1e-12
        and row["hits"] >= 0.85 * (row["hits"] + row["misses"])
        and row["false_alarms"] <= 0.06 * (row["false_alarms"] + row["correct_nonexits"])
        and row["sustained_hits"] >= 0.80 * (row["sustained_hits"] + row["sustained_misses"])
        and row["sustained_false_alarms"] <= 0.05 * (row["sustained_false_alarms"] + row["correct_nonexits"])
        and row["morphology_score"] <= 1.05
        and row["normal_spike_rate_mae"] <= rate_winner["normal_spike_rate_mae"] + 1e-12
        and row["normal_peak_q995_mae"] <= rate_winner["normal_peak_q995_mae"] + 1e-12
    ]
    selected = min(feasible, key=lambda index: rows[index]["objective"]) if feasible else min(
        selectable, key=lambda index: rows[index]["objective"]
    )
    winner_params, winner_block_s = parameters_from_unit(units[selected])
    winner, winner_fit = fit_block_model(winner_params, winner_block_s, train_n, train_raw, train_static, stable_count)

    maxima = np.linspace(-2.50, -1.90, TEST_LEVELS)
    test_targets = profiles_from_maximum(maxima, SEED + 3000)
    test_seed_values = tuple(SEED + 3100 + index for index in range(TEST_SEEDS))
    test_warmup, test_future, test_profiles, test_seed_ids = simulate_switch_cohort(
        test_targets,
        test_seed_values,
        WARMUP_S,
        FUTURE_S,
        event_threshold=DEFAULT_EVENT_THRESHOLD,
    )
    test = {"targets": test_targets, "maxima": maxima, "profiles": test_profiles, "seeds": test_seed_ids, "warmup": test_warmup, "future": test_future}
    test_metrics, test_prediction, test_rc_events = evaluate_cohort(winner, test, coordinate, normalizers, SEED + 3200)
    test_shape = shape_metrics(test_future, test_prediction)
    test_peak = peak_metrics(test_future, test_prediction)

    previous_summary = json.loads((PREVIOUS / "summary.json").read_text())
    baseline_params = RCParams(**previous_summary["selected_parameters"])
    baseline, _ = fit_block_model(
        baseline_params,
        float(previous_summary["selected_block_s"]),
        train_n,
        train_raw,
        train_static,
        stable_count,
    )
    baseline_metrics, baseline_prediction, _ = evaluate_cohort(
        baseline,
        test,
        coordinate,
        normalizers,
        SEED + 3200,
    )
    baseline_shape = shape_metrics(test_future, baseline_prediction)
    baseline_peak = peak_metrics(test_future, baseline_prediction)
    result = {
        "status": "complete",
        "selected_trial": selected + 1,
        "selected_from_feasible_set": bool(feasible),
        "feasible_trial_count": len(feasible),
        "selected_parameters": asdict(winner_params),
        "selected_block_s": winner_block_s,
        "selected_block_steps": winner.innovation_block_steps,
        "coordinate_calibration": coordinate_metrics,
        "training": {"stable_records": stable_count, "event_records": int(train_n.shape[1] - stable_count), **winner_fit},
        "validation": rows[selected],
        "fresh_test": {
            **{key: value for key, value in test_metrics.items() if key not in ("gt_survival", "rc_survival")},
            "curve_nonmonotonicity": curve_nonmonotonicity(test_metrics["rc_survival"]),
            **test_shape,
            **test_peak,
            "gt_survival": test_metrics["gt_survival"],
            "rc_survival": test_metrics["rc_survival"],
            "objective": peak_objective(test_metrics, test_peak["normal_peak_q995_mae"]),
        },
        "paired_rate_bo100": {
            **{key: value for key, value in baseline_metrics.items() if key not in ("gt_survival", "rc_survival")},
            "curve_nonmonotonicity": curve_nonmonotonicity(baseline_metrics["rc_survival"]),
            **baseline_shape,
            **baseline_peak,
            "gt_survival": baseline_metrics["gt_survival"],
            "rc_survival": baseline_metrics["rc_survival"],
            "objective": peak_objective(baseline_metrics, baseline_peak["normal_peak_q995_mae"]),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    (OUT / "protocol.json").write_text(json.dumps({
        "date": "2026-09-12",
        "previous_observations": PREVIOUS_TRIALS,
        "new_bo_trials": NEW_TRIALS,
        "total_bo_observations": TRIALS,
        "reservoir_nodes": 1500,
        "spectral_radius_range": [0.30, 2.50],
        "leakage_rate_range": [0.01, 0.70],
        "innovation_scaling_range": [0.20, 0.90],
        "block_length_range_s": [0.125, 4.0],
        "block_pool": "stable training trajectories only",
        "selection": "strict rate gate followed by a peak-weighted objective using spike-rate, morphology and normal q99.5 amplitude errors",
        "validation": {"x0_range": [-2.30, -1.90], "levels": VALIDATION_LEVELS, "seeds": list(range(SEED + 100, SEED + 100 + VALIDATION_SEEDS))},
        "fresh_test": {"x0_range": [-2.50, -1.90], "levels": TEST_LEVELS, "seeds": list(test_seed_values)},
    }, indent=2))
    np.savez_compressed(
        OUT / "fresh_test.npz",
        targets=test_targets,
        maxima=maxima,
        profiles=test_profiles,
        seeds=test_seed_ids,
        warmup=test_warmup,
        future=test_future,
        prediction=test_prediction,
        baseline_prediction=baseline_prediction,
        rc_events=test_rc_events,
    )
    np.savez_compressed(
        OUT / "winner_model.npz",
        a_data=winner.a.data,
        a_indices=winner.a.indices,
        a_indptr=winner.a.indptr,
        a_shape=winner.a.shape,
        w_in=winner.w_in,
        w_static=winner.w_static,
        w_out=winner.w_out,
        residuals=winner.residuals,
        residual_sequences=winner.residual_sequences,
        residual_pool_indices=winner.residual_pool_indices,
        readout_indices=winner.readout_indices,
        dynamic_center=dynamic_center,
        dynamic_scale=dynamic_scale,
        static_center=static_center,
        static_scale=static_scale,
        clip_low=winner.clip_low,
        clip_high=winner.clip_high,
        coordinate_coef=coordinate.coef_,
        coordinate_intercept=coordinate.intercept_,
    )
    plot_bo(rows)
    plot_survival(FIG, test_targets, test_metrics, "Peak-weighted block-BO fresh locked")
    plot_rate_comparison(
        test_targets,
        test_metrics["gt_survival"],
        baseline_metrics["rc_survival"],
        test_metrics["rc_survival"],
    )
    plot_gt_rc_grid(FIG, test_warmup, test_future, test_prediction, test_profiles, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S)
    gt_events = first_threshold_crossing(
        test_future,
        DEFAULT_EVENT_THRESHOLD,
        DEFAULT_EVENT_BLANKING_S,
    )
    plot_normal_zoom(
        test_future,
        {"rate-BO": baseline_prediction, "peak-BO": test_prediction},
        gt_events,
        figdir=FIG,
    )
    (OUT / "RESULTS.md").write_text(
        "# Three-node peak-weighted block-innovation BO100\n\n"
        f"Selected trial {selected + 1} with a {winner_block_s:.3f} s residual block. The fresh test uses {TEST_LEVELS} equally spaced maximum-x0 levels from -2.50 to -1.90 and {TEST_SEEDS} new seeds.\n\n"
        f"- Point hits/misses/false alarms: {test_metrics['hits']}/{test_metrics['misses']}/{test_metrics['false_alarms']}\n"
        f"- Survival MAE / curve nonmonotonicity: {test_metrics['survival_mae']:.4f} / {curve_nonmonotonicity(test_metrics['rc_survival']):.4f}\n"
        f"- Normal spike-rate MAE / q99.5 peak MAE / morphology: {test_metrics['normal_spike_rate_mae']:.4f} / {test_peak['normal_peak_q995_mae']:.4f} / {test_metrics['morphology_score']:.4f}\n"
        f"- Sustained hits/misses/false alarms: {test_shape['sustained_hits']}/{test_shape['sustained_misses']}/{test_shape['sustained_false_alarms']}\n"
        f"- Paired rate-BO survival MAE / nonmonotonicity: {baseline_metrics['survival_mae']:.4f} / {curve_nonmonotonicity(baseline_metrics['rc_survival']):.4f}\n"
        f"- Paired rate-BO spike-rate MAE / q99.5 peak MAE / morphology: {baseline_metrics['normal_spike_rate_mae']:.4f} / {baseline_peak['normal_peak_q995_mae']:.4f} / {baseline_metrics['morphology_score']:.4f}\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
