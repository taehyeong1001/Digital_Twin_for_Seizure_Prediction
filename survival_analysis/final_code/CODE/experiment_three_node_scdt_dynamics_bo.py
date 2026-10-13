#!/usr/bin/env python3
"""BO for normal spikes and sustained postcritical seizure-like drops."""

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

from experiment_three_node_parameter_space_rc import fit_normalizer
from experiment_three_node_scdt import (
    DEFAULT_EVENT_BLANKING_S,
    DEFAULT_EVENT_THRESHOLD,
    RC_HZ,
    build_reservoir,
    evaluate,
    fit_statistical_coordinate,
    normalize,
    parameter_from_unit,
    plot_gt_rc_grid,
    plot_survival,
    simulate_switch_cohort,
)
from experiment_three_node_scdt_dynamics_pilot import (
    collect_weighted_readout,
    event_training_profiles,
    shape_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "three_node_scdt_sp25_lr07_bo100"
OUT = ROOT / "experiments" / "three_node_scdt_dynamics_bo40"
FIG = ROOT / "figures" / "RC" / "three_node_scdt_dynamics_bo40"
TRIALS = 40


def expected_improvement(candidates, gp, best):
    mean, std = gp.predict(candidates, return_std=True)
    improvement = best - mean
    z = improvement / np.maximum(std, 1e-12)
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def objective(metrics: dict, shape: dict) -> float:
    records = metrics["hits"] + metrics["misses"] + metrics["false_alarms"] + metrics["correct_nonexits"]
    sustained_error = (
        shape["sustained_misses"] + shape["sustained_false_alarms"]
    ) / records
    shape_error = np.mean(
        [
            shape["event_q05_mae"] / 1.0,
            shape["event_occupancy_mae"] / 0.50,
            shape["event_run_mae_s"] / 50.0,
        ]
    )
    spike_error = metrics["normal_spike_rate_mae"] / 0.02
    return float(
        metrics["morphology_score"]
        + 0.5 * spike_error
        + 4.0 * sustained_error
        + shape_error
        + 50.0 * metrics["clipping_fraction"]
    )


def prepare_data():
    with np.load(SOURCE / "training.npz") as data:
        train_profiles = data["profiles"]
        train_raw_full = data["signals"]
        train_context = data["observed_context"]
    coordinate, fitted_context, calibration = fit_statistical_coordinate(
        train_profiles, train_context
    )
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
    event_n = normalize(event_raw, dynamic_center, dynamic_scale)
    event_static = normalize(
        coordinate.predict(event_profiles), static_center, static_scale
    )
    train_raw = np.concatenate((stable_raw, event_raw), axis=1)
    train_n = np.concatenate((stable_n, event_n), axis=1)
    train_static = np.concatenate((stable_static, event_static), axis=0)

    with np.load(SOURCE / "validation.npz") as data:
        validation = {name: data[name] for name in data.files}
    with np.load(SOURCE / "locked_test.npz") as data:
        locked = {name: data[name] for name in data.files}
    normalizers = (dynamic_center, dynamic_scale, static_center, static_scale)
    return coordinate, calibration, train_raw, train_n, train_static, validation, locked, normalizers, event_targets, event_raw


def plot_bo(records: list[dict]) -> None:
    trial = np.arange(1, len(records) + 1)
    score = np.asarray([row["objective"] for row in records])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(trial, score, "o-", alpha=0.7, label="trial")
    ax.plot(trial, np.minimum.accumulate(score), color="#176b3a", lw=2.5, label="best")
    ax.set(
        xlabel="BO trial",
        ylabel="normal + sustained-drop objective",
        title="Three-node SCDT dynamics-aware BO",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "bo_trace.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (
        coordinate,
        calibration,
        train_raw,
        train_n,
        train_static,
        validation,
        locked,
        normalizers,
        event_targets,
        event_raw,
    ) = prepare_data()
    dynamic_center, dynamic_scale, static_center, static_scale = normalizers
    validation_profiles = len(validation["targets"])
    validation_seeds = validation["profiles"].shape[0] // validation_profiles

    rng = np.random.default_rng(690000)
    units: list[np.ndarray] = []
    records: list[dict] = []
    for trial in range(TRIALS):
        if trial < 14:
            unit = rng.random(7)
            unit[6] = 0.25 if trial % 2 == 0 else 0.75
        else:
            x = np.vstack(units)
            y = np.asarray([row["objective"] for row in records])
            gp = GaussianProcessRegressor(
                kernel=Matern(np.ones(7), nu=2.5) + WhiteKernel(1e-5),
                normalize_y=True,
                random_state=690000,
                n_restarts_optimizer=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(x, y)
            candidates = rng.random((4096, 7))
            candidates[:1024] = np.clip(
                x[int(np.argmin(y))] + rng.normal(0.0, 0.10, (1024, 7)),
                0.0,
                1.0,
            )
            unit = candidates[
                int(np.argmax(expected_improvement(candidates, gp, float(np.min(y)))))
            ]
        params, mode = parameter_from_unit(unit, seed=694000)
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
            validation["warmup"],
            validation["future"],
            validation["profiles"],
            coordinate,
            dynamic_center,
            dynamic_scale,
            static_center,
            static_scale,
            validation_profiles,
            validation_seeds,
            seed=695000,
            event_threshold=DEFAULT_EVENT_THRESHOLD,
            event_blanking_s=DEFAULT_EVENT_BLANKING_S,
        )
        shape = shape_metrics(validation["future"], prediction)
        score = objective(metrics, shape)
        row = {
            "trial": trial + 1,
            "mode": mode,
            **{f"unit_{index}": float(value) for index, value in enumerate(unit)},
            **asdict(params),
            **fit_info,
            **{key: value for key, value in metrics.items() if key not in ("gt_survival", "rc_survival")},
            **shape,
            "objective": score,
        }
        units.append(unit)
        records.append(row)
        with (OUT / "bo_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps(row), flush=True)

    winner_index = int(np.argmin([row["objective"] for row in records]))
    winner_params, winner_mode = parameter_from_unit(units[winner_index], seed=694000)
    winner = build_reservoir(winner_params, winner_mode)
    winner_fit = collect_weighted_readout(
        winner,
        train_n,
        train_raw,
        train_static,
        warmup=5 * RC_HZ,
        retain_spikes=False,
        spike_weight=1.0,
        low_weight=1.0,
    )
    locked_profiles = len(locked["targets"])
    locked_seeds = locked["profiles"].shape[0] // locked_profiles
    locked_metrics, locked_prediction, locked_events = evaluate(
        winner,
        locked["warmup"],
        locked["future"],
        locked["profiles"],
        coordinate,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        locked_profiles,
        locked_seeds,
        seed=696000,
        event_threshold=DEFAULT_EVENT_THRESHOLD,
        event_blanking_s=DEFAULT_EVENT_BLANKING_S,
    )
    locked_shape = shape_metrics(locked["future"], locked_prediction)
    result = {
        "status": "complete",
        "selected_trial": winner_index + 1,
        "selected_mode": winner_mode,
        "selected_parameters": asdict(winner_params),
        "training": {
            "stable_records": 32,
            "postcritical_records": int(event_raw.shape[1]),
            "postcritical_target_profiles": event_targets.tolist(),
            "weighting": "uniform every-fourth-sample readout",
            **winner_fit,
        },
        "coordinate_calibration": calibration,
        "validation": records[winner_index],
        "locked_test": {
            **{key: value for key, value in locked_metrics.items() if key not in ("gt_survival", "rc_survival")},
            **locked_shape,
            "gt_survival": locked_metrics["gt_survival"],
            "rc_survival": locked_metrics["rc_survival"],
            "objective": objective(locked_metrics, locked_shape),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        OUT / "locked_test.npz",
        targets=locked["targets"],
        profiles=locked["profiles"],
        seeds=locked["seeds"],
        warmup=locked["warmup"],
        future=locked["future"],
        prediction=locked_prediction,
        rc_events=locked_events,
    )
    plot_bo(records)
    plot_survival(FIG, locked["targets"], locked_metrics)
    plot_gt_rc_grid(
        FIG,
        locked["warmup"],
        locked["future"],
        locked_prediction,
        locked["profiles"],
        DEFAULT_EVENT_THRESHOLD,
        DEFAULT_EVENT_BLANKING_S,
    )
    (OUT / "RESULTS.md").write_text(
        "# Three-node dynamics-aware SCDT BO\n\n"
        "The normal-only SCDT readout was unable to reproduce a sustained low "
        "seizure state. This experiment adds 12 postcritical training records "
        "under the extrapolated normal-branch statistical coordinate and uses "
        "a 40-trial BO objective containing normal morphology, normal spike "
        "rate, five-second sustained-event classification, drop depth, low-state "
        "occupancy, duration, and clipping.\n\n"
        f"- Selected trial / mode: {winner_index + 1} / {winner_mode}\n"
        f"- Locked morphology score: {locked_metrics['morphology_score']:.4f}\n"
        f"- Locked normal spike-rate MAE: {locked_metrics['normal_spike_rate_mae']:.4f}\n"
        f"- Locked sustained hits / misses / false alarms: {locked_shape['sustained_hits']} / {locked_shape['sustained_misses']} / {locked_shape['sustained_false_alarms']}\n"
        f"- Locked RC event q05 / occupancy / longest-run medians: {locked_shape['rc_event_q05_median']:.4f} / {locked_shape['rc_event_occupancy_median']:.4f} / {locked_shape['rc_event_run_median_s']:.2f} s\n"
        f"- Locked clipping fraction: {locked_metrics['clipping_fraction']:.6f}\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
