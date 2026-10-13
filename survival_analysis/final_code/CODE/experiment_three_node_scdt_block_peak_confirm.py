#!/usr/bin/env python3
"""Confirm peak-BO candidates on a second cohort, then run a locked test."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import experiment_three_node_scdt_block_peak_bo100 as exp
from experiment_three_node_scdt import (
    DEFAULT_EVENT_BLANKING_S,
    DEFAULT_EVENT_THRESHOLD,
    first_threshold_crossing,
    plot_gt_rc_grid,
    plot_survival,
    simulate_switch_cohort,
)
from experiment_three_node_scdt_block_innovation import plot_normal_zoom, profiles_from_maximum
from experiment_three_node_scdt_dynamics_pilot import shape_metrics
from paper_single_node_scrc import RCParams


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "three_node_scdt_block_peak_robust"
FIG = ROOT / "figures" / "RC" / "three_node_scdt_block_peak_robust"


def numeric(row: dict[str, str]) -> dict:
    return {key: value if key == "mode" else float(value) for key, value in row.items()}


def validation_candidates(rows: list[dict]) -> list[dict]:
    reference = rows[exp.PREVIOUS_TRIALS]
    result = []
    for row in rows[exp.PREVIOUS_TRIALS:]:
        normal_count = row["false_alarms"] + row["correct_nonexits"]
        sustained_normal = row["sustained_false_alarms"] + row["correct_nonexits"]
        if (
            row["survival_mae"] <= 0.05
            and row["curve_nonmonotonicity"] <= 1e-12
            and row["hits"] >= 0.85 * (row["hits"] + row["misses"])
            and row["false_alarms"] <= 0.06 * normal_count
            and row["sustained_hits"] >= 0.80 * (row["sustained_hits"] + row["sustained_misses"])
            and row["sustained_false_alarms"] <= 0.05 * sustained_normal
            and row["morphology_score"] <= 1.05
            and row["normal_spike_rate_mae"] <= reference["normal_spike_rate_mae"] + 1e-12
            and row["normal_peak_q995_mae"] <= reference["normal_peak_q995_mae"] + 1e-12
            and row["clipping_fraction"] <= 1e-12
        ):
            result.append(row)
    return sorted(result, key=lambda row: row["objective"])


def evaluate_candidate(row, cohort, coordinate, training, normalizers):
    train_raw, train_n, train_static, stable_count = training
    unit = np.asarray([row[f"unit_{index}"] for index in range(7)], dtype=float)
    params, block_s = exp.parameters_from_unit(unit)
    model, fit_info = exp.fit_block_model(
        params, block_s, train_n, train_raw, train_static, stable_count
    )
    metrics, prediction, rc_events = exp.evaluate_cohort(
        model, cohort, coordinate, normalizers, exp.SEED + 3200
    )
    shape = shape_metrics(cohort["future"], prediction)
    peak = exp.peak_metrics(cohort["future"], prediction)
    summary = {
        "trial": int(row["trial"]),
        **asdict(params),
        "block_s": block_s,
        **fit_info,
        **{key: value for key, value in metrics.items() if key not in ("gt_survival", "rc_survival")},
        "curve_nonmonotonicity": exp.curve_nonmonotonicity(metrics["rc_survival"]),
        **shape,
        **peak,
        "peak_score": float(
            metrics["morphology_score"]
            + metrics["normal_spike_rate_mae"] / 0.03
            + peak["normal_peak_q995_mae"] / 0.75
        ),
    }
    return model, summary, metrics, prediction, rc_events


def write_model(path, model, coordinate, normalizers):
    dynamic_center, dynamic_scale, static_center, static_scale = normalizers
    np.savez_compressed(
        path,
        a_data=model.a.data,
        a_indices=model.a.indices,
        a_indptr=model.a.indptr,
        a_shape=model.a.shape,
        w_in=model.w_in,
        w_static=model.w_static,
        w_out=model.w_out,
        residuals=model.residuals,
        residual_sequences=model.residual_sequences,
        residual_pool_indices=model.residual_pool_indices,
        readout_indices=model.readout_indices,
        dynamic_center=dynamic_center,
        dynamic_scale=dynamic_scale,
        static_center=static_center,
        static_scale=static_scale,
        clip_low=model.clip_low,
        clip_high=model.clip_high,
        coordinate_coef=coordinate.coef_,
        coordinate_intercept=coordinate.intercept_,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    coordinate, coordinate_metrics, train_raw, train_n, train_static, normalizers, stable_count = exp.prepare_training()
    training = train_raw, train_n, train_static, stable_count

    with (exp.OUT / "bo_history.csv").open(newline="") as handle:
        history = [numeric(row) for row in csv.DictReader(handle)]
    candidates = validation_candidates(history)

    with np.load(exp.OUT / "fresh_test.npz") as data:
        confirmation = {
            name: data[name]
            for name in ("targets", "maxima", "profiles", "seeds", "warmup", "future")
        }
    old_summary = json.loads((exp.OUT / "summary.json").read_text())
    baseline_confirmation = old_summary["paired_rate_bo100"]
    baseline_peak_score = float(
        baseline_confirmation["morphology_score"]
        + baseline_confirmation["normal_spike_rate_mae"] / 0.03
        + baseline_confirmation["normal_peak_q995_mae"] / 0.75
    )

    confirmation_path = OUT / "confirmation_metrics.csv"
    confirmed = []
    completed = set()
    if confirmation_path.exists():
        with confirmation_path.open(newline="") as handle:
            confirmed = [numeric(row) for row in csv.DictReader(handle)]
        completed = {int(row["trial"]) for row in confirmed}
    for row in candidates:
        if int(row["trial"]) in completed:
            continue
        _, summary, _, _, _ = evaluate_candidate(
            row, confirmation, coordinate, training, normalizers
        )
        confirmed.append(summary)
        with confirmation_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(confirmed[0]))
            writer.writeheader()
            writer.writerows(confirmed)
        print(json.dumps(summary), flush=True)

    strict = [
        row for row in confirmed
        if row["survival_mae"] <= 0.035
        and row["curve_nonmonotonicity"] <= 1e-12
        and row["false_alarms"] <= 4
        and row["hits"] >= 0.90 * (row["hits"] + row["misses"])
        and row["sustained_false_alarms"] <= 1
        and row["clipping_fraction"] <= 1e-12
        and row["peak_score"] < baseline_peak_score
    ]
    relaxed = [
        row for row in confirmed
        if row["survival_mae"] <= 0.05
        and row["curve_nonmonotonicity"] <= 1e-12
        and row["sustained_false_alarms"] <= 1
        and row["clipping_fraction"] <= 1e-12
        and row["peak_score"] < baseline_peak_score
    ]
    pool = strict or relaxed
    if not pool:
        raise RuntimeError("No peak-improved candidate preserved the confirmation rate curve")
    selected_confirmation = min(pool, key=lambda row: row["peak_score"])
    selected_history = next(
        row for row in history if int(row["trial"]) == int(selected_confirmation["trial"])
    )

    maxima = np.linspace(-2.50, -1.90, exp.TEST_LEVELS)
    test_targets = profiles_from_maximum(maxima, exp.SEED + 4000)
    test_seed_values = tuple(exp.SEED + 4100 + index for index in range(exp.TEST_SEEDS))
    test_warmup, test_future, test_profiles, test_seed_ids = simulate_switch_cohort(
        test_targets,
        test_seed_values,
        exp.WARMUP_S,
        exp.FUTURE_S,
        event_threshold=DEFAULT_EVENT_THRESHOLD,
    )
    test = {
        "targets": test_targets,
        "maxima": maxima,
        "profiles": test_profiles,
        "seeds": test_seed_ids,
        "warmup": test_warmup,
        "future": test_future,
    }
    winner, test_summary, test_metrics, test_prediction, test_rc_events = evaluate_candidate(
        selected_history, test, coordinate, training, normalizers
    )

    prior = json.loads((exp.PREVIOUS / "summary.json").read_text())
    baseline_params = RCParams(**prior["selected_parameters"])
    baseline, _ = exp.fit_block_model(
        baseline_params,
        float(prior["selected_block_s"]),
        train_n,
        train_raw,
        train_static,
        stable_count,
    )
    baseline_metrics, baseline_prediction, _ = exp.evaluate_cohort(
        baseline, test, coordinate, normalizers, exp.SEED + 3200
    )
    baseline_shape = shape_metrics(test_future, baseline_prediction)
    baseline_peak = exp.peak_metrics(test_future, baseline_prediction)
    baseline_test = {
        **{key: value for key, value in baseline_metrics.items() if key not in ("gt_survival", "rc_survival")},
        "curve_nonmonotonicity": exp.curve_nonmonotonicity(baseline_metrics["rc_survival"]),
        **baseline_shape,
        **baseline_peak,
        "gt_survival": baseline_metrics["gt_survival"],
        "rc_survival": baseline_metrics["rc_survival"],
    }

    result = {
        "status": "complete",
        "bo_observations": len(history),
        "confirmation_candidates": len(candidates),
        "strict_confirmation_candidates": len(strict),
        "selected_trial": int(selected_confirmation["trial"]),
        "selected_parameters": asdict(winner.params),
        "selected_block_s": float(selected_history["block_s"]),
        "coordinate_calibration": coordinate_metrics,
        "validation": selected_history,
        "confirmation": selected_confirmation,
        "fresh_test": {
            **test_summary,
            "gt_survival": test_metrics["gt_survival"],
            "rc_survival": test_metrics["rc_survival"],
        },
        "paired_rate_bo100": baseline_test,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    (OUT / "protocol.json").write_text(json.dumps({
        "date": "2026-09-12",
        "selection": "validation gate, independent confirmation rate gate, then minimum peak score",
        "confirmation": {"x0_range": [-2.50, -1.90], "levels": 33, "seeds": list(range(exp.SEED + 3100, exp.SEED + 3100 + exp.TEST_SEEDS))},
        "fresh_test": {"x0_range": [-2.50, -1.90], "levels": 33, "seeds": list(test_seed_values)},
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
    write_model(OUT / "winner_model.npz", winner, coordinate, normalizers)

    plot_survival(FIG, test_targets, test_metrics, "Robust peak-weighted block-BO")
    exp.plot_rate_comparison(
        test_targets,
        test_metrics["gt_survival"],
        baseline_metrics["rc_survival"],
        test_metrics["rc_survival"],
        figdir=FIG,
    )
    plot_gt_rc_grid(
        FIG, test_warmup, test_future, test_prediction, test_profiles,
        DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S,
    )
    gt_events = first_threshold_crossing(
        test_future, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S
    )
    plot_normal_zoom(
        test_future,
        {"rate-BO": baseline_prediction, "robust peak-BO": test_prediction},
        gt_events,
        figdir=FIG,
    )
    (OUT / "RESULTS.md").write_text(
        "# Robust peak-weighted block BO\n\n"
        f"Selected trial {int(selected_confirmation['trial'])} after confirming {len(candidates)} validation-feasible candidates on an independent cohort.\n\n"
        f"- Fresh point hits/misses/false alarms: {test_summary['hits']}/{test_summary['misses']}/{test_summary['false_alarms']}\n"
        f"- Fresh survival MAE / nonmonotonicity: {test_summary['survival_mae']:.4f} / {test_summary['curve_nonmonotonicity']:.4f}\n"
        f"- Fresh spike-rate / q99.5 peak / morphology errors: {test_summary['normal_spike_rate_mae']:.4f} / {test_summary['normal_peak_q995_mae']:.4f} / {test_summary['morphology_score']:.4f}\n"
        f"- Paired rate-BO survival MAE / nonmonotonicity: {baseline_test['survival_mae']:.4f} / {baseline_test['curve_nonmonotonicity']:.4f}\n"
        f"- Paired rate-BO spike-rate / q99.5 peak / morphology errors: {baseline_test['normal_spike_rate_mae']:.4f} / {baseline_test['normal_peak_q995_mae']:.4f} / {baseline_test['morphology_score']:.4f}\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
