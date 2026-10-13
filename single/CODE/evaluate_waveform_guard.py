#!/usr/bin/env python3
"""Exploratory normal-waveform guard on locked single-node SCDT."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from paper_single_node_scrc import RCParams
from waveform_guard import (
    fit_median_reference,
    fit_periodicity_reference,
    high_median_event_mask,
    periodicity_event_mask,
)
from strict_single_node_scdt import (
    GT_X0,
    VALIDATION_X0,
    build_rc,
    crossing,
    event_mask,
    fit_morphology_reference,
    normal_record,
    plot_clean_survival,
    simulate_switch_batch,
    statistics,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    with (ROOT / "results/protocol.json").open() as handle:
        protocol = json.load(handle)
    args = protocol["arguments"]
    base_seed = int(args["seed"])
    sample_hz = 256.0 / int(args["stride"])
    with np.load(ROOT / "data/training_signals.npz") as archive:
        training = [signal for signal in archive["signals"]]
        train_stats = archive["statistics"]
    periodicity = fit_periodicity_reference(training, sample_hz)
    median = fit_median_reference(training, sample_hz)
    morphology = fit_morphology_reference(training, sample_hz)
    print(
        f"training-only thresholds: periodicity={periodicity.threshold:.6f}, "
        f"10-s median={median.threshold:.6f}", flush=True,
    )

    dynamic_values = np.concatenate(training)
    dynamic_center = float(np.mean(dynamic_values))
    dynamic_scale = float(np.std(dynamic_values))
    static_center = np.mean(train_stats, axis=0)
    static_scale = np.std(train_stats, axis=0)
    static_scale = np.where(static_scale > 1e-9, static_scale, 1.0)
    sequences_norm = [(signal - dynamic_center) / dynamic_scale for signal in training]
    statics_norm = (train_stats - static_center) / static_scale
    model = build_rc(
        RCParams(**protocol["best_rc_parameters"]), sequences_norm,
        statics_norm, int(args["fit_warmup"]),
    )

    # Recreate the exact normal-validation cohort used by the original BO.
    gt_validation = []
    rc_validation = []
    for index, x0 in enumerate(VALIDATION_X0):
        signal = normal_record(
            float(x0), float(args["duration_s"]), int(args["stride"]),
            base_seed + 200000 + index * 1000,
        )
        gt_validation.append(signal)
        raw = statistics(signal, sample_hz)
        target = (raw - static_center) / static_scale
        nearest = int(np.argmin(np.abs(statics_norm[:, 0] - target[0])))
        repeats = 3
        warm = np.repeat(
            sequences_norm[nearest][-int(args["warmup_samples"]):, None],
            repeats, axis=1,
        )
        pred_n = model.rollout_switch_batch(
            warm,
            np.repeat(statics_norm[nearest][None, :], repeats, axis=0),
            np.repeat(target[None, :], repeats, axis=0),
            int(args["validation_rollout_samples"]),
            base_seed + 8000 + index,
        )
        rc_validation.append(pred_n * dynamic_scale + dynamic_center)
    gt_validation_periodic = periodicity_event_mask(
        np.column_stack(gt_validation), periodicity
    )
    gt_validation_high_median = high_median_event_mask(
        np.column_stack(gt_validation), median
    )
    rc_validation_old = np.concatenate([
        event_mask(pred, sample_hz, morphology) for pred in rc_validation
    ])
    rc_validation_periodic = np.concatenate([
        periodicity_event_mask(pred, periodicity) for pred in rc_validation
    ])
    rc_validation_high_median = np.concatenate([
        high_median_event_mask(pred, median) for pred in rc_validation
    ])
    print(
        f"held-out normals: GT new {sum(gt_validation_periodic | gt_validation_high_median)}/6, "
        f"RC old {sum(rc_validation_old)}/18, "
        f"RC combined {sum(rc_validation_old | rc_validation_periodic | rc_validation_high_median)}/18",
        flush=True,
    )

    sweep_original = read_csv(ROOT / "data/scdt_rate_sweep.csv")
    batch_size = int(args["rc_ensemble"])
    indices = np.arange(batch_size) % len(sequences_norm)
    warmups = np.column_stack([
        sequences_norm[index][-int(args["warmup_samples"]):] for index in indices
    ])
    warmup_statics = statics_norm[indices]
    sweep_new = []
    steps = int(round(float(args["rollout_s"]) * sample_hz))
    for grid_index, row in enumerate(sweep_original):
        raw = np.asarray([
            float(row["sigma_input"]), float(row["permutation_entropy_input"])
        ])
        target = (raw - static_center) / static_scale
        pred_n = model.rollout_switch_batch(
            warmups, warmup_statics,
            np.repeat(target[None, :], batch_size, axis=0),
            steps, base_seed + 300000 + grid_index,
        )
        prediction = pred_n * dynamic_scale + dynamic_center
        old = event_mask(prediction, sample_hz, morphology)
        periodic = periodicity_event_mask(prediction, periodicity)
        high_median = high_median_event_mask(prediction, median)
        old_survivors = int(np.sum(~old))
        assert old_survivors == int(row["survivors"])
        new_survivors = int(np.sum(~(old | periodic | high_median)))
        sweep_new.append({
            "sigma_input": float(raw[0]),
            "old_survivors": old_survivors,
            "periodic_departures": int(np.sum(periodic)),
            "new_periodic_only_departures": int(np.sum(periodic & ~old)),
            "high_median_departures": int(np.sum(high_median)),
            "new_high_median_only_departures": int(np.sum(high_median & ~old)),
            "survivors": new_survivors,
            "ensemble": batch_size,
            "survival_rate": new_survivors / batch_size,
        })
        print(
            f"SCDT {grid_index + 1:02d}/35 sigma={raw[0]:.4f}: "
            f"{old_survivors} -> {new_survivors} survivors",
            flush=True,
        )

    gt_original = read_csv(ROOT / "data/gt_boundary_extended.csv")
    original_fine = read_csv(ROOT / "data/gt_boundary.csv")
    original_indices = {round(float(row["x0"]), 6): index for index, row in enumerate(original_fine)}
    full_indices = {round(float(x0), 6): index for index, x0 in enumerate(GT_X0)}
    gt_new = []
    for row in gt_original:
        key = round(float(row["x0"]), 6)
        if int(row["survivors"]) == 0:
            new_survivors = 0  # An OR-ed detector cannot restore departed trajectories.
            periodic_count = 0
            median_count = 0
        else:
            if row["cohort"] == "original_fine":
                seed = base_seed + 500000 + original_indices[key] * 1000
            else:
                seed = base_seed + 700000 + full_indices[key] * 1000
            _, future = simulate_switch_batch(
                key, int(row["ensemble"]), -2.18,
                float(args["warmup_s"]), float(args["rollout_s"]), seed,
            )
            sampled = future[::int(args["stride"])]
            old = event_mask(sampled, sample_hz, morphology)
            periodic = periodicity_event_mask(sampled, periodicity)
            high_median = high_median_event_mask(sampled, median)
            assert int(np.sum(~old)) == int(row["survivors"])
            new_survivors = int(np.sum(~(old | periodic | high_median)))
            periodic_count = int(np.sum(periodic & ~old))
            median_count = int(np.sum(high_median & ~old))
        gt_new.append({
            "x0": key,
            "sigma_coordinate": float(row["sigma_coordinate"]),
            "old_survivors": int(row["survivors"]),
            "new_periodic_only_departures": periodic_count,
            "new_high_median_only_departures": median_count,
            "survivors": new_survivors,
            "ensemble": int(row["ensemble"]),
            "survival_rate": new_survivors / int(row["ensemble"]),
        })
        print(
            f"GT x0={key:.3f}: {row['survivors']} -> {new_survivors} survivors",
            flush=True,
        )

    write_csv(ROOT / "data/scdt_rate_sweep_waveform_guard.csv", sweep_new)
    write_csv(ROOT / "data/gt_boundary_waveform_guard.csv", gt_new)
    plot_clean_survival(
        ROOT / "figures/10_rate_with_waveform_guard.png", sweep_new, gt_new
    )
    rc_sigma = np.asarray([row["sigma_input"] for row in sweep_new])
    rc_survival = np.asarray([row["survival_rate"] for row in sweep_new])
    gt_x0 = np.asarray([row["x0"] for row in gt_new])
    gt_survival = np.asarray([row["survival_rate"] for row in gt_new])
    result = {
        "status": "exploratory post-hoc evaluation; not a new BO-selected model",
        "training_only_periodicity_reference": asdict(periodicity),
        "training_only_median_reference": asdict(median),
        "heldout_gt_normal_periodicity_false_events": int(np.sum(gt_validation_periodic)),
        "heldout_gt_normal_new_false_events": int(np.sum(gt_validation_periodic | gt_validation_high_median)),
        "heldout_gt_normal_total": len(gt_validation_periodic),
        "heldout_rc_normal_old_false_events": int(np.sum(rc_validation_old)),
        "heldout_rc_normal_new_false_events": int(np.sum(rc_validation_old | rc_validation_periodic | rc_validation_high_median)),
        "heldout_rc_normal_total": len(rc_validation_old),
        "old_rc_sigma_50": protocol["thresholds"]["rc_sigma_50"],
        "new_rc_sigma_50": crossing(rc_sigma, rc_survival),
        "old_gt_x0_50": protocol["thresholds"]["gt_x0_50"],
        "new_gt_x0_50": crossing(gt_x0, gt_survival),
        "rebound_index": 31,
        "rebound_old_survivors": sweep_new[31]["old_survivors"],
        "rebound_new_survivors": sweep_new[31]["survivors"],
    }
    with (ROOT / "results/waveform_guard.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
