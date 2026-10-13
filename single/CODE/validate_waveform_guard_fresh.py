#!/usr/bin/env python3
"""Fresh normal-regime cohort for the exploratory waveform guard."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from paper_single_node_scrc import RCParams
from strict_single_node_scdt import (
    build_rc,
    event_mask,
    fit_morphology_reference,
    simulate_fixed_batch,
    statistics,
)
from waveform_guard import (
    fit_median_reference,
    fit_periodicity_reference,
    high_median_event_mask,
    periodicity_event_mask,
)


ROOT = Path(__file__).resolve().parents[1]
X0 = [-2.285, -2.255, -2.225, -2.195, -2.165, -2.135, -2.070, -2.065]


def main() -> None:
    with (ROOT / "results/protocol.json").open() as handle:
        protocol = json.load(handle)
    args = protocol["arguments"]
    stride = int(args["stride"])
    sample_hz = 256.0 / stride
    with np.load(ROOT / "data/training_signals.npz") as archive:
        training = [signal for signal in archive["signals"]]
        train_stats = archive["statistics"]
    dynamic = np.concatenate(training)
    center = float(np.mean(dynamic))
    scale = float(np.std(dynamic))
    static_center = np.mean(train_stats, axis=0)
    static_scale = np.std(train_stats, axis=0)
    normalized_training = [(signal - center) / scale for signal in training]
    normalized_statics = (train_stats - static_center) / static_scale
    model = build_rc(
        RCParams(**protocol["best_rc_parameters"]),
        normalized_training, normalized_statics, int(args["fit_warmup"]),
    )
    morphology = fit_morphology_reference(training, sample_hz)
    periodicity = fit_periodicity_reference(training, sample_hz)
    median = fit_median_reference(training, sample_hz)
    seed_base = int(args["seed"]) + 900000
    rows = []
    for x_index, x0 in enumerate(X0):
        gt = simulate_fixed_batch(
            float(x0), 5, float(args["duration_s"]), seed_base + x_index * 1000
        )[::stride]
        gt_old = event_mask(gt, sample_hz, morphology)
        gt_periodic = periodicity_event_mask(gt, periodicity)
        gt_median = high_median_event_mask(gt, median)
        rc_old_count = 0
        rc_new_count = 0
        rc_new_periodic_only = 0
        rc_new_median_only = 0
        for replicate in range(gt.shape[1]):
            raw = statistics(gt[:, replicate], sample_hz)
            target = (raw - static_center) / static_scale
            nearest = int(np.argmin(np.abs(normalized_statics[:, 0] - target[0])))
            warm = np.repeat(
                normalized_training[nearest][-int(args["warmup_samples"]):, None],
                3, axis=1,
            )
            pred_n = model.rollout_switch_batch(
                warm,
                np.repeat(normalized_statics[nearest][None, :], 3, axis=0),
                np.repeat(target[None, :], 3, axis=0),
                int(args["validation_rollout_samples"]),
                seed_base + 100000 + x_index * 1000 + replicate,
            )
            pred = pred_n * scale + center
            old = event_mask(pred, sample_hz, morphology)
            periodic = periodicity_event_mask(pred, periodicity)
            high_median = high_median_event_mask(pred, median)
            new = old | periodic | high_median
            rc_old_count += int(np.sum(old))
            rc_new_count += int(np.sum(new))
            rc_new_periodic_only += int(np.sum(periodic & ~old))
            rc_new_median_only += int(np.sum(high_median & ~old))
        row = {
            "x0": x0,
            "gt_old_departures": int(np.sum(gt_old)),
            "gt_new_departures": int(np.sum(gt_old | gt_periodic | gt_median)),
            "gt_total": gt.shape[1],
            "rc_old_departures": rc_old_count,
            "rc_median_only_departures": rc_old_count + rc_new_median_only,
            "rc_new_departures": rc_new_count,
            "rc_new_periodic_only": rc_new_periodic_only,
            "rc_new_high_median_only": rc_new_median_only,
            "rc_total": gt.shape[1] * 3,
        }
        rows.append(row)
        print(row, flush=True)

    with (ROOT / "data/scdt_rate_sweep.csv").open(newline="") as handle:
        sweep_rows = list(csv.DictReader(handle))
    sweep_checks = []
    for grid_index in [7, 16, 29, 30, 31, 32]:
        row = sweep_rows[grid_index]
        raw = np.asarray([
            float(row["sigma_input"]), float(row["permutation_entropy_input"])
        ])
        target = (raw - static_center) / static_scale
        indices = np.arange(30) % len(normalized_training)
        warm = np.column_stack([
            normalized_training[index][-int(args["warmup_samples"]):]
            for index in indices
        ])
        pred_n = model.rollout_switch_batch(
            warm,
            normalized_statics[indices],
            np.repeat(target[None, :], 30, axis=0),
            int(round(float(args["rollout_s"]) * sample_hz)),
            seed_base + 500000 + grid_index,
        )
        pred = pred_n * scale + center
        old = event_mask(pred, sample_hz, morphology)
        median_only = old | high_median_event_mask(pred, median)
        new = median_only | periodicity_event_mask(pred, periodicity)
        check = {
            "grid_index": grid_index,
            "sigma": float(raw[0]),
            "old_survivors": int(np.sum(~old)),
            "median_only_survivors": int(np.sum(~median_only)),
            "new_survivors": int(np.sum(~new)),
            "ensemble": 30,
        }
        sweep_checks.append(check)
        print(check, flush=True)
    result = {
        "cohort": "fresh normal-regime fixed-x0 GT records and independent RC innovations",
        "seed_base": seed_base,
        "rows": rows,
        "gt_old_departures": sum(row["gt_old_departures"] for row in rows),
        "gt_new_departures": sum(row["gt_new_departures"] for row in rows),
        "gt_total": sum(row["gt_total"] for row in rows),
        "rc_old_departures": sum(row["rc_old_departures"] for row in rows),
        "rc_median_only_departures": sum(row["rc_median_only_departures"] for row in rows),
        "rc_new_departures": sum(row["rc_new_departures"] for row in rows),
        "rc_total": sum(row["rc_total"] for row in rows),
        "fresh_rc_sweep_checks": sweep_checks,
    }
    with (ROOT / "results/waveform_guard_fresh_normal.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
