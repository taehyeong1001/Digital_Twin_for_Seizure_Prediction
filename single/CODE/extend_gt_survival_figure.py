#!/usr/bin/env python3
"""Extend held-out GT survival support and redraw the two-curve figure.

Reuses the locked SCDT sweep and the original fine GT boundary cohort.
Additional GT points are independent switch trajectories with the original
warmup, rollout, event rule, and training-only morphology reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from strict_single_node_scdt import (
    GT_X0,
    event_mask,
    fit_morphology_reference,
    plot_clean_survival,
    simulate_switch_batch,
    write_csv,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.out_dir.resolve()
    data_dir = root / "data"
    result_dir = root / "results"
    figure_dir = root / "figures"

    with (result_dir / "protocol.json").open() as handle:
        protocol = json.load(handle)
    experiment_args = protocol["arguments"]
    alignment = protocol["x0_sigma_alignment_validation_only"]
    original_rows = read_csv(data_dir / "gt_boundary.csv")
    existing = {round(float(row["x0"]), 6): row for row in original_rows}
    earlier_extension = data_dir / "gt_boundary_extended.csv"
    if earlier_extension.exists():
        existing.update({
            round(float(row["x0"]), 6): row for row in read_csv(earlier_extension)
        })
    sweep_rows = read_csv(data_dir / "scdt_rate_sweep.csv")

    with np.load(data_dir / "training_signals.npz") as archive:
        sample_hz = float(archive["sample_hz"])
        training = [signal for signal in archive["signals"]]
    morphology_reference = fit_morphology_reference(training, sample_hz)
    assert np.isclose(morphology_reference.std_floor, protocol["morphology_reference"]["std_floor"])

    new_seed_base = int(experiment_args["seed"]) + 700000
    count = int(experiment_args["gt_ensemble"])
    stride = int(experiment_args["stride"])
    expanded = []
    for index, x0 in enumerate(GT_X0):
        key = round(float(x0), 6)
        if key in existing:
            row = existing[key]
            expanded.append({
                "x0": float(row["x0"]),
                "sigma_coordinate": float(row["sigma_coordinate"]),
                "survivors": int(row["survivors"]),
                "ensemble": int(row["ensemble"]),
                "survival_rate": float(row["survival_rate"]),
                "cohort": row.get("cohort", "original_fine"),
            })
            continue
        _, future = simulate_switch_batch(
            target_x0=key,
            count=count,
            base_x0=-2.18,
            warmup_s=float(experiment_args["warmup_s"]),
            rollout_s=float(experiment_args["rollout_s"]),
            seed=new_seed_base + index * 1000,
        )
        survived = ~event_mask(future[::stride], sample_hz, morphology_reference)
        expanded.append({
            "x0": key,
            "sigma_coordinate": float(alignment["slope"] * key + alignment["intercept"]),
            "survivors": int(np.sum(survived)),
            "ensemble": count,
            "survival_rate": float(np.mean(survived)),
            "cohort": "added_coarse",
        })
        print(f"GT {index + 1}/{len(GT_X0)} x0={key:.3f}: survival={np.mean(survived):.3f}", flush=True)

    expanded.sort(key=lambda row: row["x0"])
    output_csv = data_dir / "gt_boundary_extended.csv"
    write_csv(output_csv, expanded)
    plot_clean_survival(figure_dir / "04_rate_threshold.png", sweep_rows, expanded)
    metadata = {
        "original_gt_file": "data/gt_boundary.csv",
        "extended_gt_file": "data/gt_boundary_extended.csv",
        "figure": "figures/04_rate_threshold.png",
        "x0_range": [float(GT_X0[0]), float(GT_X0[-1])],
        "x0_points": len(GT_X0),
        "original_points_reused": len(original_rows),
        "additional_seed_formula": "protocol.arguments.seed + 700000 + grid_index * 1000",
        "additional_seed_base": new_seed_base,
        "gt_ensemble_per_point": count,
        "statistical_alignment_caveat": "x0 to sigma is extrapolated beyond the normal-only calibration interval [-2.30, -2.065]",
        "rc_model_or_sweep_changed": False,
    }
    with (result_dir / "extended_gt_figure.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
