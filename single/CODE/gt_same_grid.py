#!/usr/bin/env python3
"""Single-node GT survival on the same 35 σ coordinates as the SCDT sweep.

The SCDT sweep is uniform in σ, while the original GT grid is non-uniform in
x0 (0.01 / 0.001 / 0.02 steps).  Each SCDT σ coordinate is mapped back to x0
through the validation-only calibration line, and 30 GT trajectories are
simulated there with the original protocol (warm-up at x0 = -2.18, switch,
140-s future) and the same departure rules (original rule and median guard).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from strict_single_node_scdt import (
    crossing,
    event_mask,
    fit_morphology_reference,
    simulate_switch_batch,
    write_csv,
)
from waveform_guard import fit_median_reference, high_median_event_mask

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    protocol = json.loads((ROOT / "results/protocol.json").read_text())
    args = protocol["arguments"]
    base_seed = int(args["seed"])
    sample_hz = 256.0 / int(args["stride"])
    with np.load(ROOT / "data/training_signals.npz") as archive:
        training = [signal for signal in archive["signals"]]
    morphology = fit_morphology_reference(training, sample_hz)
    median = fit_median_reference(training, sample_hz)
    mapping = protocol["x0_sigma_alignment_validation_only"]
    a, b = float(mapping["slope"]), float(mapping["intercept"])

    with (ROOT / "data/scdt_rate_sweep_median_guard.csv").open(newline="") as handle:
        sigma_grid = [float(row["sigma_input"]) for row in csv.DictReader(handle)]
    ensemble = int(args["gt_ensemble"])

    rows = []
    for index, sigma in enumerate(sigma_grid):
        x0 = (sigma - b) / a
        _, future = simulate_switch_batch(
            x0, ensemble, -2.18, float(args["warmup_s"]), float(args["rollout_s"]),
            base_seed + 800000 + index * 1000,
        )
        sampled = future[:: int(args["stride"])]
        old = event_mask(sampled, sample_hz, morphology)
        guarded = old | high_median_event_mask(sampled, median)
        rows.append({
            "sigma_input": sigma, "x0": x0, "ensemble": ensemble,
            "survivors_original_rule": int(np.sum(~old)),
            "survivors": int(np.sum(~guarded)),
            "survival_rate": float(np.mean(~guarded)),
        })
        print(f"GT {index + 1:02d}/{len(sigma_grid)} sigma={sigma:.4f} (x0={x0:.4f}): "
              f"{rows[-1]['survivors']}/{ensemble} survive (original rule {rows[-1]['survivors_original_rule']})", flush=True)

    write_csv(ROOT / "data/gt_same_grid.csv", rows)
    sigma = np.asarray([r["sigma_input"] for r in rows])
    result = {
        "grid": "SCDT sweep σ coordinates, mapped to x0 by the validation-only calibration",
        "points": len(rows),
        "gt_sigma_50_same_grid": crossing(sigma, np.asarray([r["survival_rate"] for r in rows])),
        "gt_sigma_50_same_grid_original_rule": crossing(
            sigma, np.asarray([r["survivors_original_rule"] / ensemble for r in rows])),
    }
    (ROOT / "results/gt_same_grid.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
