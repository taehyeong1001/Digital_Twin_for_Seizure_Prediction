#!/usr/bin/env python3
"""Per-rollout statistic reproduction of the locked single-node SCDT.

No model selection.  The locked model is rebuilt from the saved training
records and parameters, the original sweep is replayed with the same seeds
from the first grid point up to the SCDT 50% crossing, and the statistics of
every autonomous rollout are remeasured (SCDT manuscript Fig. 3a style check).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_single_node_scrc import RCParams
from strict_single_node_scdt import build_rc, event_mask, fit_morphology_reference, statistics

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    protocol = json.loads((ROOT / "results" / "protocol.json").read_text())
    args = protocol["arguments"]
    saved = np.load(ROOT / "data" / "training_signals.npz")
    signals = np.asarray(saved["signals"], dtype=float)
    statics = np.asarray(saved["statistics"], dtype=float)
    sample_hz = float(saved["sample_hz"])
    dynamic_center = float(np.mean(signals))
    dynamic_scale = float(np.std(signals))
    static_center = np.mean(statics, axis=0)
    static_scale = np.std(statics, axis=0)
    static_scale = np.where(static_scale > 1e-9, static_scale, 1.0)
    sequences_norm = [(signal - dynamic_center) / dynamic_scale for signal in signals]
    statics_norm = (statics - static_center) / static_scale
    morphology_reference = fit_morphology_reference([signal for signal in signals], sample_hz)
    rc = build_rc(RCParams(**protocol["best_rc_parameters"]), sequences_norm, statics_norm, int(args["fit_warmup"]))

    sweep = pd.read_csv(ROOT / "data" / "scdt_rate_sweep.csv")
    crossing = float(protocol["thresholds"]["rc_sigma_50"])
    last = int(np.flatnonzero(sweep["sigma_input"].to_numpy() <= crossing + 1e-9).max())
    path = protocol["statistical_path"]
    ensemble = int(args["rc_ensemble"])
    warmup_samples = int(args["warmup_samples"])
    rollout_samples = int(round(float(args["rollout_s"]) * sample_hz))
    indices = np.arange(ensemble) % len(sequences_norm)

    rows = []
    for grid_index in range(last + 1):
        sigma = float(sweep["sigma_input"].iloc[grid_index])
        raw = np.asarray([sigma, path["slope"] * sigma + path["intercept"]])
        target = (raw - static_center) / static_scale
        warmups = np.column_stack([sequences_norm[i][-warmup_samples:] for i in indices])
        pred_norm = rc.rollout_switch_batch(
            warmups, statics_norm[indices], np.repeat(target[None, :], ensemble, axis=0),
            rollout_samples, int(args["seed"]) + 300000 + grid_index,
        )
        pred = pred_norm * dynamic_scale + dynamic_center
        events = event_mask(pred, sample_hz, morphology_reference)
        expected = int(sweep["survivors"].iloc[grid_index])
        if int(np.sum(~events)) != expected:
            print(f"WARNING grid {grid_index}: {int(np.sum(~events))} survivors vs {expected} in the original sweep", flush=True)
        for member in range(ensemble):
            generated = statistics(pred[:, member], sample_hz)
            rows.append({
                "grid_index": grid_index, "member": member,
                "sigma_input": sigma, "pe_input": float(raw[1]),
                "departed": bool(events[member]),
                "sigma_reproduced": float(generated[0]), "pe_reproduced": float(generated[1]),
            })
        print(f"grid {grid_index:02d} sigma={sigma:.4f}: {int(np.sum(~events))}/{ensemble} survive", flush=True)

    out = ROOT / "data" / "statistic_reproduction.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
