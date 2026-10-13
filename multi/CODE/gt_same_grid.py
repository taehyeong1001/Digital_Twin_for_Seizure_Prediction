#!/usr/bin/env python3
"""GT survival on the same 36-point grid as the locked SCDT sweep.

Each SCDT sweep coordinate (leading statistic) is mapped to λ with the validation-only
calibration fit (lead = a·λ + b). GT is then simulated at that λ with the same protocol
as the GT boundary: 600 s preparation at PREP_LAM, switch, 200 s, 30 trajectories,
any-node trough rule after 5 s.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from network_gt import N_NODES, BatchNetwork, first_departure
from strict_multi_node_scdt import PREP_LAM, ROLLOUT_S, TRANSIENT_S, crossing, write_csv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = json.loads((ROOT / "results/summary.json").read_text())
    args = json.loads((ROOT / "results/protocol.json").read_text())["arguments"]
    a, b, _ = summary["alignment_validation_only"]["lead_vs_lam"]
    with (ROOT / "data/scdt_sweep_seed733900.csv").open(newline="") as handle:
        lead = np.asarray([float(r["lead_input"]) for r in csv.DictReader(handle)])
    lam = (lead - b) / a
    count = int(args["gt_ensemble"])
    seed = int(args["seed"]) + 500000
    net = BatchNetwork(count, seed)
    prepared = net.initial_state()
    prepared, _ = net.run(prepared, PREP_LAM, TRANSIENT_S, record=False)
    rows = []
    for index, (x, value) in enumerate(zip(lead, lam)):
        net.rng = np.random.default_rng(seed + 3000 + index)
        _, proxy = net.run(prepared.copy(), float(value), ROLLOUT_S)
        times, nodes = first_departure(proxy)
        survivors = int(np.sum(~np.isfinite(times)))
        rows.append({
            "lead_input": float(x), "lam": float(value), "survivors": survivors, "ensemble": count,
            "survival_rate": survivors / count,
            "median_onset_s": float(np.nanmedian(times)) if np.isfinite(times).any() else float("nan"),
            "first_node_counts": json.dumps(np.bincount(nodes[nodes >= 0], minlength=N_NODES).tolist()),
        })
        print(f"GT {index + 1:02d}/{len(lam)} lead={x:.4f} lam={value:.4f}: {survivors}/{count}", flush=True)
    write_csv(ROOT / "data/gt_same_grid.csv", rows)
    s = np.asarray([r["survival_rate"] for r in rows])
    lead50 = crossing(lead, s)
    out = {"gt_same_grid_lead_50": lead50, "gt_same_grid_lam_50": (lead50 - b) / a,
           "gt_fine_lam_50": summary["gt"].get("lam_50"), "gt_fine_lead_50": summary["gt"]["lead_50_mapped"]}
    (ROOT / "results/gt_same_grid.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
