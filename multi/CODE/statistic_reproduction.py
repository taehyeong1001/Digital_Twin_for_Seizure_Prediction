#!/usr/bin/env python3
"""Per-rollout statistic reproduction of the locked three-node SCDT.

No model selection.  The locked model is rebuilt exactly as in
build_final_figures.py, the original sweep is replayed with the same seeds
from the first grid point up to the SCDT 50% crossing, and the conditioning
statistics of every autonomous rollout are remeasured (SCDT manuscript
Fig. 3a style check).
"""

from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from network_gt import RC_HZ, first_departure
from strict_multi_node_scdt import ROLLOUT_S, Experiment, RCParams, record_statistics

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = json.loads((ROOT / "results/summary.json").read_text())
    protocol = json.loads((ROOT / "results/protocol.json").read_text())
    args = protocol["arguments"]
    experiment = Experiment(Namespace(**{**args, "out_dir": ROOT}))
    experiment.prepare()
    print("Fitting locked 1500-node reservoir...", flush=True)
    model = experiment.build(RCParams(**summary["locked_parameters"]))

    with (ROOT / "data/scdt_sweep_seed733900.csv").open(newline="") as handle:
        sweep = list(csv.DictReader(handle))
    crossing = float(summary["scdt_primary"]["lead_50"])
    lead_values = np.asarray([float(r["lead_input"]) for r in sweep])
    # The sweep moves the leading statistic downward; replay up to the first point past the crossing.
    last = int(np.flatnonzero(lead_values >= crossing - 1e-9).max()) + 1

    count = int(args["rc_ensemble"])
    out_rows = []
    for grid_index in range(last + 1):
        row = sweep[grid_index]
        lead, second = float(row["lead_input"]), float(row["second_input"])
        warm, warm_static = experiment.warmups(lead, count)
        target = experiment.norm_stat(np.asarray([lead, second]))
        pred_n, _ = model.rollout(warm, warm_static, np.repeat(target[None], count, axis=0),
                                  int(round(ROLLOUT_S * RC_HZ)), int(args["seed"]) + 300000 + grid_index)
        prediction = experiment.denorm(pred_n)
        times, _ = first_departure(prediction)
        departed = np.isfinite(times)
        if int(np.sum(~departed)) != int(row["survivors"]):
            print(f"WARNING grid {grid_index}: {int(np.sum(~departed))} survivors vs {row['survivors']} in the original sweep", flush=True)
        for member in range(count):
            stats = record_statistics(prediction[:, member])[experiment.idx]
            out_rows.append({
                "grid_index": grid_index, "member": member,
                "lead_input": lead, "second_input": second, "departed": bool(departed[member]),
                "lead_reproduced": float(stats[0]), "second_reproduced": float(stats[1]),
            })
        print(f"grid {grid_index:02d} {experiment.names[0]}={lead:.4f}: {int(np.sum(~departed))}/{count} survive", flush=True)

    out = ROOT / "data/statistic_reproduction.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
