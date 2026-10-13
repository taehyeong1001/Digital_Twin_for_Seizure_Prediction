#!/usr/bin/env python3
"""Validation-only statistics cohort for the statistic-vs-parameter figure.

Twelve evenly spaced x0 levels over the normal range, 20 event-free 160 s records each
(same generator, sampling and normal rule as the training records, different seeds).
Never used for training, selection or thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from strict_single_node_scdt import event_mask, simulate_fixed_batch, statistics, write_csv

ROOT = Path(__file__).resolve().parents[1]
LEVELS = np.round(np.linspace(-2.30, -2.065, 12), 5)
REPLICATES = 20
SEED = 20260929 + 700000


def main() -> None:
    args = json.loads((ROOT / "results/protocol.json").read_text()).get("arguments", {})
    duration = float(args.get("duration_s", 160.0))
    stride = int(args.get("stride", 8))
    hz = 256.0 / stride
    rows = []
    for index, x0 in enumerate(LEVELS):
        kept, attempt = [], 0
        while len(kept) < REPLICATES and attempt < 10:
            batch = simulate_fixed_batch(float(x0), REPLICATES, duration, SEED + index * 1000 + attempt * 100003)[::stride]
            ok = ~event_mask(batch, hz)
            kept.extend(batch[:, j] for j in np.flatnonzero(ok))
            attempt += 1
        for signal in kept[:REPLICATES]:
            sigma, pe = statistics(signal, hz)
            rows.append({"x0": float(x0), "sigma": float(sigma), "permutation_entropy": float(pe)})
        print(f"x0={x0:.4f}: {min(len(kept), REPLICATES)} records", flush=True)
    write_csv(ROOT / "data/statistics_cohort.csv", rows)


if __name__ == "__main__":
    main()
