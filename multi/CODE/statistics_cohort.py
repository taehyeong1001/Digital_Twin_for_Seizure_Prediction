#!/usr/bin/env python3
"""Validation-only statistics cohort for the statistic-vs-parameter figure.

Twelve evenly spaced λ levels over the normal range, 20 event-free records each
(600 s transient, 160 s record; same generator and rejection rule as the training
records, different seeds). Never used for training, selection or thresholds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from strict_multi_node_scdt import STAT_NAMES, normal_records, record_statistics, write_csv

ROOT = Path(__file__).resolve().parents[1]
LEVELS = np.round(np.linspace(-2.14, -2.045, 12), 5)
REPLICATES = 20
SEED = 20260924 + 900000


def main() -> None:
    proxy, lam = normal_records(LEVELS, REPLICATES, SEED)
    rows = []
    for j in range(proxy.shape[1]):
        stats = record_statistics(proxy[:, j])
        rows.append({"lam": float(lam[j]), **{name: float(v) for name, v in zip(STAT_NAMES, stats)}})
    write_csv(ROOT / "data/statistics_cohort.csv", rows)
    print(f"saved {len(rows)} records at {len(LEVELS)} levels")


if __name__ == "__main__":
    main()
