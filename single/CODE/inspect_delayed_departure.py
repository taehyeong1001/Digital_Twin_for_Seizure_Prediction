#!/usr/bin/env python3
"""Audit delayed trough crossings in the locked single-node SCDT sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_final_figures import ROOT, rc_prediction
from strict_single_node_scdt import EVENT_THRESHOLD


def main() -> None:
    protocol = json.loads((ROOT / "results/protocol.json").read_text())
    with (ROOT / "data/scdt_rate_sweep.csv").open(newline="") as handle:
        sweep = list(csv.DictReader(handle))
    sample_hz = 256.0 / int(protocol["arguments"]["stride"])
    rows = []
    signals = []
    for grid_index in [16, 17]:
        prediction, events = rc_prediction(protocol, grid_index)
        sigma = float(sweep[grid_index]["sigma_input"])
        for member in range(prediction.shape[1]):
            signal = prediction[:, member]
            crossing = np.flatnonzero(signal <= EVENT_THRESHOLD)
            if crossing.size == 0:
                continue
            onset_s = float(crossing[0] / sample_hz)
            if onset_s <= 10:
                continue
            pre = signal[:crossing[0]]
            row = {
                "grid_index": grid_index,
                "sigma": sigma,
                "member": member,
                "first_trough_s": onset_s,
                "pre_trough_min": float(np.min(pre)),
                "pre_trough_max": float(np.max(pre)),
                "pre_trough_median": float(np.median(pre)),
                "pre_trough_std": float(np.std(pre)),
                "pre_trough_fraction_above_5": float(np.mean(pre > 5)),
                "event_under_current_rule": bool(events[member]),
            }
            rows.append(row)
            signals.append(signal)

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.4 * len(rows)), sharex=True)
    time = np.arange(len(signals[0])) / sample_hz
    for axis, row, signal in zip(axes, rows, signals):
        axis.plot(time, signal, lw=.75, color="#2166ac")
        axis.axhline(EVENT_THRESHOLD, color="#b2182b", ls="--", lw=.8)
        axis.axvline(row["first_trough_s"], color="#b2182b", ls=":", lw=.8)
        axis.set_xlim(0, 140)
        axis.set_ylim(-6.5, 5.9)
        axis.set_ylabel("proxy")
        axis.set_title(
            f"grid {row['grid_index']}, member {row['member']}, "
            f"sigma={row['sigma']:.6f}, first trough={row['first_trough_s']:.1f}s, "
            f"pre-trough SD={row['pre_trough_std']:.2f}",
            fontsize=9, loc="left",
        )
        axis.grid(alpha=.15)
    axes[-1].set_xlabel("autonomous time after condition switch (s)")
    fig.tight_layout()
    out = ROOT / "archive/delayed_departure_candidates.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    (ROOT / "results/delayed_departure_candidates.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(out)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
