#!/usr/bin/env python3
"""SCDT-paper Fig. 3(a) style: reproduced vs supplied leading statistic along the conditioning path (surviving rollouts)."""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from build_final_figures import LABEL, PAPER_BLUE, ROOT, rows, save, style

DOT_BLUE = "#8fb3d9"


def main() -> None:
    style()
    summary = json.loads((ROOT / "results/summary.json").read_text())
    name = LABEL.get(summary["statistics"]["leading"], summary["statistics"]["leading"])
    data = rows(ROOT / "data/statistic_reproduction.csv")
    keep = [r for r in data if r["departed"] == "False"]
    supplied = np.asarray([float(r["lead_input"]) for r in keep])
    reproduced = np.asarray([float(r["lead_reproduced"]) for r in keep])
    levels = np.unique(supplied)
    means = np.asarray([reproduced[supplied == v].mean() for v in levels])
    train_lo, train_hi = summary["training_lead_range"]
    lead_col = summary["statistics"]["leading"]
    train_values = np.asarray([float(r[lead_col]) for r in rows(ROOT / "data/training_statistics.csv")])
    training_only = "--training-only" in sys.argv
    if training_only:
        inside = (supplied >= train_lo - 1e-9) & (supplied <= train_hi + 1e-9)
        supplied, reproduced = supplied[inside], reproduced[inside]
        levels = np.unique(supplied)
        means = np.asarray([reproduced[supplied == v].mean() for v in levels])

    fig, ax = plt.subplots(figsize=(5.0, 4.7))
    xpad = 0.05 * np.ptp(supplied)
    ypad = 0.05 * np.ptp(reproduced)
    xlim = (supplied.min() - xpad, supplied.max() + xpad)
    ylim = (reproduced.min() - ypad, reproduced.max() + ypad)
    lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
    if not training_only:
        ax.axvspan(train_lo, train_hi, color="0.5", alpha=0.10, lw=0, label="training range")
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.45", lw=1.1, label=r"$y = x$")
    ax.plot(supplied, reproduced, "o", ms=3.2, color=DOT_BLUE, alpha=0.65, mec="none", label="surviving rollout")
    ax.plot(levels, means, "s", ms=5, color=PAPER_BLUE, label="condition mean")
    ax.plot(train_values, np.full(len(train_values), 0.025), "|", ms=11, mew=1.2, color="#b5413c",
            transform=ax.get_xaxis_transform(), clip_on=True, label="training record")
    ax.set(xlim=xlim, ylim=ylim,
           xlabel=f"Supplied {name}", ylabel=f"Reproduced {name}")
    ax.legend(loc="upper left", fontsize=9.5, handletextpad=0.4)
    ax.set_title("Epileptor network (3 nodes)", fontsize=14)
    save(fig, "05_statistic_reproduction_training.png" if training_only else "05_statistic_reproduction.png")

    slope = float(np.polyfit(supplied, reproduced, 1)[0])
    second_in = np.asarray([float(r["second_input"]) for r in keep])
    second_out = np.asarray([float(r["second_reproduced"]) for r in keep])
    print(json.dumps({"rollouts": int(len(supplied)), "conditions": int(len(levels)), "slope": slope,
                      "corr": float(np.corrcoef(supplied, reproduced)[0, 1]),
                      "mean_abs_error": float(np.mean(np.abs(reproduced - supplied))),
                      "second_supplied_range": [float(second_in.min()), float(second_in.max())],
                      "second_reproduced_mean": float(second_out.mean())}, indent=2))


if __name__ == "__main__":
    main()
