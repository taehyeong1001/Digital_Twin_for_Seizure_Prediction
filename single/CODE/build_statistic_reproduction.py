#!/usr/bin/env python3
"""SCDT-paper Fig. 3(a) style: reproduced vs supplied σ along the conditioning path (surviving rollouts)."""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from build_final_figures import PAPER_BLUE, ROOT, rows, save, style

DOT_BLUE = "#8fb3d9"


def main() -> None:
    style()
    data = rows(ROOT / "data/statistic_reproduction.csv")
    keep = [r for r in data if r["departed"] == "False"]
    supplied = np.asarray([float(r["sigma_input"]) for r in keep])
    reproduced = np.asarray([float(r["sigma_reproduced"]) for r in keep])
    levels = np.unique(supplied)
    means = np.asarray([reproduced[supplied == v].mean() for v in levels])
    train = np.load(ROOT / "data/training_signals.npz")["statistics"][:, 0]
    training_only = "--training-only" in sys.argv
    if training_only:
        inside = (supplied >= train.min() - 1e-9) & (supplied <= train.max() + 1e-9)
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
        ax.axvspan(train.min(), train.max(), color="0.5", alpha=0.10, lw=0, label="training range")
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.45", lw=1.1, label=r"$y = x$")
    ax.plot(supplied, reproduced, "o", ms=3.2, color=DOT_BLUE, alpha=0.65, mec="none", label="surviving rollout")
    ax.plot(levels, means, "s", ms=5, color=PAPER_BLUE, label="condition mean")
    ax.plot(train, np.full(len(train), 0.025), "|", ms=11, mew=1.2, color="#b5413c",
            transform=ax.get_xaxis_transform(), clip_on=True, label="training record")
    ax.set(xlim=xlim, ylim=ylim,
           xlabel=r"Supplied $\sigma$", ylabel=r"Reproduced $\sigma$")
    ax.legend(loc="upper left", fontsize=9.5, handletextpad=0.4)
    ax.set_title("Epileptor (single node)", fontsize=14)
    save(fig, "05_statistic_reproduction_training.png" if training_only else "05_statistic_reproduction.png")

    slope = float(np.polyfit(supplied, reproduced, 1)[0])
    print(json.dumps({"rollouts": int(len(supplied)), "conditions": int(len(levels)), "slope": slope,
                      "corr": float(np.corrcoef(supplied, reproduced)[0, 1]),
                      "mean_abs_error": float(np.mean(np.abs(reproduced - supplied)))}, indent=2))


if __name__ == "__main__":
    main()
