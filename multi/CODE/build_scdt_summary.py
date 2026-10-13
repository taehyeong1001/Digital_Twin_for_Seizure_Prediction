#!/usr/bin/env python3
"""SCDT-paper summary figure: (a) GT vs SCDT survival, (b) leading statistic vs hidden λ."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from build_final_figures import LABEL, OUT, PAPER_BLACK, PAPER_BLUE, PAPER_RED, ROOT, names, rows, save, style


def main() -> None:
    style()
    summary = json.loads((ROOT / "results/summary.json").read_text())
    lead = names(summary)[0]
    sweep = rows(ROOT / "data/scdt_sweep_seed733900.csv")
    gt = rows(ROOT / "data/gt_same_grid.csv")
    x = np.asarray([float(r["lead_input"]) for r in sweep])
    y = 100 * np.asarray([float(r["survival_rate"]) for r in sweep])
    gy = 100 * np.asarray([float(r["survival_rate"]) for r in gt])
    a, b, _ = summary["alignment_validation_only"]["lead_vs_lam"]

    fig = plt.figure(figsize=(13.5, 4.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.35, 1], wspace=.28)
    ax = fig.add_subplot(grid[0])
    ax.plot(x, gy, color=PAPER_BLACK, lw=1.6, marker="o", ms=5, label="GT")
    ax.plot(x, y, color=PAPER_BLUE, lw=1.6, marker="s", ms=5, label="SCDT")
    ax.axvline(summary["scdt_primary"]["lead_50"], color=PAPER_RED, ls="--", lw=2, label="estimated critical point")
    ax.set(xlabel=LABEL.get(lead, lead), ylabel="survival rate (%)", ylim=(-3, 103))
    ax.invert_xaxis()
    top = ax.secondary_xaxis("top", functions=(lambda s: (s - b) / a, lambda l: a * l + b))
    top.set_xlabel(r"$\lambda$")
    top.tick_params(direction="in")
    ax.legend(loc="upper right")

    bx = fig.add_subplot(grid[1])
    cohort = rows(ROOT / "data/statistics_cohort.csv")
    lam = np.asarray([float(r["lam"]) for r in cohort])
    value = np.asarray([float(r[lead]) for r in cohort])
    levels = np.unique(lam)
    mean = np.asarray([value[lam == v].mean() for v in levels])
    slope, intercept = np.polyfit(levels, mean, 1)
    bx.plot(levels, slope * levels + intercept, color=PAPER_RED, ls=(0, (5, 2, 1, 2)), lw=1.6, zorder=1)
    bx.plot(levels, mean, "o", ms=7, color=PAPER_BLUE, zorder=2)
    bx.set(xlabel=r"$\lambda$", ylabel=LABEL.get(lead, lead))

    fig.text(.07, .93, "(a)", fontsize=16)
    fig.text(.66, .93, "(b)", fontsize=16)
    fig.suptitle("Epileptor Network (three nodes)", fontsize=17, y=1.03)
    save(fig, "00_scdt_summary.png")


if __name__ == "__main__":
    main()
