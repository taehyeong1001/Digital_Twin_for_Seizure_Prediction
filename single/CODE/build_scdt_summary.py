#!/usr/bin/env python3
"""SCDT-paper summary figure: (a) GT vs SCDT survival, (b) leading statistic σ vs hidden x0."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from build_final_figures import PAPER_BLACK, PAPER_BLUE, PAPER_RED, ROOT, crossing50, rows, save, style


def main() -> None:
    style()
    protocol = json.loads((ROOT / "results/protocol.json").read_text())
    rc = rows(ROOT / "data/scdt_rate_sweep_median_guard.csv")
    gt = rows(ROOT / "data/gt_same_grid.csv")  # GT on the same 35 σ coordinates as the SCDT sweep
    rx = np.asarray([float(r["sigma_input"]) for r in rc])
    rs = np.asarray([float(r["survival_rate"]) for r in rc])
    gx = np.asarray([float(r["sigma_input"]) for r in gt])
    gs = np.asarray([float(r["survival_rate"]) for r in gt])
    order = np.argsort(gx)
    mapping = protocol["x0_sigma_alignment_validation_only"]
    a, b = mapping["slope"], mapping["intercept"]

    fig = plt.figure(figsize=(13.5, 4.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.35, 1], wspace=.28)
    ax = fig.add_subplot(grid[0])
    ax.plot(gx[order], 100 * gs[order], color=PAPER_BLACK, lw=1.6, marker="o", ms=5, label="GT")
    ax.plot(rx, 100 * rs, color=PAPER_BLUE, lw=1.6, marker="s", ms=5, label="SCDT")
    ax.axvline(crossing50(rx, rs), color=PAPER_RED, ls="--", lw=2, label="estimated critical point")
    ax.set(xlabel=r"std ($\sigma$)", ylabel="survival rate (%)", ylim=(-3, 103))
    top = ax.secondary_xaxis("top", functions=(lambda s: (s - b) / a, lambda x: a * x + b))
    top.set_xlabel(r"$x_0$")
    top.tick_params(direction="in")
    ax.legend(loc="upper right")

    bx = fig.add_subplot(grid[1])
    cohort = rows(ROOT / "data/statistics_cohort.csv")
    x0 = np.asarray([float(r["x0"]) for r in cohort])
    sigma = np.asarray([float(r["sigma"]) for r in cohort])
    levels = np.unique(x0)
    mean = np.asarray([sigma[x0 == v].mean() for v in levels])
    slope, intercept = np.polyfit(levels, mean, 1)
    bx.plot(levels, slope * levels + intercept, color=PAPER_RED, ls=(0, (5, 2, 1, 2)), lw=1.6, zorder=1)
    bx.plot(levels, mean, "o", ms=7, color=PAPER_BLUE, zorder=2)
    bx.set(xlabel=r"$x_0$", ylabel=r"std ($\sigma$)")

    fig.text(.07, .93, "(a)", fontsize=16)
    fig.text(.66, .93, "(b)", fontsize=16)
    fig.suptitle("Epileptor (single node)", fontsize=17, y=1.03)
    save(fig, "00_scdt_summary.png")


if __name__ == "__main__":
    main()
