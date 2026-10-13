#!/usr/bin/env python3
"""Compare node-0 extrema and departure after 600 s and 2400 s preparation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    600: ROOT / "data/node0_bifurcation_scan.npz",
    2400: ROOT / "data/node0_bifurcation_scan_prep2400.npz",
}
COLORS = {600: "#7b3294", 2400: "#008837"}


def load(duration: int) -> dict[str, np.ndarray]:
    with np.load(FILES[duration]) as archive:
        assert int(archive["preparation_s"]) == duration
        return {key: archive[f"fine_{key}"] for key in
                ("x0_0", "minimum", "maximum", "departed")}


def aggregate(part: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    x = part["x0_0"]
    grid = np.unique(x)
    minimum = np.asarray([np.median(part["minimum"][x == v]) for v in grid])
    maximum = np.asarray([np.median(part["maximum"][x == v]) for v in grid])
    rate = np.asarray([np.mean(part["departed"][x == v]) for v in grid])
    return grid, minimum, maximum, rate


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.4, 7.1), sharex=True,
                                  gridspec_kw={"height_ratios": [1.35, 1]})
    for duration in FILES:
        grid, minimum, maximum, rate = aggregate(load(duration))
        selected = (grid >= -2.05) & (grid <= -2.02)
        color = COLORS[duration]
        ax1.plot(grid[selected], minimum[selected], color=color, lw=1.9,
                 label=f"minimum · {duration} s wait")
        ax1.plot(grid[selected], maximum[selected], color=color, lw=1.2, ls="--",
                 label=f"maximum · {duration} s wait")
        ax2.plot(grid[selected], rate[selected], color=color, lw=1.8,
                 marker="o", ms=2.4, label=f"{duration} s wait")
    ax1.axhline(-.5, color="#b2182b", lw=1, ls=":", label="departure rule −0.5")
    ax1.set(ylabel=r"node 0 GT proxy extrema $x_2-x_1$",
            title="Effect of longer preparation at fixed $x_{0,0}$")
    ax2.set(xlabel=r"varied $x_{0,0}$ (other coordinates fixed)",
            ylabel="node 0 departure fraction", ylim=(-.04, 1.04),
            xlim=(-2.05, -2.02))
    ax1.legend(frameon=False, ncol=2, fontsize=8)
    ax2.legend(frameon=False, fontsize=9)
    for ax in (ax1, ax2):
        ax.grid(alpha=.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(.5, .018,
             r"$x_{0,1}=-2.12$, $x_{0,2}=-2.17$; 200 s measurement; "
             "16 stochastic copies/point; dense step 0.0005.",
             fontsize=8, color=".35", ha="center")
    fig.tight_layout(rect=(0, .035, 1, 1))
    output = ROOT / "explore/node0_wait_comparison.png"
    fig.savefig(output, dpi=230, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
