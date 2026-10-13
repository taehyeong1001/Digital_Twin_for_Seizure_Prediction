#!/usr/bin/env python3
"""Build five curated figures for the locked strict three-node SCDT result."""

from __future__ import annotations

import csv
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from network_gt import BLANK_S, EVENT_THRESHOLD, RC_HZ, BatchNetwork, first_departure
from strict_multi_node_scdt import RCParams, Experiment


ROOT = Path(__file__).resolve().parents[1]
AICP = ROOT.parents[1]
OUT = ROOT / "figure"
COHORT_CACHE = ROOT / "data/final_trajectory_cohorts.npz"
BIFURCATION_CACHE = ROOT / "data/node0_bifurcation_scan_prep2400.npz"
RC_NORMAL_GRID = 8   # inside the training range
RC_EVENT_GRID = 13   # first sweep point past the SCDT 50% crossing
COLORS = ("#2166ac", "#e08214", "#1b9e77")


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def style() -> None:
    """SCDT paper figure style: serif text, inward ticks on all four sides, no grid."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.linewidth": 1.1,
        "axes.grid": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


PAPER_BLUE, PAPER_RED, PAPER_BLACK = "#2f5d8a", "#b5413c", "#1a1a1a"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=230, bbox_inches="tight")
    plt.close(fig)


def names(summary: dict) -> tuple[str, str]:
    return summary["statistics"]["leading"], summary["statistics"]["second"]


LABEL = {"mean": "mean", "sigma": r"$\sigma$", "q05": r"$q_{05}$", "abs_corr": "|corr|"}


def parameter_statistics(summary: dict) -> None:
    """Statistic vs hidden drift (paper panel-b style): one point per λ level = mean of 20 records."""
    lead, second = names(summary)
    cohort = rows(ROOT / "data/statistics_cohort.csv")
    lam = np.asarray([float(r["lam"]) for r in cohort])
    levels = np.unique(lam)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, name in zip(axes, (lead, second)):
        value = np.asarray([float(r[name]) for r in cohort])
        mean = np.asarray([value[lam == v].mean() for v in levels])
        se = np.asarray([value[lam == v].std(ddof=1) / np.sqrt(np.sum(lam == v)) for v in levels])
        slope, intercept = np.polyfit(levels, mean, 1)
        r2 = 1 - np.sum((mean - (slope * levels + intercept)) ** 2) / np.sum((mean - mean.mean()) ** 2)
        ax.plot(levels, slope * levels + intercept, color=PAPER_RED, ls=(0, (5, 2, 1, 2)), lw=1.6, zorder=1)
        ax.errorbar(levels, mean, yerr=se, fmt="o", ms=7, color=PAPER_BLUE, ecolor=PAPER_BLUE,
                    elinewidth=1, capsize=0, zorder=2)
        ax.set(xlabel=r"$\lambda$", ylabel=LABEL.get(name, name))
        ax.text(.04 if slope > 0 else .96, .92, fr"$R^2={r2:.3f}$", transform=ax.transAxes,
                ha="left" if slope > 0 else "right", va="top", fontsize=11)
    fig.tight_layout()
    save(fig, "01_parameter_statistics.png")


def survival_curve(summary: dict) -> None:
    """GT vs SCDT survival (paper panel-a style) on the same 36 coordinates, λ on the top axis."""
    lead = names(summary)[0]
    primary = rows(ROOT / "data/scdt_sweep_seed733900.csv")
    gt = rows(ROOT / "data/gt_same_grid.csv")
    x = np.asarray([float(r["lead_input"]) for r in primary])
    y = 100 * np.asarray([float(r["survival_rate"]) for r in primary])
    gt_x = np.asarray([float(r["lead_input"]) for r in gt])
    gt_y = 100 * np.asarray([float(r["survival_rate"]) for r in gt])
    assert np.allclose(gt_x, x), "GT must share the SCDT sweep grid"
    a, b, _ = summary["alignment_validation_only"]["lead_vs_lam"]
    scdt50 = summary["scdt_primary"]["lead_50"]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.plot(gt_x, gt_y, color=PAPER_BLACK, lw=1.6, marker="o", ms=5, label="GT")
    ax.plot(x, y, color=PAPER_BLUE, lw=1.6, marker="s", ms=5, label="SCDT")
    ax.axvline(scdt50, color=PAPER_RED, ls="--", lw=2, label="estimated critical point")
    ax.set(xlabel=LABEL.get(lead, lead), ylabel="survival rate (%)", ylim=(-3, 103))
    ax.invert_xaxis()   # the statistic decreases toward the transition; λ increases to the right
    top = ax.secondary_xaxis("top", functions=(lambda s: (s - b) / a, lambda l: a * l + b))
    top.set_xlabel(r"$\lambda$")
    top.tick_params(direction="in")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "03_survival_curve.png")


def reconstruct_cohorts(summary: dict, protocol: dict) -> dict[str, np.ndarray]:
    """Recreate exact locked RC sweep seeds and GT switch cohorts once."""
    if COHORT_CACHE.exists():
        with np.load(COHORT_CACHE) as archive:
            if "event_grid_index" in archive and int(archive["event_grid_index"][0]) == RC_EVENT_GRID:
                return {key: archive[key] for key in archive.files}

    args = protocol["arguments"]
    experiment = Experiment(Namespace(**{**args, "out_dir": ROOT}))
    experiment.prepare()
    print("Fitting locked 1500-node reservoir for trajectory figures...", flush=True)
    model = experiment.build(RCParams(**summary["locked_parameters"]))

    sweep = rows(ROOT / "data/scdt_sweep_seed733900.csv")
    rc_out = {}
    for name, grid_index in [("normal_rc", RC_NORMAL_GRID), ("event_rc", RC_EVENT_GRID)]:
        row = sweep[grid_index]
        lead, second = float(row["lead_input"]), float(row["second_input"])
        count = int(args["rc_ensemble"])
        warm, warm_static = experiment.warmups(lead, count)
        target = experiment.norm_stat(np.asarray([lead, second]))
        pred_n, _ = model.rollout(warm, warm_static,
                                  np.repeat(target[None], count, axis=0),
                                  200 * RC_HZ, int(args["seed"]) + 300000 + grid_index)
        prediction = experiment.denorm(pred_n)
        times, _ = first_departure(prediction)
        observed = int(np.sum(~np.isfinite(times)))
        expected = int(row["survivors"])
        if observed != expected:
            print(f"WARNING {name}: reconstructed survival {observed}/{count}; "
                  f"original sweep recorded {expected}/{count}", flush=True)
        rc_out[name] = prediction
        print(f"{name}: {observed}/{count} survive", flush=True)

    gt_rows = rows(ROOT / "data/gt_boundary.csv")
    seed = int(args["seed"]) + 500000
    net = BatchNetwork(int(args["gt_ensemble"]), seed)
    state = net.initial_state()
    state, _ = net.run(state, float(protocol["lam"]["gt_preparation"]), 600, record=False)
    gt_out = {}
    for name, grid_index in [("normal_gt", 0), ("event_gt", 12)]:
        lam = float(gt_rows[grid_index]["lam"])
        net.rng = np.random.default_rng(seed + 1000 + grid_index)
        _, prediction = net.run(state.copy(), lam, 200)
        times, _ = first_departure(prediction)
        assert int(np.sum(~np.isfinite(times))) == int(gt_rows[grid_index]["survivors"]), name
        gt_out[name] = prediction
        print(f"{name}: {gt_rows[grid_index]['survivors']}/{len(times)} survive", flush=True)
    result = {**rc_out, **gt_out, "event_grid_index": np.asarray([RC_EVENT_GRID])}
    np.savez_compressed(COHORT_CACHE, **result)
    return result


def trajectory_comparison(summary: dict, protocol: dict) -> dict:
    lead = names(summary)[0]
    sweep = rows(ROOT / "data/scdt_sweep_seed733900.csv")
    normal_value = float(sweep[RC_NORMAL_GRID]["lead_input"])
    event_value = float(sweep[RC_EVENT_GRID]["lead_input"])
    symbol = {"mean": r"\mathrm{mean}", "q05": r"q_{05}", "sigma": r"\sigma"}.get(lead, lead)
    cohorts = reconstruct_cohorts(summary, protocol)
    normal_gt, normal_rc = cohorts["normal_gt"], cohorts["normal_rc"]
    event_gt, event_rc = cohorts["event_gt"], cohorts["event_rc"]
    gt_normal_index = int(np.argmin(np.abs(np.std(normal_gt, axis=(0, 2)) -
                                       np.median(np.std(normal_gt, axis=(0, 2))))))
    rc_normal_index = int(np.argmin(np.abs(np.std(normal_rc, axis=(0, 2)) -
                                       np.median(np.std(normal_rc, axis=(0, 2))))))
    gt_onsets, gt_nodes = first_departure(event_gt)
    rc_onsets, rc_nodes = first_departure(event_rc)
    gt_departed = np.flatnonzero(np.isfinite(gt_onsets))
    rc_departed = np.flatnonzero(np.isfinite(rc_onsets))
    gt_event_index = int(gt_departed[np.argmin(np.abs(gt_onsets[gt_departed] -
                                                    np.median(gt_onsets[gt_departed])))] )
    rc_event_index = int(rc_departed[np.argmin(np.abs(rc_onsets[rc_departed] -
                                                    np.median(rc_onsets[rc_departed])))] )

    panels = [
        (normal_gt[:, gt_normal_index], "Normal · GT", r"$\lambda=-2.070$", None),
        (normal_rc[:, rc_normal_index], "Normal · SCDT", fr"${symbol}={normal_value:.4f}$", None),
        (event_gt[:, gt_event_index], "Departure · GT", r"$\lambda=-2.040$", gt_onsets[gt_event_index]),
        (event_rc[:, rc_event_index], "Departure · SCDT", fr"${symbol}={event_value:.4f}$", rc_onsets[rc_event_index]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4), sharex=True, sharey=True)
    for ax, (signal, title, condition, onset) in zip(axes.flat, panels):
        t = np.arange(len(signal)) / RC_HZ
        for node in range(3):
            ax.plot(t, signal[:, node], lw=.7, color=COLORS[node], label=f"node {node}")
        ax.axhline(EVENT_THRESHOLD, color="#b2182b", lw=.8, ls="--")
        if onset is not None:
            ax.axvline(onset, color="#b2182b", lw=.9, ls=":")
        ax.set(title=f"{title}   {condition}", xlim=(0, 200), ylim=(-6.5, 3.2))
    for ax in axes[:, 0]:
        ax.set_ylabel(r"node proxy $x_2-x_1$")
    for ax in axes[1]:
        ax.set_xlabel("time after parameter/condition switch (s)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8, loc="lower left")
    fig.suptitle("Three-node GT and autonomous SCDT trajectories", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, "04_normal_departure_trajectories_2x2.png")
    return {
        "normal_gt_lam": -2.07,
        "normal_scdt_lead": normal_value,
        "departure_gt_lam": -2.04,
        "departure_scdt_lead": event_value,
        "normal_gt_member": gt_normal_index,
        "normal_scdt_member": rc_normal_index,
        "departure_gt_member": gt_event_index,
        "departure_scdt_member": rc_event_index,
        "departure_gt_onset_s": float(gt_onsets[gt_event_index]),
        "departure_gt_first_node": int(gt_nodes[gt_event_index]),
        "departure_scdt_onset_s": float(rc_onsets[rc_event_index]),
        "departure_scdt_first_node": int(rc_nodes[rc_event_index]),
        "departure_scdt_departed_count": len(rc_departed),
        "departure_scdt_original_sweep_survivors": int(rows(ROOT / "data/scdt_sweep_seed733900.csv")[RC_EVENT_GRID]["survivors"]),
        "normal_scdt_reconstructed_survivors": int(np.sum(~np.isfinite(first_departure(normal_rc)[0]))),
        "normal_scdt_original_sweep_survivors": int(sweep[RC_NORMAL_GRID]["survivors"]),
    }


def bifurcation_scan() -> None:
    """Node-0-only finite-time minima over the broad x0,0 scan (not an asymptotic bifurcation proof)."""
    with np.load(BIFURCATION_CACHE) as archive:
        x = archive["broad_x0_0"]
        minimum = archive["broad_minimum"]
    values = np.unique(x)
    median = np.asarray([np.median(minimum[x == value]) for value in values])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.scatter(x, minimum, s=8, color=COLORS[0], alpha=.25, linewidths=0,
               label="node 0 minimum (16 noisy replicates)")
    ax.plot(values, median, color=COLORS[0], lw=1.6, label="median")
    ax.axhline(EVENT_THRESHOLD, color="#b2182b", ls="--", lw=1.1,
               label="departure threshold −0.5")
    ax.set(xlabel=r"$x_{0,0}$  (nodes 1, 2 fixed at $-2.12$, $-2.17$)",
           ylabel=r"node 0 min $(x_2-x_1)$", xlim=(values[0], values[-1]))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "05_gt_bifurcation_scan.png")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--bifurcation-only", action="store_true",
                        help="Rebuild only figure 05 from saved node-0 scan data")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    style()
    if args.bifurcation_only:
        bifurcation_scan()
        return
    summary = json.loads((ROOT / "results/summary.json").read_text())
    protocol = json.loads((ROOT / "results/protocol.json").read_text())
    parameter_statistics(summary)
    survival_curve(summary)
    selection = trajectory_comparison(summary, protocol)
    # figure 05 (bifurcation) removed from the presentation set on 2026-09-29; use --bifurcation-only to rebuild it
    (ROOT / "results/final_figure_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(json.dumps({"figures": sorted(p.name for p in OUT.glob("*.png")),
                      "trajectory_selection": selection}, indent=2))


if __name__ == "__main__":
    main()
