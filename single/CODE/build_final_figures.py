#!/usr/bin/env python3
"""Build the five curated single-node SCDT figures from audited results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_single_node_scrc import RCParams
from strict_single_node_scdt import (
    EVENT_BLANK_S,
    EVENT_THRESHOLD,
    build_rc,
    event_mask,
    fit_morphology_reference,
    simulate_switch_batch,
)
from waveform_guard import fit_median_reference, high_median_event_mask


ROOT = Path(__file__).resolve().parents[1]
AICP = ROOT.parents[1]
OUT = ROOT / "figure"


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


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUT / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parameter_statistics(protocol: dict) -> None:
    """Statistic vs hidden x0 (paper panel-b style): one point per x0 level = mean of 20 records."""
    cohort = rows(ROOT / "data/statistics_cohort.csv")
    x0 = np.asarray([float(r["x0"]) for r in cohort])
    levels = np.unique(x0)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, key, label in [(axes[0], "sigma", r"std ($\sigma$)"), (axes[1], "permutation_entropy", "permutation entropy")]:
        value = np.asarray([float(r[key]) for r in cohort])
        mean = np.asarray([value[x0 == v].mean() for v in levels])
        se = np.asarray([value[x0 == v].std(ddof=1) / np.sqrt(np.sum(x0 == v)) for v in levels])
        slope, intercept = np.polyfit(levels, mean, 1)
        r2 = 1 - np.sum((mean - (slope * levels + intercept)) ** 2) / np.sum((mean - mean.mean()) ** 2)
        ax.plot(levels, slope * levels + intercept, color=PAPER_RED, ls=(0, (5, 2, 1, 2)), lw=1.6, zorder=1)
        ax.errorbar(levels, mean, yerr=se, fmt="o", ms=7, color=PAPER_BLUE, ecolor=PAPER_BLUE,
                    elinewidth=1, capsize=0, zorder=2)
        ax.set(xlabel=r"$x_0$", ylabel=label)
        ax.text(.04 if slope > 0 else .96, .92, fr"$R^2={r2:.3f}$", transform=ax.transAxes,
                ha="left" if slope > 0 else "right", va="top", fontsize=11)
    fig.tight_layout()
    save(fig, "01_parameter_statistics.png")


def crossing50(x: np.ndarray, s: np.ndarray) -> float:
    k = int(np.flatnonzero(s <= .5)[0])
    return float(x[k - 1] + (.5 - s[k - 1]) * (x[k] - x[k - 1]) / (s[k] - s[k - 1]))


def survival_curve(protocol: dict) -> None:
    """GT vs SCDT survival (paper panel-a style), x0 on the top axis."""
    rc = rows(ROOT / "data/scdt_rate_sweep_median_guard.csv")
    gt = rows(ROOT / "data/gt_same_grid.csv")  # GT on the same 35 σ coordinates as the SCDT sweep
    rc_x = np.asarray([float(row["sigma_input"]) for row in rc])
    rc_s = np.asarray([float(row["survival_rate"]) for row in rc])
    gt_x = np.asarray([float(row["sigma_input"]) for row in gt])
    gt_s = np.asarray([float(row["survival_rate"]) for row in gt])
    order = np.argsort(gt_x)
    mapping = protocol["x0_sigma_alignment_validation_only"]
    a, b = mapping["slope"], mapping["intercept"]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.plot(gt_x[order], 100 * gt_s[order], color=PAPER_BLACK, lw=1.6, marker="o", ms=5, label="GT")
    ax.plot(rc_x, 100 * rc_s, color=PAPER_BLUE, lw=1.6, marker="s", ms=5, label="SCDT")
    ax.axvline(crossing50(rc_x, rc_s), color=PAPER_RED, ls="--", lw=2, label="estimated critical point")
    ax.set(xlabel=r"std ($\sigma$)", ylabel="survival rate (%)", ylim=(-3, 103))
    top = ax.secondary_xaxis("top", functions=(lambda s: (s - b) / a, lambda x: a * x + b))
    top.set_xlabel(r"$x_0$")
    top.tick_params(direction="in")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "03_survival_curve.png")


def rc_prediction(protocol: dict, grid_index: int) -> tuple[np.ndarray, np.ndarray]:
    args = protocol["arguments"]
    with np.load(ROOT / "data/training_signals.npz") as archive:
        training = [signal for signal in archive["signals"]]
        stats = archive["statistics"]
        sample_hz = float(archive["sample_hz"])
    all_values = np.concatenate(training)
    center, scale = float(np.mean(all_values)), float(np.std(all_values))
    static_center = np.mean(stats, axis=0)
    static_scale = np.std(stats, axis=0)
    normalized = [(signal - center) / scale for signal in training]
    normalized_stats = (stats - static_center) / static_scale
    model = build_rc(RCParams(**protocol["best_rc_parameters"]),
                     normalized, normalized_stats, int(args["fit_warmup"]))
    sweep = rows(ROOT / "data/scdt_rate_sweep.csv")
    row = sweep[grid_index]
    raw = np.asarray([float(row["sigma_input"]),
                      float(row["permutation_entropy_input"])])
    target = (raw - static_center) / static_scale
    count = int(args["rc_ensemble"])
    indices = np.arange(count) % len(normalized)
    warmup = np.column_stack([
        normalized[index][-int(args["warmup_samples"]):] for index in indices
    ])
    pred_norm = model.rollout_switch_batch(
        warmup, normalized_stats[indices],
        np.repeat(target[None, :], count, axis=0),
        int(round(float(args["rollout_s"]) * sample_hz)),
        int(args["seed"]) + 300000 + grid_index,
    )
    prediction = pred_norm * scale + center
    morphology = fit_morphology_reference(training, sample_hz)
    median = fit_median_reference(training, sample_hz)
    events = event_mask(prediction, sample_hz, morphology) | high_median_event_mask(prediction, median)
    expected = rows(ROOT / "data/scdt_rate_sweep_median_guard.csv")[grid_index]
    assert int(np.sum(~events)) == int(expected["survivors"])
    return prediction, events


def representative_normal(signals: np.ndarray, events: np.ndarray) -> int:
    survivors = np.flatnonzero(~events)
    signal_std = np.std(signals[:, survivors], axis=0)
    return int(survivors[np.argmin(np.abs(signal_std - np.median(signal_std)))])


def representative_departure(signals: np.ndarray, sample_hz: float) -> tuple[int, float]:
    start = int(round(EVENT_BLANK_S * sample_hz))
    hit = signals[start:] <= EVENT_THRESHOLD
    columns = np.flatnonzero(np.any(hit, axis=0))
    onset = (start + np.argmax(hit[:, columns], axis=0)) / sample_hz
    selected = int(np.argmin(np.abs(onset - np.median(onset))))
    return int(columns[selected]), float(onset[selected])


def delayed_departure_example(signals: np.ndarray, sample_hz: float) -> tuple[int, float, int]:
    """Select a late collapse with visible pre-collapse peaks; report its frequency."""
    from scipy.signal import find_peaks

    candidates = []
    for member in range(signals.shape[1]):
        crossing = np.flatnonzero(signals[:, member] <= EVENT_THRESHOLD)
        if crossing.size == 0:
            continue
        onset = float(crossing[0] / sample_hz)
        if onset > 10:
            candidates.append((member, onset))
    eligible = [(member, onset) for member, onset in candidates if 25 <= onset <= 110]
    if not eligible:
        raise RuntimeError("No delayed departure with enough pre- and post-event time")
    scored = []
    for member, onset in eligible:
        pre = signals[:int(round(onset * sample_hz)), member]
        peak_count = len(find_peaks(pre, height=1.2, prominence=.6)[0])
        scored.append((peak_count, member, onset))
    _, member, onset = max(scored)
    return int(member), float(onset), len(candidates)


def trajectory_comparison(protocol: dict) -> dict:
    args = protocol["arguments"]
    sample_hz = 256.0 / int(args["stride"])
    normal_rc, normal_events = rc_prediction(protocol, 7)
    event_rc, event_events = rc_prediction(protocol, 17)
    normal_gt = simulate_switch_batch(
        -2.22, 30, -2.18, float(args["warmup_s"]), float(args["rollout_s"]),
        int(args["seed"]) + 700000 + 8 * 1000,
    )[1][::int(args["stride"])]
    event_gt = simulate_switch_batch(
        -2.04, 30, -2.18, float(args["warmup_s"]), float(args["rollout_s"]),
        int(args["seed"]) + 500000 + 40 * 1000,
    )[1][::int(args["stride"])]
    with np.load(ROOT / "data/training_signals.npz") as archive:
        training = [signal for signal in archive["signals"]]
    morphology = fit_morphology_reference(training, sample_hz)
    median = fit_median_reference(training, sample_hz)
    normal_gt_events = event_mask(normal_gt, sample_hz, morphology) | high_median_event_mask(normal_gt, median)
    event_gt_events = event_mask(event_gt, sample_hz, morphology) | high_median_event_mask(event_gt, median)
    assert np.sum(~normal_gt_events) == 30
    assert np.sum(~event_gt_events) == 0
    normal_gt_index = representative_normal(normal_gt, normal_gt_events)
    normal_rc_index = representative_normal(normal_rc, normal_events)
    event_gt_index, event_gt_onset = representative_departure(event_gt, sample_hz)
    event_rc_index, event_rc_onset, delayed_rc_count = delayed_departure_example(event_rc, sample_hz)

    time = np.arange(normal_gt.shape[0]) / sample_hz
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.2), sharex=True)
    panels = [
        (axes[0, 0], normal_gt[:, normal_gt_index], "GT", "Normal", r"$x_0=-2.22$", None),
        (axes[0, 1], normal_rc[:, normal_rc_index], "SCDT", "Normal", r"$\sigma=0.32786$", None),
        (axes[1, 0], event_gt[:, event_gt_index], "GT", "Departure", r"$x_0=-2.04$", event_gt_onset),
        (axes[1, 1], event_rc[:, event_rc_index], "SCDT", "Departure", r"$\sigma=0.38612$", event_rc_onset),
    ]
    for axis, signal, model_name, regime, condition, onset in panels:
        axis.plot(time, signal, lw=.8, color="#202020" if model_name == "GT" else "#2166ac")
        axis.axhline(EVENT_THRESHOLD, color="#b2182b", ls="--", lw=.9)
        if onset is not None:
            axis.axvline(onset, color="#b2182b", ls=":", lw=.9)
        axis.set_title(f"{regime} · {model_name}   {condition}", loc="left", fontsize=11)
        axis.set_xlim(0, 140)
    for axis in axes[0]:
        axis.set_ylim(-.9, 3.1)
    for axis in axes[1]:
        axis.set_ylim(-6.6, 3.1)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"proxy $x_2-x_1$")
    for axis in axes[1]:
        axis.set_xlabel("time after parameter/condition switch (s)")
    fig.suptitle("GT and autonomous SCDT trajectories", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, "04_normal_departure_trajectories_2x2.png")
    return {
        "normal_gt_x0": -2.22, "normal_scdt_sigma": 0.32785964118919847,
        "departure_gt_x0": -2.04, "departure_scdt_sigma": 0.3861177009457887,
        "normal_gt_member": normal_gt_index, "normal_scdt_member": normal_rc_index,
        "departure_gt_member": event_gt_index, "departure_scdt_member": event_rc_index,
        "departure_gt_first_trough_s": event_gt_onset,
        "departure_scdt_first_trough_s": event_rc_onset,
        "departure_scdt_delayed_over_10s_count": delayed_rc_count,
        "departure_scdt_selection": "Among first-trough crossings at 25–110 s, select the trace with most peaks above 1.2 and prominence 0.6 before crossing.",
    }


def bifurcation() -> None:
    """Broad-range, minimum-only deterministic long-time bifurcation diagram."""
    source = AICP / "MultiNode_Epileptor/experiments/single_node_longtime_bifurcation"
    with np.load(source / "longtime_bifurcation.npz") as archive:
        x0 = archive["x0"]
        minimum = archive["minimum"]
    with (source / "summary.json").open() as handle:
        summary = json.load(handle)
    critical = summary["empirical_first_x0_with_longtime_min_below_minus_0p5"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(x0, minimum, color="#2166ac", lw=1.6, label=r"minimum of $x_2-x_1$")
    ax.axvline(critical, color="#1a9850", ls=":", lw=1.3,
               label=rf"critical grid point $x_0={critical:.4f}$")
    ax.axhline(EVENT_THRESHOLD, color="#b2182b", ls="--", lw=1.1,
               label="departure threshold −0.5")
    ax.set(xlabel=r"$x_0$", ylabel=r"min $(x_2-x_1)$", xlim=(x0.min(), x0.max()))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "05_bifurcation.png")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bifurcation-only", action="store_true",
                        help="Rebuild only figure 05 from the saved long-time bifurcation data")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    if args.bifurcation_only:
        bifurcation()
        return
    with (ROOT / "results/protocol.json").open() as handle:
        protocol = json.load(handle)
    parameter_statistics(protocol)
    survival_curve(protocol)
    selection = trajectory_comparison(protocol)
    # figure 05 (bifurcation) removed from the presentation set on 2026-09-29; use --bifurcation-only to rebuild it
    with (ROOT / "results/final_figure_selection.json").open("w") as handle:
        json.dump(selection, handle, indent=2)
    print(json.dumps({"figures": [path.name for path in sorted(OUT.glob("*.png"))],
                      "trajectory_selection": selection}, indent=2))


if __name__ == "__main__":
    main()
