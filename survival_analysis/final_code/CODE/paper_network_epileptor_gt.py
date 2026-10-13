#!/usr/bin/env python3
"""Paper-style eight-node Network Epileptor ground truth.

The local dynamics and slow permittivity coupling follow Eq. (4) of
Proix et al., Brain 2017. The paper used patient-specific diffusion-MRI
connectomes. This controlled experiment instead uses an explicitly labeled
eight-node symmetric structural proxy so that the dynamical assumptions can
be inspected before introducing patient data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.signal import butter, sosfiltfilt

from paper_epileptor_core import (
    PaperEpileptorConfig as PaperNetworkConfig,
    X0_CRITICAL,
    detect_sustained_onset,
    simulate_fixed_x0,
    source_proxy,
)

NODE_COLORS = plt.get_cmap("tab10").colors


@dataclass(frozen=True)
class PaperNetworkTrajectory:
    time_s: np.ndarray
    states: np.ndarray
    proxy: np.ndarray
    filtered_proxy: np.ndarray
    x0: np.ndarray
    connectivity: np.ndarray
    onset_idx: np.ndarray


def build_paper_x0_profile(n_nodes: int = 8) -> tuple[np.ndarray, list[str]]:
    """Return fixed EZ/PZ/HZ excitability values in the Proix-2017 convention."""
    if n_nodes != 8:
        raise ValueError("The current paper-style synthetic network is defined for 8 nodes.")
    # Node 0 is autonomously epileptogenic. Nodes 1-3 remain just below the
    # critical value and can be recruited. Nodes 4-7 are stable healthy nodes.
    x0 = np.array([-1.98, -2.12, -2.16, -2.20, -2.50, -2.50, -2.50, -2.50])
    classes = ["EZ", "PZ", "PZ", "PZ", "HZ", "HZ", "HZ", "HZ"]
    return x0, classes


def build_synthetic_structural_proxy(n_nodes: int = 8) -> np.ndarray:
    """Return a symmetric weighted proxy for a structural connectome.

    Proix et al. used patient-specific tractography. These weights are not
    anatomical data; they only create a transparent delayed recruitment path.
    """
    if n_nodes != 8:
        raise ValueError("The current structural proxy is defined for 8 nodes.")
    weights = np.zeros((n_nodes, n_nodes), dtype=float)
    for node, weight in enumerate([1.00, 0.85, 0.70, 0.55, 0.40, 0.30, 0.22]):
        weights[node, node + 1] = weight
        weights[node + 1, node] = weight
    for source, target, weight in [
        (0, 3, 0.28),
        (1, 4, 0.22),
        (2, 5, 0.18),
        (3, 6, 0.14),
        (4, 7, 0.10),
    ]:
        weights[source, target] = weight
        weights[target, source] = weight
    return weights


def detect_recruitment(
    time_s: np.ndarray,
    proxy: np.ndarray,
    threshold: float = -2.0,
) -> np.ndarray:
    """Operational marker: first sustained oscillatory event below threshold."""
    onset = np.full(proxy.shape[1], np.nan, dtype=float)
    for node in range(proxy.shape[1]):
        onset_s = detect_sustained_onset(
            time_s,
            proxy[:, node],
            threshold=threshold,
        )
        if onset_s is not None:
            onset[node] = float(np.searchsorted(time_s, onset_s))
    return onset


def bandpass_proxy(proxy: np.ndarray, sampling_hz: int) -> np.ndarray:
    """Apply the fifth-order 0.16-97 Hz filter reported by Proix et al."""
    sos = butter(
        5,
        [0.16, 97.0],
        btype="bandpass",
        fs=float(sampling_hz),
        output="sos",
    )
    return sosfiltfilt(sos, proxy, axis=0)


def simulate_paper_network(
    config: PaperNetworkConfig,
) -> PaperNetworkTrajectory:
    """Simulate fixed-x0 Network Epileptor dynamics with slow coupling only."""
    x0, _ = build_paper_x0_profile(config.n_nodes)
    connectivity = build_synthetic_structural_proxy(config.n_nodes)
    time_s, states = simulate_fixed_x0(config, x0, connectivity)
    proxy = source_proxy(states)
    filtered_proxy = bandpass_proxy(proxy, config.sampling_hz)
    onset_idx = detect_recruitment(time_s, proxy)
    return PaperNetworkTrajectory(
        time_s=time_s,
        states=states,
        proxy=proxy,
        filtered_proxy=filtered_proxy,
        x0=x0,
        connectivity=connectivity,
        onset_idx=onset_idx,
    )


def onset_time(trajectory: PaperNetworkTrajectory, node: int) -> float | None:
    value = trajectory.onset_idx[node]
    return None if not np.isfinite(value) else float(trajectory.time_s[int(value)])


def add_onset_lines(ax: plt.Axes, trajectory: PaperNetworkTrajectory) -> None:
    for node, index in enumerate(trajectory.onset_idx):
        if np.isfinite(index):
            ax.axvline(
                trajectory.time_s[int(index)],
                color=NODE_COLORS[node],
                lw=1.1,
                ls="--",
                alpha=0.90,
            )


def plot_overview(
    trajectory: PaperNetworkTrajectory,
    config: PaperNetworkConfig,
) -> plt.Figure:
    _, classes = build_paper_x0_profile(config.n_nodes)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))

    for node in range(config.n_nodes):
        axes[0, 0].plot(
            trajectory.time_s,
            trajectory.proxy[:, node],
            color=NODE_COLORS[node],
            lw=0.65,
            alpha=0.80,
            label=f"node {node} ({classes[node]})",
        )
    add_onset_lines(axes[0, 0], trajectory)
    axes[0, 0].axhline(-2.0, color="#d62728", lw=1.0, ls=":")
    axes[0, 0].set_title("(a) Source proxy and recruitment")
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].set_ylabel(r"$x_2-x_1$ (a.u.)")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="upper right")
    axes[0, 0].grid(alpha=0.20)

    image = axes[0, 1].imshow(
        trajectory.proxy.T,
        origin="lower",
        aspect="auto",
        extent=[
            trajectory.time_s[0],
            trajectory.time_s[-1],
            -0.5,
            config.n_nodes - 0.5,
        ],
        cmap="RdBu_r",
        vmin=-3.0,
        vmax=3.0,
    )
    for node, index in enumerate(trajectory.onset_idx):
        if np.isfinite(index):
            axes[0, 1].scatter(
                trajectory.time_s[int(index)],
                node,
                s=28,
                marker="D",
                facecolor="black",
                edgecolor="white",
                linewidth=0.5,
            )
    axes[0, 1].set_title("(b) Recruitment heatmap")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("node")
    axes[0, 1].set_yticks(np.arange(config.n_nodes))
    fig.colorbar(image, ax=axes[0, 1], shrink=0.84, label="source proxy")

    nodes = np.arange(config.n_nodes)
    class_colors = {
        "EZ": "#c0392b",
        "PZ": "#d9911b",
        "HZ": "#3977a8",
    }
    axes[1, 0].bar(
        nodes,
        trajectory.x0,
        color=[class_colors[value] for value in classes],
    )
    axes[1, 0].axhline(
        X0_CRITICAL,
        color="#202020",
        lw=1.5,
        ls="--",
        label=r"$x_0^C=-2.1$",
    )
    axes[1, 0].set_ylim(-2.62, -1.82)
    axes[1, 0].set_title("(c) Fixed regional excitability")
    axes[1, 0].set_xlabel("node")
    axes[1, 0].set_ylabel(r"$x_{0,i}$")
    axes[1, 0].set_xticks(nodes)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.20)

    matrix = axes[1, 1].imshow(
        trajectory.connectivity,
        cmap="Blues",
        vmin=0.0,
        vmax=np.max(trajectory.connectivity),
    )
    for row in range(config.n_nodes):
        for column in range(config.n_nodes):
            axes[1, 1].text(
                column,
                row,
                f"{trajectory.connectivity[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "white"
                    if trajectory.connectivity[row, column] > 0.52
                    else "black"
                ),
            )
    axes[1, 1].set_title("(d) Synthetic structural proxy W")
    axes[1, 1].set_xlabel("source node")
    axes[1, 1].set_ylabel("target node")
    axes[1, 1].set_xticks(nodes)
    axes[1, 1].set_yticks(nodes)
    fig.colorbar(matrix, ax=axes[1, 1], shrink=0.84, label="connection weight")

    fig.suptitle(
        "Paper-style Network Epileptor GT: fixed x0, slow coupling only",
        fontsize=15,
    )
    fig.tight_layout()
    return fig


def plot_node_traces(
    trajectory: PaperNetworkTrajectory,
    config: PaperNetworkConfig,
) -> plt.Figure:
    _, classes = build_paper_x0_profile(config.n_nodes)
    fig, axes = plt.subplots(config.n_nodes, 1, figsize=(13.2, 10.0), sharex=True)
    for node, ax in enumerate(axes):
        ax.plot(
            trajectory.time_s,
            trajectory.filtered_proxy[:, node],
            color=NODE_COLORS[node],
            lw=0.75,
        )
        onset = onset_time(trajectory, node)
        if onset is not None:
            ax.axvline(onset, color=NODE_COLORS[node], lw=1.2, ls="--")
        ax.set_ylabel(f"{node} {classes[node]}")
        ax.grid(alpha=0.18)
    axes[0].set_title(
        "Filtered source proxies; dashed line = sustained recruitment onset"
    )
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def write_outputs(
    trajectory: PaperNetworkTrajectory,
    config: PaperNetworkConfig,
    out_dir: Path,
) -> None:
    figure_dir = out_dir / "figures" / "GT" / "paper_network"
    data_dir = out_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    overview = plot_overview(trajectory, config)
    traces = plot_node_traces(trajectory, config)
    overview_path = figure_dir / "paper_network_gt_overview.png"
    traces_path = figure_dir / "paper_network_gt_node_traces.png"
    pdf_path = figure_dir / "paper_network_gt.pdf"
    overview.savefig(overview_path, dpi=230, bbox_inches="tight")
    traces.savefig(traces_path, dpi=230, bbox_inches="tight")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(overview, bbox_inches="tight")
        pdf.savefig(traces, bbox_inches="tight")
    plt.close(overview)
    plt.close(traces)

    csv_path = data_dir / "paper_network_gt.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["time_s"]
        header += [f"node{node}_proxy" for node in range(config.n_nodes)]
        header += [
            f"node{node}_filtered_proxy" for node in range(config.n_nodes)
        ]
        writer.writerow(header)
        for index in range(len(trajectory.time_s)):
            writer.writerow(
                [
                    trajectory.time_s[index],
                    *trajectory.proxy[index].tolist(),
                    *trajectory.filtered_proxy[index].tolist(),
                ]
            )

    summary_path = data_dir / "paper_network_gt_summary.csv"
    _, classes = build_paper_x0_profile(config.n_nodes)
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", "class", "x0", "recruitment_time_s"])
        for node in range(config.n_nodes):
            writer.writerow(
                [
                    node,
                    classes[node],
                    trajectory.x0[node],
                    onset_time(trajectory, node),
                ]
            )

    np.savez_compressed(
        data_dir / "paper_network_gt.npz",
        time_s=trajectory.time_s,
        states=trajectory.states,
        proxy=trajectory.proxy,
        filtered_proxy=trajectory.filtered_proxy,
        x0=trajectory.x0,
        connectivity=trajectory.connectivity,
        onset_idx=trajectory.onset_idx,
    )
    print(f"Saved figures to {figure_dir}")
    print(f"Saved data to {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--global-coupling", type=float, default=1.0)
    parser.add_argument("--noise-variance", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()

    config = PaperNetworkConfig(
        duration_s=args.duration_s,
        global_coupling=args.global_coupling,
        noise_variance=args.noise_variance,
        seed=args.seed,
    )
    trajectory = simulate_paper_network(config)
    write_outputs(trajectory, config, args.out_dir)
    print(
        "Recruitment times (s):",
        [
            None
            if not np.isfinite(index)
            else round(float(trajectory.time_s[int(index)]), 3)
            for index in trajectory.onset_idx
        ],
    )


if __name__ == "__main__":
    main()
