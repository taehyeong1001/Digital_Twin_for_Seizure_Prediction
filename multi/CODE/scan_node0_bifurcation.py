#!/usr/bin/env python3
"""Finite-time stochastic scan with only node 0's x0 varying.

This is an extrema/escape diagram, not a continuation calculation or a
mathematical proof of a bifurcation. Nodes 1 and 2 stay at the normal
reference coordinates used by the three-node protocol (lambda = -2.07).
"""

from __future__ import annotations

import csv
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from network_gt import (
    EVENT_THRESHOLD,
    N_NODES,
    RAW_HZ,
    STRIDE,
    BatchNetwork,
)
from paper_epileptor_core import source_proxy


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIXED_X0 = (-2.12, -2.17)
REPLICATES = 16
PREP_S = 600
RECORD_S = 200
SEED = 20260926
BROAD_GRID = np.round(np.arange(-2.30, -1.92 + 0.00001, 0.01), 6)
FINE_STEP = 0.0005
FINE_PAD = 0.010


def scan(grid: np.ndarray, seed: int, name: str, preparation_s: int,
         recording_s: int) -> dict[str, np.ndarray]:
    """Prepare at each fixed coordinate, then stream node-0 extrema."""
    started = time.monotonic()
    n = len(grid) * REPLICATES
    net = BatchNetwork(n, seed=seed)
    state = net.initial_state()
    x0_0 = np.repeat(grid, REPLICATES)
    x0 = np.column_stack((x0_0, np.full(n, FIXED_X0[0]),
                          np.full(n, FIXED_X0[1])))
    for _ in range(preparation_s * RAW_HZ):
        state = net.step(state, x0)
    print(f"{name}: preparation done in {time.monotonic()-started:.1f}s", flush=True)

    minimum = np.full(n, np.inf)
    maximum = np.full(n, -np.inf)
    departed = np.zeros(n, dtype=bool)
    for step in range(recording_s * RAW_HZ):
        state = net.step(state, x0)
        if (step + 1) % STRIDE:
            continue
        node0 = source_proxy(state)[:, 0]
        minimum = np.minimum(minimum, node0)
        maximum = np.maximum(maximum, node0)
        if (step + 1) / RAW_HZ > 5.0:
            departed |= node0 <= EVENT_THRESHOLD
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise FloatingPointError(f"non-finite {name} scan")
    print(f"{name}: {len(grid)} points x {REPLICATES} copies, "
          f"total {time.monotonic()-started:.1f}s", flush=True)
    return {
        "x0_0": x0_0,
        "x0_1": x0[:, 1],
        "x0_2": x0[:, 2],
        "replicate": np.tile(np.arange(REPLICATES), len(grid)),
        "minimum": minimum,
        "maximum": maximum,
        "departed": departed,
    }


def fine_grid_from_broad(broad: dict[str, np.ndarray]) -> np.ndarray:
    values = np.unique(broad["x0_0"])
    rate = np.asarray([np.mean(broad["departed"][broad["x0_0"] == x])
                       for x in values])
    changed = np.flatnonzero(np.diff(rate) != 0)
    if len(changed) == 0:
        raise RuntimeError("Broad scan found no change in node-0 escape rate; "
                           "inspect the broad results before choosing a fine grid")
    low = values[changed[0]] - FINE_PAD
    high = values[changed[-1] + 1] + FINE_PAD
    return np.round(np.arange(low, high + FINE_STEP / 2, FINE_STEP), 6)


def write_csv(path: Path, scans: dict[str, dict[str, np.ndarray]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("grid", "x0_0", "x0_1_fixed", "x0_2_fixed", "replicate",
                         "node0_proxy_min", "node0_proxy_max", "node0_departed"))
        for name, scan_data in scans.items():
            for i in range(len(scan_data["x0_0"])):
                writer.writerow((name, *(scan_data[key][i] for key in (
                    "x0_0", "x0_1", "x0_2", "replicate", "minimum", "maximum", "departed"))))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-s", type=int, default=PREP_S)
    parser.add_argument("--recording-s", type=int, default=RECORD_S)
    parser.add_argument("--fine-window", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        help="Use this fine-grid interval instead of selecting it from the broad scan")
    parser.add_argument("--output-stem", default="node0_bifurcation_scan")
    args = parser.parse_args()
    if args.preparation_s <= 0 or args.recording_s <= 0:
        parser.error("preparation and recording durations must be positive")
    if not args.output_stem.startswith("node0_bifurcation_scan"):
        parser.error("output stem must start with node0_bifurcation_scan")
    DATA.mkdir(exist_ok=True)
    broad = scan(BROAD_GRID, SEED, "broad", args.preparation_s, args.recording_s)
    # Save the broad result even if a finer transition interval cannot be selected.
    np.savez_compressed(DATA / f"{args.output_stem}_broad.npz", **broad)
    if args.fine_window is None:
        fine_grid = fine_grid_from_broad(broad)
    else:
        low, high = args.fine_window
        if low >= high:
            parser.error("fine-window LOW must be smaller than HIGH")
        fine_grid = np.round(np.arange(low, high + FINE_STEP / 2, FINE_STEP), 6)
    print(f"fine grid: [{fine_grid[0]:.4f}, {fine_grid[-1]:.4f}], "
          f"step {FINE_STEP:.4f}, {len(fine_grid)} points", flush=True)
    fine = scan(fine_grid, SEED + 1, "fine", args.preparation_s, args.recording_s)
    np.savez_compressed(DATA / f"{args.output_stem}.npz",
                        **{f"{name}_{key}": value
                           for name, part in (("broad", broad), ("fine", fine))
                           for key, value in part.items()},
                        preparation_s=args.preparation_s, recording_s=args.recording_s,
                        repeats=REPLICATES, noise_variance=net_noise_variance())
    write_csv(DATA / f"{args.output_stem}.csv", {"broad": broad, "fine": fine})
    for name, part in (("broad", broad), ("fine", fine)):
        grid = np.unique(part["x0_0"])
        rates = np.asarray([np.mean(part["departed"][part["x0_0"] == x]) for x in grid])
        first = np.flatnonzero(rates > 0)
        complete = np.flatnonzero(rates == 1)
        print(f"{name}: first departure at {grid[first[0]] if len(first) else 'none'}; "
              f"first {REPLICATES}/{REPLICATES} departure at "
              f"{grid[complete[0]] if len(complete) else 'none'}",
              flush=True)


def net_noise_variance() -> float:
    return BatchNetwork(1, SEED).cfg.noise_variance


if __name__ == "__main__":
    main()
