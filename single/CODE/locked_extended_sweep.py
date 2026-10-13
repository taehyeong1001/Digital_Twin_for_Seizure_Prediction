#!/usr/bin/env python3
"""Extend the statistical sweep of the already selected strict SCDT model.

This script never performs model selection.  It reconstructs the locked model
from saved training records and parameters, extends only the supplied statistic
range, and records whether a 50% departure transition exists.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_single_node_scrc import RCParams
from strict_single_node_scdt import build_rc, crossing, fit_morphology_reference, sweep_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(".."))
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--points", type=int, default=27)
    parser.add_argument("--ensemble", type=int, default=30)
    args = parser.parse_args()
    root = args.out_dir.resolve()

    with (root / "results" / "protocol.json").open() as handle:
        protocol = json.load(handle)
    saved = np.load(root / "data" / "training_signals.npz")
    signals = np.asarray(saved["signals"], dtype=float)
    statics = np.asarray(saved["statistics"], dtype=float)
    sample_hz = float(saved["sample_hz"])
    dynamic_center = float(np.mean(signals))
    dynamic_scale = float(np.std(signals))
    static_center = np.mean(statics, axis=0)
    static_scale = np.std(statics, axis=0)
    static_scale = np.where(static_scale > 1e-9, static_scale, 1.0)
    sequences_norm = [(signal - dynamic_center) / dynamic_scale for signal in signals]
    statics_norm = (statics - static_center) / static_scale
    morphology_reference = fit_morphology_reference([signal for signal in signals], sample_hz)
    params = RCParams(**protocol["best_rc_parameters"])
    fit_warmup = int(protocol["arguments"]["fit_warmup"])
    rc = build_rc(params, sequences_norm, statics_norm, fit_warmup)

    original = pd.read_csv(root / "data" / "scdt_rate_sweep.csv")
    start = float(original["sigma_input"].max()) + 0.01
    sigma = np.linspace(start, args.sigma_max, args.points)
    path = protocol["statistical_path"]
    warmup_samples = int(protocol["arguments"]["warmup_samples"])
    rollout_samples = int(round(float(protocol["arguments"]["rollout_s"]) * sample_hz))
    rows, _ = sweep_rate(
        rc,
        sigma,
        float(path["slope"]),
        float(path["intercept"]),
        sequences_norm,
        statics_norm,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        sample_hz,
        warmup_samples,
        rollout_samples,
        args.ensemble,
        20260924 + 700000,
        morphology_reference,
    )
    path_csv = root / "data" / "locked_extended_sweep.csv"
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    combined_sigma = np.r_[original["sigma_input"].to_numpy(), np.asarray([r["sigma_input"] for r in rows])]
    combined_rate = np.r_[original["survival_rate"].to_numpy(), np.asarray([r["survival_rate"] for r in rows])]
    threshold = crossing(combined_sigma, combined_rate)
    generated_sigma = np.r_[original["generated_sigma_mean"].to_numpy(), np.asarray([r["generated_sigma_mean"] for r in rows])]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(combined_sigma, combined_rate, "o-", ms=3.5, color="#2166ac")
    axes[0].axhline(0.5, color="0.4", ls=":")
    axes[0].axvline(float(protocol["thresholds"]["gt_sigma_50"]), color="#111111", ls="--", label="GT 50% reference")
    if np.isfinite(threshold):
        axes[0].axvline(threshold, color="#b2182b", ls="--", label=f"RC 50%: {threshold:.4f}")
    axes[0].set(ylabel="normal-survival rate", ylim=(-0.04, 1.04), title="Locked-model extended statistical sweep")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(combined_sigma, generated_sigma, "o-", ms=3.5, color="#1b7837", label="generated sigma")
    axes[1].plot(combined_sigma, combined_sigma, "--", color="0.35", label="identity")
    axes[1].set(xlabel="supplied sigma", ylabel="generated sigma", title="Extrapolation self-consistency")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "figures" / "07_locked_extended_sweep.png", dpi=240)
    plt.close(fig)

    result = {
        "model_reselected": False,
        "extended_sigma_max": args.sigma_max,
        "rc_sigma_50": threshold,
        "minimum_survival_rate": float(np.min(combined_rate)),
        "generated_sigma_at_max_input": float(generated_sigma[-1]),
    }
    with (root / "results" / "extended_sweep_summary.json").open("w") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
