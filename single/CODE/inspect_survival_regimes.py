#!/usr/bin/env python3
"""Reproduce selected locked-SCDT sweep batches and inspect their trajectories."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
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
    morphology_event_mask,
)


ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    ("normal", 7),
    ("50% crossing", 16),
    ("after collapse", 25),
    ("rebound in extrapolation", 31),
)


def choose_examples(survived: np.ndarray, phase: str) -> list[int]:
    alive = np.flatnonzero(survived)
    departed = np.flatnonzero(~survived)
    if phase in {"50% crossing", "rebound in extrapolation"}:
        return [*alive[:2], *departed[:2]]
    if phase == "normal":
        return alive[:4].tolist()
    return departed[:4].tolist()


def draw(path: Path, records: list[dict], zoom_s: float | None = None) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 11.5), sharex=True)
    for axis, record in zip(axes, records):
        predictions = record["prediction"]
        seconds = np.arange(len(predictions)) / record["sample_hz"]
        if zoom_s is not None:
            shown = seconds <= zoom_s
        else:
            shown = slice(None)
        for column in record["selected"]:
            survival = record["survived"][column]
            axis.plot(
                seconds[shown], predictions[shown, column], lw=0.85, alpha=0.86,
                color="#2171b5" if survival else "#cb181d",
                label="survived" if survival else "departed",
            )
        axis.axhline(EVENT_THRESHOLD, color="0.35", lw=0.9, ls="--")
        axis.set_title(
            f"{record['phase']}   σ={record['sigma']:.6f}   "
            f"survival={record['survivors']}/30",
            loc="left", fontsize=11,
        )
        axis.set_ylabel("proxy x₂ − x₁")
        axis.grid(alpha=0.18)
        if zoom_s is not None:
            axis.set_xlim(0, zoom_s)
    axes[-1].set_xlabel("autonomous time (s)")
    # One legend applies to every panel; repeated lines are intentionally omitted.
    handles, labels = [], []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    with (ROOT / "results/protocol.json").open() as handle:
        protocol = json.load(handle)
    with (ROOT / "data/scdt_rate_sweep.csv").open(newline="") as handle:
        sweep = list(csv.DictReader(handle))
    with np.load(ROOT / "data/training_signals.npz") as archive:
        training = [signal for signal in archive["signals"]]
        train_stats = archive["statistics"]
        sample_hz = float(archive["sample_hz"])

    dynamic_values = np.concatenate(training)
    dynamic_center = float(np.mean(dynamic_values))
    dynamic_scale = float(np.std(dynamic_values))
    static_center = np.mean(train_stats, axis=0)
    static_scale = np.std(train_stats, axis=0)
    static_scale = np.where(static_scale > 1e-9, static_scale, 1.0)
    sequences_norm = [(signal - dynamic_center) / dynamic_scale for signal in training]
    statics_norm = (train_stats - static_center) / static_scale
    morphology_reference = fit_morphology_reference(training, sample_hz)

    params = RCParams(**protocol["best_rc_parameters"])
    args = protocol["arguments"]
    model = build_rc(params, sequences_norm, statics_norm, int(args["fit_warmup"]))
    batch_size = int(args["rc_ensemble"])
    steps = int(round(float(args["rollout_s"]) * sample_hz))
    indices = np.arange(batch_size) % len(sequences_norm)
    warmups = np.column_stack([
        sequences_norm[index][-int(args["warmup_samples"]):] for index in indices
    ])
    warmup_statics = statics_norm[indices]
    records = []
    metadata = {
        "model_parameters": asdict(params),
        "selection": "fixed grid indices from original sweep; reservoir reconstructed with the locked parameters and original training records, without new BO",
        "event_rule": protocol["event_rule"],
        "training_record_time_median_range": [
            float(min(np.median(signal) for signal in training)),
            float(max(np.median(signal) for signal in training)),
        ],
        "phases": [],
    }
    saved_arrays = {}
    for phase, grid_index in PHASES:
        row = sweep[grid_index]
        sigma = float(row["sigma_input"])
        pe = float(row["permutation_entropy_input"])
        target = (np.asarray([sigma, pe]) - static_center) / static_scale
        targets = np.repeat(target[None, :], batch_size, axis=0)
        prediction_norm = model.rollout_switch_batch(
            warmups, warmup_statics, targets, steps,
            int(args["seed"]) + 300000 + grid_index,
        )
        prediction = prediction_norm * dynamic_scale + dynamic_center
        trough = np.any(
            prediction[int(round(EVENT_BLANK_S * sample_hz)):] <= EVENT_THRESHOLD,
            axis=0,
        )
        low_activity = morphology_event_mask(prediction, sample_hz, morphology_reference)
        survived = ~event_mask(prediction, sample_hz, morphology_reference)
        survivors = int(np.sum(survived))
        expected = int(row["survivors"])
        if survivors != expected:
            raise RuntimeError(
                f"{phase}: reconstructed survivors {survivors} != recorded {expected}"
            )
        selected = [int(index) for index in choose_examples(survived, phase)]
        clip_low_raw = model.clip_low * dynamic_scale + dynamic_center
        clip_high_raw = model.clip_high * dynamic_scale + dynamic_center
        clipped_fraction = float(np.mean(
            np.isclose(prediction, clip_low_raw, atol=1e-10)
            | np.isclose(prediction, clip_high_raw, atol=1e-10)
        ))
        survivor_medians = np.median(prediction[:, survived], axis=0) if survivors else np.asarray([])
        records.append({
            "phase": phase, "sigma": sigma, "survivors": survivors,
            "prediction": prediction, "survived": survived,
            "selected": selected, "sample_hz": sample_hz,
        })
        saved_arrays[phase.replace(" ", "_")] = prediction[:, selected]
        metadata["phases"].append({
            "name": phase,
            "grid_index": grid_index,
            "sigma": sigma,
            "survivors": survivors,
            "ensemble": batch_size,
            "trough_departures": int(np.sum(trough)),
            "low_activity_departures": int(np.sum(low_activity)),
            "selected_indices": selected,
            "selected_survived": survived[selected].tolist(),
            "selected_trough": trough[selected].tolist(),
            "selected_low_activity": low_activity[selected].tolist(),
            "clip_low_raw": float(clip_low_raw),
            "clip_high_raw": float(clip_high_raw),
            "clipped_sample_fraction": clipped_fraction,
            "survivor_time_median_median": (
                float(np.median(survivor_medians))
                if survivors else None
            ),
            "survivor_time_median_range": (
                [float(np.min(survivor_medians)), float(np.max(survivor_medians))]
                if survivors else None
            ),
        })
        print(
            f"{phase}: sigma={sigma:.6f}, survival={survivors}/{batch_size}, "
            f"trough={sum(trough)}, low_activity={sum(low_activity)}, "
            f"clip_fraction={clipped_fraction:.3f}",
            flush=True,
        )

    figure_dir = ROOT / "figures"
    draw(figure_dir / "07_survival_regime_trajectories.png", records)
    draw(figure_dir / "08_survival_regime_trajectories_zoom.png", records, zoom_s=25.0)
    np.savez_compressed(ROOT / "data/survival_regime_trajectories.npz", **saved_arrays)
    with (ROOT / "results/survival_regime_trajectories.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
