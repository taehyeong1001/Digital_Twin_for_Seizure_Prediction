#!/usr/bin/env python3
"""Ablate temporally coherent residual innovations for the locked SCDT RC."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from experiment_three_node_parameter_space_rc import fit_normalizer, trajectory_context
from experiment_three_node_scdt import (
    BASE_X0,
    DEFAULT_EVENT_BLANKING_S,
    DEFAULT_EVENT_THRESHOLD,
    N_NODES,
    RC_HZ,
    evaluate,
    first_threshold_crossing,
    fit_statistical_coordinate,
    normalize,
    plot_gt_rc_grid,
    plot_survival,
    simulate_switch_cohort,
)
from experiment_three_node_scdt_dynamics_bo import objective
from experiment_three_node_scdt_dynamics_pilot import shape_metrics
from experiment_three_node_scdt_empirical_boundary_bo import profiles_from_maximum
from paper_multinode_coupled_scdt import MultiOutputStaticRC
from paper_single_node_scrc import RCParams


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "three_node_scdt_empirical_boundary_bo100"
OUT = ROOT / "experiments" / "three_node_scdt_block_innovation_pilot"
FIG = ROOT / "figures" / "RC" / "three_node_scdt_block_innovation_pilot"
SEED = 732000
WARMUP_S = 40.0
FUTURE_S = 120.0
TEST_LEVELS = 16
TEST_SEEDS = 8


@dataclass(frozen=True)
class InnovationVariant:
    name: str
    mode: str
    scaling: float
    block_s: float = 0.0
    pool: str = "stable"


VARIANTS = (
    InnovationVariant("deterministic", "none", 0.0),
    InnovationVariant("iid_original_0.60", "iid_original", 0.60),
    InnovationVariant("iid_fullrate_stable_0.60", "iid_fullrate", 0.60),
    *tuple(
        InnovationVariant(
            f"block_{block_s:g}s_{scaling:.2f}",
            "block",
            scaling,
            block_s,
        )
        for scaling in (0.30, 0.60, 0.90)
        for block_s in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    ),
)


class BlockInnovationRC(MultiOutputStaticRC):
    """Locked multi-output RC with selectable residual innovation dynamics."""

    innovation_mode = "iid_original"
    innovation_block_steps = 1
    residual_sequences: np.ndarray | None = None
    residual_pool_indices: np.ndarray | None = None

    def rollout_switch_batch(
        self,
        warmup: np.ndarray,
        warmup_statics: np.ndarray,
        target_statics: np.ndarray,
        n_steps: int,
        seed: int,
        ramp_steps: int = 0,
    ) -> np.ndarray:
        if self.w_out is None or self.residuals is None:
            raise RuntimeError("RC must be fitted before rollout")
        if warmup.ndim != 3:
            raise ValueError("warmup must have shape (time, batch, dynamic_dim)")
        batch = warmup.shape[1]
        reservoir = np.zeros((self.params.n_reservoir, batch))
        warmup_static = warmup_statics.T
        target_static = target_statics.T
        for index in range(len(warmup)):
            reservoir = self.update(reservoir, warmup[index].T, warmup_static)
        current = warmup[-1].T
        prediction = np.empty((n_steps, batch, self.dynamic_dim), dtype=float)
        rng = np.random.default_rng(seed)
        block_sources = np.zeros(batch, dtype=int)
        block_starts = np.zeros(batch, dtype=int)
        pool = self.residual_pool_indices
        if pool is None:
            pool = np.arange(self.residual_sequences.shape[1]) if self.residual_sequences is not None else np.empty(0, dtype=int)
        for index in range(n_steps):
            if ramp_steps > 0 and index < ramp_steps:
                fraction = (index + 1) / ramp_steps
                static = (1.0 - fraction) * warmup_static + fraction * target_static
            else:
                static = target_static
            reservoir = self.update(reservoir, current, static)
            current = self.w_out @ self.augment(reservoir, current)
            if self.innovation_mode == "iid_original":
                residual_index = rng.integers(0, len(self.residuals), size=batch)
                innovations = self.residuals[residual_index].T
                current += self.params.innovation_scaling * innovations
            elif self.innovation_mode == "iid_fullrate":
                if self.residual_sequences is None:
                    raise RuntimeError("full-rate residual sequences are unavailable")
                times = rng.integers(0, len(self.residual_sequences), size=batch)
                sources = rng.choice(pool, size=batch)
                current += self.params.innovation_scaling * self.residual_sequences[times, sources].T
            elif self.innovation_mode == "block":
                if self.residual_sequences is None:
                    raise RuntimeError("full-rate residual sequences are unavailable")
                block_steps = self.innovation_block_steps
                offset = index % block_steps
                if offset == 0:
                    block_sources = rng.choice(pool, size=batch)
                    last_start = max(len(self.residual_sequences) - block_steps + 1, 1)
                    block_starts = rng.integers(0, last_start, size=batch)
                times = np.minimum(block_starts + offset, len(self.residual_sequences) - 1)
                current += self.params.innovation_scaling * self.residual_sequences[times, block_sources].T
            elif self.innovation_mode != "none":
                raise ValueError(f"unknown innovation mode: {self.innovation_mode}")
            current = np.clip(current, self.clip_low[:, None], self.clip_high[:, None])
            prediction[index] = current.T
        return prediction


def load_locked_model() -> tuple[BlockInnovationRC, RCParams, dict]:
    summary = json.loads((SOURCE / "summary.json").read_text())
    params = RCParams(**summary["selected_parameters"])
    model = BlockInnovationRC.__new__(BlockInnovationRC)
    with np.load(SOURCE / "winner_model.npz") as data:
        model.params = params
        model.dynamic_dim = N_NODES
        model.static_dim = 9
        model.readout_stride = 4
        model.readout_size = len(data["readout_indices"])
        model.readout_indices = data["readout_indices"].copy()
        model.w_in = data["w_in"].copy()
        model.w_static = data["w_static"].copy()
        model.w_out = data["w_out"].copy()
        model.residuals = data["residuals"].copy()
        model.clip_low = data["clip_low"].copy()
        model.clip_high = data["clip_high"].copy()
        model.a = sparse.csr_matrix(
            (data["a_data"], data["a_indices"], data["a_indptr"]),
            shape=tuple(data["a_shape"]),
        )
    return model, params, summary


def prepare_training(model: BlockInnovationRC):
    with np.load(SOURCE / "training.npz") as data:
        profiles = data["profiles"]
        stable_full = data["stable_signals"]
        observed_context = data["observed_context"]
        event_profiles = data["event_profiles"]
        event_raw = data["event_signals"]
    coordinate, fitted_context, _ = fit_statistical_coordinate(profiles, observed_context)
    dynamic_center, dynamic_scale = fit_normalizer(stable_full)
    static_center, static_scale = fit_normalizer(fitted_context)
    stable_raw = stable_full[-int(FUTURE_S * RC_HZ) :]
    stable_n = normalize(stable_raw, dynamic_center, dynamic_scale)
    event_n = normalize(event_raw, dynamic_center, dynamic_scale)
    train_n = np.concatenate((stable_n, event_n), axis=1)
    stable_static = normalize(fitted_context, static_center, static_scale)
    event_static = normalize(coordinate.predict(event_profiles), static_center, static_scale)
    train_static = np.concatenate((stable_static, event_static), axis=0)
    model.residual_sequences = sequential_residuals(model, train_n, train_static, 5 * RC_HZ)
    return coordinate, dynamic_center, dynamic_scale, static_center, static_scale, stable_n.shape[1]


def sequential_residuals(
    model: BlockInnovationRC,
    sequences: np.ndarray,
    statics: np.ndarray,
    warmup: int,
) -> np.ndarray:
    """Compute teacher-forced one-step residuals at the full 32 Hz cadence."""
    batch = sequences.shape[1]
    state = np.zeros((model.params.n_reservoir, batch))
    static_batch = statics.T
    residuals = np.empty((len(sequences) - warmup - 1, batch, model.dynamic_dim), dtype=np.float32)
    output = 0
    for index in range(len(sequences) - 1):
        current = sequences[index]
        state = model.update(state, current.T, static_batch)
        if index < warmup:
            continue
        predicted = model.w_out @ model.augment(state, current.T)
        residuals[output] = sequences[index + 1] - predicted.T
        output += 1
    lower = np.quantile(residuals, 0.0025, axis=(0, 1))
    upper = np.quantile(residuals, 0.9975, axis=(0, 1))
    return np.clip(residuals, lower, upper)


def configure(model: BlockInnovationRC, base_params: RCParams, variant: InnovationVariant, stable_count: int) -> None:
    model.params = replace(base_params, innovation_scaling=variant.scaling)
    model.innovation_mode = variant.mode
    model.innovation_block_steps = max(1, int(round(variant.block_s * RC_HZ)))
    if model.residual_sequences is not None:
        upper = stable_count if variant.pool == "stable" else model.residual_sequences.shape[1]
        model.residual_pool_indices = np.arange(upper)


def metric_row(name: str, variant: InnovationVariant, metrics: dict, shape: dict) -> dict:
    return {
        "name": name,
        "mode": variant.mode,
        "pool": variant.pool,
        "block_s": variant.block_s,
        "innovation_scaling": variant.scaling,
        **{key: value for key, value in metrics.items() if key not in ("gt_survival", "rc_survival")},
        **shape,
        "objective": objective(metrics, shape),
    }


def plot_tradeoff(rows: list[dict], selected: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for row in rows:
        marker = "*" if row["name"] == selected else "o"
        size = 180 if marker == "*" else 55
        axes[0].scatter(row["normal_spike_rate_mae"], row["survival_mae"], s=size, marker=marker, label=row["name"])
        axes[1].scatter(row["normal_psd_l1"], row["morphology_score"], s=size, marker=marker)
    axes[0].set(xlabel="normal spike-rate MAE", ylabel="survival MAE", title="Peak–boundary tradeoff")
    axes[1].set(xlabel="normal PSD L1", ylabel="morphology score", title="Normal waveform fidelity")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.tight_layout()
    fig.savefig(FIG / "innovation_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_normal_zoom(
    gt: np.ndarray,
    predictions: dict[str, np.ndarray],
    gt_events: np.ndarray,
    figdir: Path = FIG,
) -> None:
    stable = np.flatnonzero(~np.isfinite(gt_events))
    record = int(stable[len(stable) // 2])
    start, stop = 10 * RC_HZ, 40 * RC_HZ
    time = np.arange(stop - start) / RC_HZ
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for node, ax in enumerate(axes):
        ax.plot(time, gt[start:stop, record, node], color="black", lw=1.2, label="GT")
        for name, prediction in predictions.items():
            ax.plot(time, prediction[start:stop, record, node], lw=1.0, alpha=0.85, label=name)
        ax.set_ylabel(f"node {node + 1}")
        ax.grid(alpha=0.18)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("time after switch (s)")
    fig.suptitle("Normal waveform: temporal residual structure ablation")
    fig.tight_layout()
    fig.savefig(figdir / "normal_waveform_zoom.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    model, base_params, source_summary = load_locked_model()
    coordinate, dynamic_center, dynamic_scale, static_center, static_scale, stable_count = prepare_training(model)
    with np.load(SOURCE / "validation.npz") as data:
        validation = {name: data[name] for name in data.files}
    profile_count = len(validation["targets"])
    replicate_count = len(validation["profiles"]) // profile_count
    metrics_path = OUT / "validation_metrics.csv"
    if metrics_path.exists():
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        numeric = set(rows[0]) - {"name", "mode", "pool"}
        rows = [{key: (float(value) if key in numeric else value) for key, value in row.items()} for row in rows]
    else:
        rows = []
        for variant in VARIANTS:
            configure(model, base_params, variant, stable_count)
            metrics, prediction, _ = evaluate(
                model,
                validation["warmup"],
                validation["future"],
                validation["profiles"],
                coordinate,
                dynamic_center,
                dynamic_scale,
                static_center,
                static_scale,
                profile_count,
                replicate_count,
                SEED + 100,
                DEFAULT_EVENT_THRESHOLD,
                DEFAULT_EVENT_BLANKING_S,
            )
            shape = shape_metrics(validation["future"], prediction)
            row = metric_row(variant.name, variant, metrics, shape)
            rows.append(row)
            print(json.dumps(row), flush=True)
        with metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    feasible = [
        index for index, row in enumerate(rows)
        if row["sustained_hits"] >= 0.75 * (row["sustained_hits"] + row["sustained_misses"])
        and row["sustained_false_alarms"] <= 0.10 * (row["gt_exits"] + row["correct_nonexits"])
        and row["false_alarms"] <= 0.10 * (row["gt_exits"] + row["correct_nonexits"])
    ]
    selected_index = min(feasible, key=lambda index: rows[index]["objective"])
    selected_variant = VARIANTS[selected_index]
    configure(model, base_params, selected_variant, stable_count)

    x50 = float(source_summary["boundary_calibration"]["empirical_x50"])
    test_maxima = x50 + np.linspace(-0.12, 0.16, TEST_LEVELS)
    test_targets = profiles_from_maximum(test_maxima, SEED + 200)
    test_seed_values = tuple(SEED + 300 + index for index in range(TEST_SEEDS))
    test_warmup, test_future, test_profiles, test_seed_ids = simulate_switch_cohort(
        test_targets,
        test_seed_values,
        WARMUP_S,
        FUTURE_S,
        event_threshold=DEFAULT_EVENT_THRESHOLD,
    )
    test_metrics, test_prediction, test_rc_events = evaluate(
        model,
        test_warmup,
        test_future,
        test_profiles,
        coordinate,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        TEST_LEVELS,
        TEST_SEEDS,
        SEED + 400,
        DEFAULT_EVENT_THRESHOLD,
        DEFAULT_EVENT_BLANKING_S,
    )
    test_shape = shape_metrics(test_future, test_prediction)
    test_row = metric_row(selected_variant.name, selected_variant, test_metrics, test_shape)

    # Evaluate the old IID method on the same new cohort for a paired comparison.
    baseline_variant = VARIANTS[1]
    configure(model, base_params, baseline_variant, stable_count)
    baseline_metrics, baseline_prediction, _ = evaluate(
        model,
        test_warmup,
        test_future,
        test_profiles,
        coordinate,
        dynamic_center,
        dynamic_scale,
        static_center,
        static_scale,
        TEST_LEVELS,
        TEST_SEEDS,
        SEED + 400,
        DEFAULT_EVENT_THRESHOLD,
        DEFAULT_EVENT_BLANKING_S,
    )
    baseline_shape = shape_metrics(test_future, baseline_prediction)
    baseline_row = metric_row(baseline_variant.name, baseline_variant, baseline_metrics, baseline_shape)

    summary = {
        "status": "complete",
        "source_experiment": SOURCE.name,
        "selection_data": "unchanged source validation cohort",
        "selection_constraint": "at least 75% sustained sensitivity and at most 10% pointwise and sustained false-alarm fractions",
        "selected_variant": selected_variant.__dict__,
        "selected_validation": rows[selected_index],
        "fresh_test_seeds": list(test_seed_values),
        "fresh_test": test_row,
        "paired_iid_baseline": baseline_row,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "protocol.json").write_text(
        json.dumps(
            {
                "date": "2026-09-12",
                "purpose": "test whether full-rate residual block bootstrap restores autonomous normal peaks without losing normal-departure discrimination",
                "reservoir_and_readout": "locked empirical-boundary BO100 winner; no refit or hyperparameter change",
                "validation_variants": len(VARIANTS),
                "block_lengths_s": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
                "block_scalings": [0.30, 0.60, 0.90],
                "block_pool": "48 stable training trajectories only",
                "selection_constraint": "sustained sensitivity >=75%; pointwise and sustained false alarms <=10% of records",
                "fresh_test_seed_family": list(test_seed_values),
                "event": "any node <= -0.5 after 5 s; sustained endpoint additionally requires a 5 s continuous low run",
            },
            indent=2,
        )
    )
    np.savez_compressed(
        OUT / "fresh_test.npz",
        targets=test_targets,
        profiles=test_profiles,
        seeds=test_seed_ids,
        warmup=test_warmup,
        future=test_future,
        selected_prediction=test_prediction,
        baseline_prediction=baseline_prediction,
        selected_rc_events=test_rc_events,
    )
    plot_tradeoff(rows, selected_variant.name)
    plot_survival(FIG, test_targets, test_metrics, "Block innovation fresh test")
    plot_gt_rc_grid(
        FIG,
        test_warmup,
        test_future,
        test_prediction,
        test_profiles,
        DEFAULT_EVENT_THRESHOLD,
        DEFAULT_EVENT_BLANKING_S,
    )
    gt_events = first_threshold_crossing(
        test_future,
        DEFAULT_EVENT_THRESHOLD,
        DEFAULT_EVENT_BLANKING_S,
    )
    plot_normal_zoom(
        test_future,
        {
            "IID residual": baseline_prediction,
            "selected block": test_prediction,
        },
        gt_events,
    )
    (OUT / "RESULTS.md").write_text(
        "# Three-node SCDT block-innovation pilot\n\n"
        f"Selected `{selected_variant.name}` on the unchanged validation cohort, then evaluated it on {TEST_LEVELS * TEST_SEEDS} newly simulated records.\n\n"
        f"- Selected fresh sustained hits/misses/false alarms: {test_shape['sustained_hits']}/{test_shape['sustained_misses']}/{test_shape['sustained_false_alarms']}\n"
        f"- Selected fresh survival MAE / spike-rate MAE / morphology: {test_metrics['survival_mae']:.4f} / {test_metrics['normal_spike_rate_mae']:.4f} / {test_metrics['morphology_score']:.4f}\n"
        f"- IID fresh survival MAE / spike-rate MAE / morphology: {baseline_metrics['survival_mae']:.4f} / {baseline_metrics['normal_spike_rate_mae']:.4f} / {baseline_metrics['morphology_score']:.4f}\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
