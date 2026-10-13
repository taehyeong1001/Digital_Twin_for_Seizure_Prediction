"""Rebuild the locked single-node SCDT setup from the original run's protocol.

`final_code/CODE/paper_single_node_scdt.py` is run unchanged once; its
`protocol.json` stores the selected statistics, the x0 -> statistics
mapping, the normalizers and the BO-selected RC parameters. Everything here
re-derives the training data and cohorts from the same seeds and calls the
original functions, so no modelling logic is duplicated.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final_code" / "CODE"))

import paper_single_node_scdt as scdt  # noqa: E402
from paper_single_node_scrc import RCParams, SparseStaticRC, fit_normalizer, normalize, static_from_x0  # noqa: E402

DEFAULT_PROTOCOL = (
    ROOT / "survival_dist" / "outputs" / "single_node" / "original"
    / "data" / "paper_single_node_scdt" / "protocol.json"
)


@dataclass
class SingleNodeSetup:
    args: argparse.Namespace
    params: RCParams
    selected: list[str]
    coefficients: np.ndarray
    dynamic_center: float
    dynamic_scale: float
    static_center: np.ndarray
    static_scale: np.ndarray
    train_sequences: list[np.ndarray]
    train_statics: list[np.ndarray]
    sample_hz: float

    @property
    def horizon(self) -> float:
        """Rollout length in seconds; events are timed from rollout start."""
        return float(self.args.rollout_s)

    def test_x0(self) -> np.ndarray:
        return np.linspace(self.args.x0_min, self.args.x0_max, self.args.survival_x0)

    def fit_member(self, seed: int | None = None) -> SparseStaticRC:
        """Fit an RC with the locked hyperparameters; `seed` replaces the reservoir seed."""
        params = self.params if seed is None else replace(self.params, seed=int(seed))
        rc = SparseStaticRC(params, static_dim=len(self.selected))
        rc.fit(self.train_sequences, self.train_statics, self.args.fit_warmup)
        return rc

    def cohort(self, x0: np.ndarray, per_x0: int, seed_base: int) -> scdt.SwitchCohort:
        """GT switch cohort (base-x0 warmup, then fixed target x0) via the original code."""
        return scdt.build_switch_cohort(
            np.asarray(x0, dtype=float),
            per_x0,
            self.args.base_x0,
            self.args.warmup_s,
            self.args.rollout_s,
            self.args.rc_stride,
            seed_base,
        )

    def rc_events(self, rc: SparseStaticRC, cohort: scdt.SwitchCohort, seed: int) -> np.ndarray:
        """RC event times (s from rollout start, NaN = none), one rollout per GT warmup."""
        return np.asarray(self.evaluate(rc, cohort, seed)["rc_events"], dtype=float)

    def evaluate(self, rc: SparseStaticRC, cohort: scdt.SwitchCohort, seed: int) -> dict:
        """Original `evaluate_rc` result (predictions in signal units, rc_events, ...)."""
        return scdt.evaluate_rc(
            rc,
            cohort,
            self.coefficients,
            self.dynamic_center,
            self.dynamic_scale,
            self.static_center,
            self.static_scale,
            self.args.base_x0,
            self.sample_hz,
            seed,
            self.args.ramp_steps,
        )


def load_setup(protocol_path: Path = DEFAULT_PROTOCOL) -> SingleNodeSetup:
    protocol = json.loads(Path(protocol_path).read_text())
    args = argparse.Namespace(**protocol["arguments"])
    selected = list(protocol["selected_features"])
    coefficients = np.asarray(protocol["feature_coefficients"], dtype=float)
    params = RCParams(**protocol["best_rc_parameters"])

    # Same steps as paper_single_node_scdt.main().
    raw_sequences, train_coordinates, _ = scdt.build_training_sequences(args)
    dynamic_values = np.concatenate(raw_sequences)
    dynamic_center = float(np.mean(dynamic_values))
    dynamic_scale = float(np.std(dynamic_values))
    if not (np.isclose(dynamic_center, protocol["dynamic_center"]) and np.isclose(dynamic_scale, protocol["dynamic_scale"])):
        raise RuntimeError("Rebuilt training data does not match the protocol normalizer")
    train_sequences = [normalize(s, dynamic_center, dynamic_scale) for s in raw_sequences]
    static_center, static_scale = fit_normalizer(static_from_x0(scdt.TRAIN_X0, coefficients))
    train_statics = list(normalize(static_from_x0(train_coordinates, coefficients), static_center, static_scale))
    return SingleNodeSetup(
        args=args,
        params=params,
        selected=selected,
        coefficients=coefficients,
        dynamic_center=dynamic_center,
        dynamic_scale=dynamic_scale,
        static_center=static_center,
        static_scale=static_scale,
        train_sequences=train_sequences,
        train_statics=train_statics,
        sample_hz=256.0 / args.rc_stride,
    )
