"""Seed ensemble + distributional report for a finished multi-node SCDT run.

The run directory is an isolated copy of final_code/CODE in which
`run_multinode_pipeline.sh` has run the BO stages. Locked model (`--model`):
  rate : block-BO winner      (experiments/three_node_scdt_block_bo100/summary.json)
  peak : peak-weighted winner (experiments/three_node_scdt_block_peak_bo100/summary.json)
Both are scored on the peak stage's fresh test cohort
(experiments/three_node_scdt_block_peak_bo100/fresh_test.npz), where the
original code already evaluated them with innovation seed SEED + 3200
(`prediction` = peak model, `baseline_prediction` = rate model).
The final confirmation stage selected no model in our rerun (docs/03), so the
published "robust" model is not available.
(The two-node variant lives in a patched copy with N_NODES = 2; module names
stay the same, so each variant must be analysed in its own process.)

Members share the locked hyperparameters (`selected_parameters`,
`selected_block_s`); only RCParams.seed changes. Member 0 uses the original
seed and the original innovation seed (exp.SEED + 3200), so it reproduces the
locked fresh-test result exactly (gate). Note: the multi-node BO kept the
reservoir seed fixed (SEED + 900) for every trial, so this ensemble asks whether
hyperparameters chosen on one reservoir carry over to others.

Run:  .venv/bin/python survival_dist/run_multinode_ensemble.py --run-dir survival_dist/outputs/three_node/run --label three_node --model rate
Arm B (docs/04): add `--arm-b 20` (runs only the member-0 reservoir x 20 innovation rollouts).
Dense transition cohort (docs/05 §1b): `--cohort fine --arm-b 20` runs members and arm B on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HORIZONS = (30.0, 60.0, 90.0, 120.0)
ORACLE_SEED_OFFSET = 50_000  # independent of every seed used by the pipeline (SEED + 0 ... SEED + 4107)
_W: dict = {}


def _import_run(run_dir: Path):
    sys.path.insert(0, str(run_dir / "CODE"))
    import experiment_three_node_scdt_block_peak_bo100 as exp  # noqa: E402
    from experiment_three_node_scdt import (  # noqa: E402
        DEFAULT_EVENT_BLANKING_S, DEFAULT_EVENT_THRESHOLD, N_NODES, first_threshold_crossing, simulate_switch_cohort,
    )
    from paper_single_node_scrc import RCParams  # noqa: E402
    return exp, DEFAULT_EVENT_THRESHOLD, DEFAULT_EVENT_BLANKING_S, N_NODES, first_threshold_crossing, simulate_switch_cohort, RCParams


MODEL_SUMMARY = {
    "rate": "three_node_scdt_block_bo100",
    "peak": "three_node_scdt_block_peak_bo100",
}
TEST_COHORT = "three_node_scdt_block_peak_bo100"

# Dense transition cohort (docs/05 §1b): more trials per level near the
# critical value, so one reservoir's critical value is less noisy. GT seeds and
# profile seed are unused elsewhere (pipeline: SEED + 0 ... SEED + 4107, oracle:
# SEED + 50000 ...). Innovation seeds get +600 so they differ from the test-cohort ones.
FINE_MAXIMA = np.round(np.arange(-2.08, -1.95 + 1e-9, 0.005), 4)
FINE_SEEDS = 32
FINE_PROFILE_SEED_OFFSET = 6000
FINE_GT_SEED_OFFSET = 6100
COHORTS = {"test": {"suffix": "", "innovation_offset": 0}, "fine": {"suffix": "_fine", "innovation_offset": 600}}


def cohort_path(run_dir: Path, cohort: str) -> Path:
    return run_dir / "experiments" / TEST_COHORT / "fresh_test.npz" if cohort == "test" else run_dir / "fine_cohort.npz"


def build_fine_cohort(run_dir: Path) -> None:
    path = cohort_path(run_dir, "fine")
    if path.exists():
        return
    exp, thr, _, _, _, sim, _ = _import_run(run_dir)
    from experiment_three_node_scdt_block_innovation import profiles_from_maximum  # noqa: E402
    targets = profiles_from_maximum(FINE_MAXIMA, exp.SEED + FINE_PROFILE_SEED_OFFSET)
    seeds = tuple(exp.SEED + FINE_GT_SEED_OFFSET + i for i in range(FINE_SEEDS))
    warmup, future, profiles, seed_ids = sim(targets, seeds, exp.WARMUP_S, exp.FUTURE_S, event_threshold=thr)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, targets=targets, maxima=FINE_MAXIMA, profiles=profiles, seeds=seed_ids, warmup=warmup, future=future)
    os.replace(tmp, path)


def _init(run_dir: str, model: str, cohort: str = "test") -> None:
    exp, thr, blank, n_nodes, ftc, sim, RCParams = _import_run(Path(run_dir))
    summary = json.loads((Path(run_dir) / "experiments" / MODEL_SUMMARY[model] / "summary.json").read_text())
    coordinate, _, train_raw, train_n, train_static, normalizers, stable_count = exp.prepare_training()
    with np.load(cohort_path(Path(run_dir), cohort)) as d:
        test = {k: d[k] for k in ("targets", "maxima", "profiles", "seeds", "warmup", "future")}
    sfx = COHORTS[cohort]["suffix"]
    _W.update(exp=exp, params=RCParams(**summary["selected_parameters"]), block_s=float(summary["selected_block_s"]),
              coordinate=coordinate, training=(train_n, train_raw, train_static, stable_count),
              normalizers=normalizers, test=test, ens_dir=Path(run_dir) / f"ensemble_{model}{sfx}",
              innovation_base=exp.SEED + 3200 + COHORTS[cohort]["innovation_offset"])


def _member(j: int) -> str:
    out = _W["ens_dir"] / f"member_{j:04d}.npz"
    if out.exists():
        return f"member {j}: cached"
    exp = _W["exp"]
    params = replace(_W["params"], seed=int(_W["params"].seed) + 10007 * j)
    train_n, train_raw, train_static, stable_count = _W["training"]
    model, _ = exp.fit_block_model(params, _W["block_s"], train_n, train_raw, train_static, stable_count)
    metrics, prediction, rc_events = exp.evaluate_cohort(model, _W["test"], _W["coordinate"], _W["normalizers"], _W["innovation_base"] + 1000 * j)
    dc, ds = _W["normalizers"][0], _W["normalizers"][1]
    low, high = model.clip_low * ds + dc, model.clip_high * ds + dc
    at_clip = ((prediction <= low + 1e-9) | (prediction >= high - 1e-9)).any(axis=2)  # (T, N)
    first_clip = np.where(at_clip.any(axis=0), np.argmax(at_clip, axis=0) / 32.0, np.nan)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, events=np.asarray(rc_events, float), first_clip_s=first_clip,
                        nonfinite=~np.isfinite(prediction).all(axis=(0, 2)), clipping_fraction=metrics["clipping_fraction"],
                        reservoir_seed=params.seed)
    os.replace(tmp, out)
    return f"member {j}: done (seed {params.seed}, clip fraction {metrics['clipping_fraction']:.4f})"


# ---- arm B: member-0 reservoir x K innovation rollouts (docs/04) ------------
# Rollout 0 = member 0 (seed SEED + 3200); rollout k >= 1 uses member k's
# innovation seed + 250, which no ensemble member uses.

ARM_B_SEED_SHIFT = 250


def _init_arm_b(run_dir: str, model: str, cohort: str = "test") -> None:
    _init(run_dir, model, cohort)
    train_n, train_raw, train_static, stable_count = _W["training"]
    _W["model0"], _ = _W["exp"].fit_block_model(_W["params"], _W["block_s"], train_n, train_raw, train_static, stable_count)
    _W["arm_b_dir"] = Path(run_dir) / f"armB_{model}{COHORTS[cohort]['suffix']}"


def _arm_b(k: int) -> str:
    out = _W["arm_b_dir"] / f"rollout_{k:04d}.npz"
    if out.exists():
        return f"arm B rollout {k}: cached"
    exp, model = _W["exp"], _W["model0"]
    seed = _W["innovation_base"] + 1000 * k + (ARM_B_SEED_SHIFT if k else 0)
    metrics, prediction, rc_events = exp.evaluate_cohort(model, _W["test"], _W["coordinate"], _W["normalizers"], seed)
    dc, ds = _W["normalizers"][0], _W["normalizers"][1]
    low, high = model.clip_low * ds + dc, model.clip_high * ds + dc
    at_clip = ((prediction <= low + 1e-9) | (prediction >= high - 1e-9)).any(axis=2)
    first_clip = np.where(at_clip.any(axis=0), np.argmax(at_clip, axis=0) / 32.0, np.nan)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, events=np.asarray(rc_events, float), first_clip_s=first_clip,
                        nonfinite=~np.isfinite(prediction).all(axis=(0, 2)), clipping_fraction=metrics["clipping_fraction"],
                        innovation_seed=seed)
    os.replace(tmp, out)
    return f"arm B rollout {k}: done (innovation seed {seed})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", choices=sorted(MODEL_SUMMARY), required=True)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--arm-b", type=int, default=0, metavar="K",
                    help="also run arm B (member-0 reservoir x K innovation rollouts) and save cross-table inputs")
    ap.add_argument("--cohort", choices=sorted(COHORTS), default="test",
                    help="test = the pipeline's fresh test (33 levels x 8); fine = dense transition cohort (docs/05 §1b)")
    a = ap.parse_args()
    run_dir = a.run_dir.resolve()
    sfx = COHORTS[a.cohort]["suffix"]
    ens_dir = run_dir / f"ensemble_{a.model}{sfx}"
    ens_dir.mkdir(exist_ok=True)
    exp, thr, blank, n_nodes, ftc, sim, _ = _import_run(run_dir)
    report_dir = HERE / "outputs" / a.label / f"ensemble_report_{a.model}"
    if a.cohort == "fine":
        build_fine_cohort(run_dir)

    # BLAS threads are left at their default: the original pipeline ran
    # multi-threaded, and a different thread count changes float summation
    # order enough to shift some departure times (docs/03).
    if not a.analyze_only and (a.cohort == "fine" or not a.arm_b):
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(str(run_dir), a.model, a.cohort)) as pool:
            for msg in pool.map(_member, range(a.members)):
                print(msg, flush=True)
    if a.arm_b and not a.analyze_only:
        (run_dir / f"armB_{a.model}{sfx}").mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_arm_b, initargs=(str(run_dir), a.model, a.cohort)) as pool:
            for msg in pool.map(_arm_b, range(a.arm_b)):
                print(msg, flush=True)

    with np.load(cohort_path(run_dir, a.cohort)) as d:
        test = {k: d[k] for k in d.files}
    x_trial = np.max(test["profiles"], axis=1)
    gt_events = ftc(test["future"], thr, blank)
    members = [np.load(ens_dir / f"member_{j:04d}.npz") for j in range(a.members)]
    rc_events = np.vstack([m["events"] for m in members])
    if a.cohort == "fine":
        # no stored single-RC result on this cohort; the arm-B gate (rollout 0 == member 0)
        # is checked in critical_values.py / cross_table.py instead. No oracle cohort.
        np.savez_compressed(report_dir.parent / f"cross_inputs_{a.model}{sfx}.npz", x_trial=x_trial, gt_events=gt_events,
                            oracle_x=np.empty(0), oracle_events=np.empty(0), horizon=float(exp.FUTURE_S), blanking_s=float(blank),
                            ensemble_dir=str(ens_dir), arm_b_dir=str(run_dir / f"armB_{a.model}{sfx}"))
        print(f"fine cohort: {len(np.unique(x_trial))} levels x {len(x_trial) // len(np.unique(x_trial))} trials; inputs saved", flush=True)
        return
    stored = test["rc_events"] if a.model == "peak" else ftc(test["baseline_prediction"], thr, blank)
    gate = bool(np.allclose(rc_events[0], stored, equal_nan=True))
    print(f"gate member0 == stored {a.model}-model fresh-test RC: {gate}", flush=True)

    oracle_path = run_dir / "oracle_cohort.npz"
    if oracle_path.exists():
        with np.load(oracle_path) as d:
            oracle_x, oracle_events = d["x"], d["events"]
    else:
        seeds = tuple(exp.SEED + ORACLE_SEED_OFFSET + i for i in range(exp.TEST_SEEDS))
        _, fut, prof, _ = sim(test["targets"], seeds, exp.WARMUP_S, exp.FUTURE_S, event_threshold=thr)
        oracle_x, oracle_events = np.max(prof, axis=1), ftc(fut, thr, blank)
        np.savez_compressed(oracle_path, x=oracle_x, events=oracle_events)

    # plain arrays for cross_table.py, which must not import the run modules
    # (two-node and three-node copies share module names)
    np.savez_compressed(report_dir.parent / f"cross_inputs_{a.model}.npz", x_trial=x_trial, gt_events=gt_events,
                        oracle_x=oracle_x, oracle_events=oracle_events, horizon=float(exp.FUTURE_S), blanking_s=float(blank),
                        ensemble_dir=str(ens_dir), arm_b_dir=str(run_dir / f"armB_{a.model}"))
    if a.arm_b:
        return

    from dist_report import ensemble_report  # noqa: E402

    res = ensemble_report(
        x_trial=x_trial, gt_events=gt_events, rc_events=rc_events,
        first_clip=np.vstack([m["first_clip_s"] for m in members]),
        nonfinite=np.vstack([m["nonfinite"] for m in members]),
        oracle_x=oracle_x, oracle_events=oracle_events,
        horizon=float(exp.FUTURE_S), horizons=HORIZONS,
        out_dir=report_dir,
        x_label="max x0", title=f"{n_nodes}-node, {a.model}-BO model", blanking_s=float(blank),
    )
    res["gate_member0_equals_locked_rc"] = gate
    res["member_clipping_fraction"] = [float(m["clipping_fraction"]) for m in members]
    (report_dir / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("gate_member0_equals_locked_rc", "departure_types", "time")}, indent=2, default=float))


if __name__ == "__main__":
    main()
