"""Seed ensemble of the locked single-node SCDT-RC, with per-member checkpoints.

Port of EESN `ensemble/training.jl::train_ensemble_checkpointed`, adapted to the
static-conditioned RC:
- every member shares the BO-locked hyperparameters; only `RCParams.seed`
  (which draws W_in, W_static and the reservoir matrix) changes;
- member 0 is the original BO-selected reservoir with the original innovation
  seed, so it reproduces the single-RC result exactly;
- each member is fitted, rolled out on every cohort (one rollout per GT
  warmup, paired), and saved to `member_XXXX.npz` (tmp file then rename, so a
  killed run never leaves a corrupt file); finished members are skipped on
  restart, and a fingerprint guards against resuming with different settings.

No refill / no resync (docs/03_progress_log.md): every member is kept as is.
Numerical health per member is recorded (non-finite trials, clip-saturated
steps) because NaN passes `np.clip` and would be scored as "survived".
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from single_node_setup import SingleNodeSetup, load_setup, scdt

ORIGINAL_ROLLOUT_SEED_OFFSET = 90000  # args.seed + 90000 in paper_single_node_scdt.main


@dataclass(frozen=True)
class CohortSpec:
    name: str
    x0: tuple[float, ...]
    per_x0: int
    seed_base: int
    rollout_seed_offset: int  # added to the member's rollout seed for this cohort


def member_seed(setup: SingleNodeSetup, j: int) -> int:
    """Reservoir seed of member j (member 0 = BO-selected reservoir)."""
    return int(setup.params.seed) + 10007 * j


def rollout_seed(setup: SingleNodeSetup, j: int, spec: CohortSpec) -> int:
    """Innovation-noise seed; member 0 on the original cohort = original seed."""
    return int(setup.args.seed) + ORIGINAL_ROLLOUT_SEED_OFFSET + 1000 * j + spec.rollout_seed_offset


def load_or_build_cohort(setup: SingleNodeSetup, spec: CohortSpec, cache_dir: Path) -> scdt.SwitchCohort:
    """GT cohort, cached as npz (warmups + futures + events)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()[:12]
    path = cache_dir / f"cohort_{spec.name}_{key}.npz"
    if path.exists():
        d = np.load(path)
        return scdt.SwitchCohort(d["x0"], d["warmups_raw"], d["futures_raw"], d["gt_events"])
    cohort = setup.cohort(np.asarray(spec.x0), spec.per_x0, spec.seed_base)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, x0=cohort.x0, warmups_raw=cohort.warmups_raw, futures_raw=cohort.futures_raw, gt_events=cohort.gt_events)
    os.replace(tmp, path)
    return cohort


def fingerprint(setup: SingleNodeSetup, specs: list[CohortSpec]) -> str:
    params = asdict(setup.params)
    params.pop("seed")
    payload = {
        "params": params,
        "selected": setup.selected,
        "coefficients": np.round(setup.coefficients, 12).tolist(),
        "specs": [asdict(s) for s in specs],
        "rollout_offset": ORIGINAL_ROLLOUT_SEED_OFFSET,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---- worker -----------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(specs: list[CohortSpec], cache_dir: str) -> None:
    setup = load_setup()
    cohorts = {s.name: load_or_build_cohort(setup, s, Path(cache_dir)) for s in specs}
    _WORKER.update(setup=setup, cohorts=cohorts, specs=specs)


def _run_member(args: tuple[int, str, str]) -> str:
    j, out_dir, fp = args
    path = Path(out_dir) / f"member_{j:04d}.npz"
    if path.exists():
        return f"member {j}: cached"
    setup: SingleNodeSetup = _WORKER["setup"]
    seed = member_seed(setup, j)
    rc = setup.fit_member(seed)
    payload: dict[str, np.ndarray] = {"member": np.array(j), "reservoir_seed": np.array(seed), "fingerprint": np.array(fp)}
    for spec in _WORKER["specs"]:
        result = setup.evaluate(rc, _WORKER["cohorts"][spec.name], rollout_seed(setup, j, spec))
        payload.update(_rollout_record(setup, rc, result, spec.name))
    _save(path, payload)
    return f"member {j}: done (seed {seed})"


def _rollout_record(setup: SingleNodeSetup, rc, result: dict, name: str) -> dict[str, np.ndarray]:
    """Events and numerical health of one cohort rollout, keyed by cohort name."""
    pred = result["predictions"]
    finite = np.isfinite(pred)
    at_clip = (pred <= rc.clip_low * setup.dynamic_scale + setup.dynamic_center + 1e-9) | (
        pred >= rc.clip_high * setup.dynamic_scale + setup.dynamic_center - 1e-9
    )
    first_clip = np.where(at_clip.any(axis=0), np.argmax(at_clip, axis=0) / setup.sample_hz, np.nan)
    return {
        f"{name}_events": np.asarray(result["rc_events"], dtype=float),
        f"{name}_nonfinite": ~finite.all(axis=0),
        f"{name}_clip_fraction": at_clip.mean(axis=0),
        f"{name}_first_clip_s": first_clip,  # s from rollout start, NaN = never
    }


def _save(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def train_ensemble_checkpointed(
    n_members: int,
    specs: list[CohortSpec],
    out_dir: Path,
    workers: int = 6,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    setup = load_setup()
    # build GT cohorts once in the parent so workers only read the cache
    for spec in specs:
        load_or_build_cohort(setup, spec, out_dir / "cohorts")
    fp = fingerprint(setup, specs)
    meta = out_dir / "checkpoint_meta.json"
    if meta.exists():
        stored = json.loads(meta.read_text())["fingerprint"]
        if stored != fp:
            raise RuntimeError(f"{out_dir} holds members from different settings; use a new directory")
    else:
        meta.write_text(json.dumps({"fingerprint": fp, "specs": [asdict(s) for s in specs]}, indent=2))
    jobs = [(j, str(out_dir), fp) for j in range(n_members)]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(specs, str(out_dir / "cohorts"))) as pool:
        for message in pool.map(_run_member, jobs):
            print(message, flush=True)
    return [out_dir / f"member_{j:04d}.npz" for j in range(n_members)]


def load_members(out_dir: Path, n_members: int, cohort_name: str, prefix: str = "member") -> dict[str, np.ndarray]:
    """Stack per-member arrays: events (n_members, n_trials), health flags."""
    events, nonfinite, clip, first_clip = [], [], [], []
    for j in range(n_members):
        d = np.load(out_dir / f"{prefix}_{j:04d}.npz")
        events.append(d[f"{cohort_name}_events"])
        nonfinite.append(d[f"{cohort_name}_nonfinite"])
        clip.append(d[f"{cohort_name}_clip_fraction"])
        first_clip.append(d[f"{cohort_name}_first_clip_s"])
    return {
        "events": np.vstack(events),
        "nonfinite": np.vstack(nonfinite),
        "clip_fraction": np.vstack(clip),
        "first_clip_s": np.vstack(first_clip),
    }


# ---- arm B: one reservoir, K innovation-noise rollouts (docs/04) -------------
#
# Control for the ensemble: same number of rollouts per trial as a K-member
# ensemble, but every rollout uses the member-0 reservoir, so only the
# innovation noise differs. Rollout 0 reuses member 0's seed (it must equal
# member 0 exactly); rollout k >= 1 uses member k's rollout seed + 250, which
# collides with no ensemble seed (those are base + 1000 j and base + 1000 j + 500).

ARM_B_SEED_SHIFT = 250


def arm_b_rollout_seed(setup: SingleNodeSetup, k: int, spec: CohortSpec) -> int:
    seed = rollout_seed(setup, k, spec)
    return seed if k == 0 else seed + ARM_B_SEED_SHIFT


def _init_arm_b(specs: list[CohortSpec], cache_dir: str) -> None:
    _init_worker(specs, cache_dir)
    setup = _WORKER["setup"]
    _WORKER["rc"] = setup.fit_member(member_seed(setup, 0))


def _run_arm_b(args: tuple[int, str, str]) -> str:
    k, out_dir, fp = args
    path = Path(out_dir) / f"rollout_{k:04d}.npz"
    if path.exists():
        return f"arm B rollout {k}: cached"
    setup: SingleNodeSetup = _WORKER["setup"]
    rc = _WORKER["rc"]
    payload: dict[str, np.ndarray] = {"rollout": np.array(k), "fingerprint": np.array(fp)}
    for spec in _WORKER["specs"]:
        seed = arm_b_rollout_seed(setup, k, spec)
        result = setup.evaluate(rc, _WORKER["cohorts"][spec.name], seed)
        payload.update(_rollout_record(setup, rc, result, spec.name))
        payload[f"{spec.name}_innovation_seed"] = np.array(seed)
    _save(path, payload)
    return f"arm B rollout {k}: done"


def run_arm_b(n_rollouts: int, specs: list[CohortSpec], out_dir: Path, cache_dir: Path, workers: int = 6) -> None:
    """Member-0 reservoir x `n_rollouts` innovation seeds, one file per rollout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    setup = load_setup()
    for spec in specs:
        load_or_build_cohort(setup, spec, cache_dir)
    fp = fingerprint(setup, specs)
    meta = out_dir / "checkpoint_meta.json"
    if meta.exists():
        if json.loads(meta.read_text())["fingerprint"] != fp:
            raise RuntimeError(f"{out_dir} holds rollouts from different settings; use a new directory")
    else:
        meta.write_text(json.dumps({"fingerprint": fp, "seed_shift": ARM_B_SEED_SHIFT, "specs": [asdict(s) for s in specs]}, indent=2))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    jobs = [(k, str(out_dir), fp) for k in range(n_rollouts)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_arm_b, initargs=(specs, str(cache_dir))) as pool:
        for message in pool.map(_run_arm_b, jobs):
            print(message, flush=True)


def load_arm_b(out_dir: Path, n_rollouts: int, cohort_name: str) -> dict[str, np.ndarray]:
    """Same layout as `load_members`, rows = rollouts of the member-0 reservoir."""
    return load_members(out_dir, n_rollouts, cohort_name, prefix="rollout")
