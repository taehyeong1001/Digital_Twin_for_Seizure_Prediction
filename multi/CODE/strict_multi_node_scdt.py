#!/usr/bin/env python3
"""Strict three-node SCDT experiment.

Follows SCDT_7 (Sec. 2.3) with the separation rules of ../single:
  * the hidden drift lam (x0_i = lam + offset_i) generates records and aligns a
    validation axis only; it never enters the reservoir, the statistic choice,
    the path, or model selection;
  * reservoir inputs are the three node proxies plus two statistics measured
    directly from each normal training record;
  * training and hyperparameter selection use normal (event-free) records only;
  * the statistical sweep, controls and GT boundary are evaluated after locking.
Network, reservoir class, block-residual innovation and BO ranges follow the
MultiNode_Epileptor three-node SCDT.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from network_gt import (
    BLANK_S, EVENT_THRESHOLD, N_NODES, OFFSETS, RC_HZ, BatchNetwork, departure, first_departure,
)

# Final (2026-09-29): 13 training conditions over the same range (step 0.005; was 5 at step 0.015).
# None coincides with a validation lam. See ../DESIGN_ADDENDUM_dense_training.md.
TRAIN_LAM = np.round(np.arange(-2.130, -2.0699, 0.005), 4)  # acquisition order
VALID_LAM = np.array([-2.1225, -2.1075, -2.0925, -2.0775])
CALIB_LAM = np.round(np.arange(-2.140, -2.0449, 0.0095), 4)
GT_LAM = np.round(np.arange(-2.070, -1.98999, 0.0025), 4)
PREP_LAM = -2.070

RECORD_S = 160.0
TRANSIENT_S = 600.0
WARMUP_S = 20.0
ROLLOUT_S = 200.0
VALID_ROLLOUT_S = 120.0

STAT_NAMES = ("mean", "sigma", "ac1", "ac8", "perm_entropy", "q05", "iqr", "abs_corr")
LEVEL_FAMILY = ("mean", "q05")


# --------------------------------------------------------------------------- utilities
def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def permutation_entropy(x: np.ndarray, order: int = 3) -> float:
    windows = np.lib.stride_tricks.sliding_window_view(x, order)
    ranks = np.argsort(windows, axis=1, kind="mergesort")
    codes = ranks @ (order ** np.arange(order)[::-1])
    counts = np.bincount(codes)
    prob = counts[counts > 0] / counts.sum()
    return float(-np.sum(prob * np.log(prob)) / math.log(math.factorial(order)))


def record_statistics(q: np.ndarray) -> np.ndarray:
    """Network-averaged candidate statistics of one record q[time, node]."""
    def ac(lag: int) -> float:
        return float(np.mean([np.corrcoef(q[:-lag, i], q[lag:, i])[0, 1] for i in range(q.shape[1])]))
    corr = np.corrcoef(q.T)
    iu = np.triu_indices(q.shape[1], 1)
    return np.array([
        q.mean(),
        q.std(axis=0).mean(),
        ac(1),
        ac(8),
        np.mean([permutation_entropy(q[::2, i]) for i in range(q.shape[1])]),  # 16 Hz
        np.quantile(q, 0.05, axis=0).mean(),
        (np.quantile(q, 0.75, axis=0) - np.quantile(q, 0.25, axis=0)).mean(),
        np.abs(corr[iu]).mean(),
    ])


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - np.sum(resid**2) / max(np.sum((y - y.mean()) ** 2), 1e-15)
    return float(slope), float(intercept), float(r2)


def crossing(x: np.ndarray, survival: np.ndarray, level: float = 0.5) -> float:
    """First downward crossing along the given order (Supp. Eq. 25)."""
    hits = np.flatnonzero(np.asarray(survival) <= level)
    if not len(hits):
        return float("nan")
    right = int(hits[0])
    if right == 0:
        return float(x[0])
    left = right - 1
    if survival[left] == survival[right]:
        return float(0.5 * (x[left] + x[right]))
    return float(x[left] + (level - survival[left]) * (x[right] - x[left]) / (survival[right] - survival[left]))


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def bootstrap_crossing(x: np.ndarray, k: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = [crossing(x, rng.binomial(n, k / n) / n) for _ in range(2000)]
    values = [v for v in values if np.isfinite(v)]
    if len(values) < 100:
        return float("nan"), float("nan")
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


# --------------------------------------------------------------------------- ground truth
def normal_records(lams: np.ndarray, replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-lam records after a 600 s target transient; departures are rejected and redrawn."""
    wanted = np.repeat(lams, replicates)
    kept: list[np.ndarray | None] = [None] * len(wanted)
    for attempt in range(6):
        todo = [i for i, r in enumerate(kept) if r is None]
        if not todo:
            break
        net = BatchNetwork(len(todo), seed + 7919 * attempt)
        state = net.initial_state()
        state, _ = net.run(state, wanted[todo], TRANSIENT_S, record=False)
        _, proxy = net.run(state, wanted[todo], RECORD_S)
        ok = ~departure(proxy, blank_s=0.0)
        for j, i in enumerate(todo):
            if ok[j]:
                kept[i] = proxy[:, j]
        print(f"  records seed={seed} attempt={attempt}: kept {int(ok.sum())}/{len(todo)}", flush=True)
    if any(r is None for r in kept):
        raise RuntimeError("could not obtain event-free records")
    return np.stack(kept, axis=1), wanted


def gt_boundary(ensemble: int, seed: int) -> list[dict]:
    """Prepare at PREP_LAM (600 s), switch to each target lam, 200 s horizon."""
    net = BatchNetwork(ensemble, seed)
    prepared = net.initial_state()
    prepared, _ = net.run(prepared, PREP_LAM, TRANSIENT_S, record=False)
    rows = []
    for index, lam in enumerate(GT_LAM):
        net.rng = np.random.default_rng(seed + 1000 + index)
        _, proxy = net.run(prepared.copy(), lam, ROLLOUT_S)
        times, nodes = first_departure(proxy)
        survivors = int(np.sum(~np.isfinite(times)))
        rows.append({
            "lam": float(lam), "survivors": survivors, "ensemble": ensemble,
            "survival_rate": survivors / ensemble,
            "median_onset_s": float(np.nanmedian(times)) if np.isfinite(times).any() else float("nan"),
            "first_node_counts": json.dumps(np.bincount(nodes[nodes >= 0], minlength=N_NODES).tolist()),
        })
        print(f"  GT lam={lam:.4f}: survival {survivors}/{ensemble}", flush=True)
    return rows


# --------------------------------------------------------------------------- reservoir
@dataclass(frozen=True)
class RCParams:
    n_reservoir: int = 1500
    spectral_radius: float = 0.72
    leakage_rate: float = 0.02
    input_scaling: float = 0.03
    static_scaling: float = 0.005
    ridge: float = 0.4
    density: float = 0.003
    innovation_scaling: float = 0.45
    block_s: float = 1.0
    delta_mode: bool = False
    seed: int = 733900


class StaticBlockRC:
    """Monolithic sparse ESN with joint readout and block residual innovations.

    Same construction as MultiOutputStaticRC / BlockInnovationRC of the three-node
    SCDT: readout on [phi(r), u, 1] with odd reservoir entries squared.
    """

    readout_stride = 4

    def __init__(self, params: RCParams, dynamic_dim: int = N_NODES, static_dim: int = 2):
        self.p = params
        rng = np.random.default_rng(params.seed)
        n = params.n_reservoir
        self.w_in = rng.uniform(-1, 1, (n, dynamic_dim)) * params.input_scaling
        self.w_static = rng.uniform(-1, 1, (n, static_dim)) * params.static_scaling
        nnz = max(1, int(params.density * n * n))
        matrix = sparse.coo_matrix(
            (rng.normal(0, 1, nnz), (rng.integers(0, n, nnz), rng.integers(0, n, nnz))), shape=(n, n)
        ).tocsr()
        radius = float(np.max(np.abs(eigs(matrix, k=1, which="LM", return_eigenvectors=False, v0=np.ones(n)))))
        self.a = matrix * (params.spectral_radius / max(radius, 1e-9))
        self.block_steps = max(1, int(round(params.block_s * RC_HZ)))

    def update(self, r: np.ndarray, u: np.ndarray, s: np.ndarray) -> np.ndarray:
        return (1 - self.p.leakage_rate) * r + self.p.leakage_rate * np.tanh(self.a @ r + self.w_in @ u + self.w_static @ s)

    @staticmethod
    def features(r: np.ndarray, u: np.ndarray) -> np.ndarray:
        phi = r.copy()
        phi[1::2] = phi[1::2] ** 2
        return np.vstack([phi, u, np.ones((1, r.shape[1]))])

    def fit(self, seqs: np.ndarray, statics: np.ndarray, washout: int) -> None:
        """seqs[time, record, node] (normalized); statics[record, 2]."""
        values = seqs.reshape(-1, seqs.shape[2])
        span = np.ptp(values, axis=0)
        margin = 1.0 if self.p.delta_mode else 2.5
        self.clip_low = values.min(axis=0) - margin * span
        self.clip_high = values.max(axis=0) + (1.0 if self.p.delta_mode else 1.2) * span
        s = statics.T
        r = np.zeros((self.p.n_reservoir, seqs.shape[1]))
        dim = self.p.n_reservoir + seqs.shape[2] + 1
        gram = np.zeros((dim, dim))
        cross = np.zeros((seqs.shape[2], dim))
        for t in range(len(seqs) - 1):
            r = self.update(r, seqs[t].T, s)
            if t >= washout and (t - washout) % self.readout_stride == 0:
                f = self.features(r, seqs[t].T)
                target = seqs[t + 1].T - (seqs[t].T if self.p.delta_mode else 0.0)
                gram += f @ f.T
                cross += target @ f.T
        self.w_out = np.linalg.solve(gram + self.p.ridge * np.eye(dim), cross.T).T
        # Full-rate teacher-forced residual sequences for block innovations.
        r = np.zeros((self.p.n_reservoir, seqs.shape[1]))
        residual = np.zeros((len(seqs) - 1 - washout, seqs.shape[1], seqs.shape[2]))
        for t in range(len(seqs) - 1):
            r = self.update(r, seqs[t].T, s)
            if t >= washout:
                pred = self.w_out @ self.features(r, seqs[t].T) + (seqs[t].T if self.p.delta_mode else 0.0)
                residual[t - washout] = seqs[t + 1] - pred.T
        lo, hi = np.quantile(residual, [0.0025, 0.9975], axis=(0, 1))
        self.residuals = np.clip(residual, lo, hi)
        self.teacher_rmse = float(np.sqrt(np.mean(residual**2)))

    def rollout(self, warm: np.ndarray, warm_static: np.ndarray, target_static: np.ndarray,
                n_steps: int, seed: int) -> tuple[np.ndarray, float]:
        """warm[time, batch, node]; statics[batch, 2]. Returns (prediction, clip fraction)."""
        batch = warm.shape[1]
        r = np.zeros((self.p.n_reservoir, batch))
        for t in range(len(warm)):
            r = self.update(r, warm[t].T, warm_static.T)
        u = warm[-1].T
        out = np.empty((n_steps, batch, warm.shape[2]))
        rng = np.random.default_rng(seed)
        clipped = 0
        n_res, n_rec = self.residuals.shape[:2]
        for t in range(n_steps):
            r = self.update(r, u, target_static.T)
            step = self.w_out @ self.features(r, u)
            u = step + (u if self.p.delta_mode else 0.0)
            offset = t % self.block_steps
            if offset == 0:
                sources = rng.integers(0, n_rec, batch)
                starts = rng.integers(0, max(n_res - self.block_steps, 1), batch)
            u = u + self.p.innovation_scaling * self.residuals[np.minimum(starts + offset, n_res - 1), sources].T
            low = u < self.clip_low[:, None]
            high = u > self.clip_high[:, None]
            clipped += int(low.sum() + high.sum())
            u = np.clip(u, self.clip_low[:, None], self.clip_high[:, None])
            out[t] = u.T
        return out, clipped / out.size


# --------------------------------------------------------------------------- BO
BOUNDS_NOTE = "sr .3-2.5, leak 10^[-2,-0.155], input 10^[-1.7,-0.1], static 10^[-2.5,-0.3], ridge 10^[-3,1.5], innovation 0-0.9, block 10^[-0.9,0.6] s, mode"


def params_from_unit(u: np.ndarray, n_reservoir: int, seed: int) -> RCParams:
    return RCParams(
        n_reservoir=n_reservoir,
        spectral_radius=0.30 + 2.20 * u[0],
        leakage_rate=10 ** (-2.0 + math.log10(70.0) * u[1]),
        input_scaling=10 ** (-1.7 + 1.6 * u[2]),
        static_scaling=10 ** (-2.5 + 2.2 * u[3]),
        ridge=10 ** (-3.0 + 4.5 * u[4]),
        innovation_scaling=0.90 * u[5],
        block_s=10 ** (-0.9 + 1.5 * u[6]),
        delta_mode=bool(u[7] >= 0.5),
        seed=seed,
    )


def unit_from_params(p: RCParams) -> np.ndarray:
    return np.clip(np.array([
        (p.spectral_radius - 0.30) / 2.20,
        (math.log10(p.leakage_rate) + 2.0) / math.log10(70.0),
        (math.log10(p.input_scaling) + 1.7) / 1.6,
        (math.log10(p.static_scaling) + 2.5) / 2.2,
        (math.log10(p.ridge) + 3.0) / 4.5,
        p.innovation_scaling / 0.90,
        (math.log10(p.block_s) + 0.9) / 1.5,
        0.75 if p.delta_mode else 0.25,
    ]), 0.0, 1.0)


# Starting points: the two MultiNode_Epileptor three-node winners (stable-only BO
# trial 25; robust block-peak trial 152) and two generic settings.
STARTERS = [
    RCParams(spectral_radius=0.7216, leakage_rate=0.02064, input_scaling=0.02314, static_scaling=0.004177,
             ridge=0.4731, innovation_scaling=0.4712, block_s=1.211),
    RCParams(spectral_radius=0.7242, leakage_rate=0.01, input_scaling=0.05411, static_scaling=0.01051,
             ridge=0.3970, innovation_scaling=0.4529, block_s=0.841),
    RCParams(spectral_radius=0.90, leakage_rate=0.05, input_scaling=0.10, static_scaling=0.05,
             ridge=0.1, innovation_scaling=0.30, block_s=0.5),
    RCParams(spectral_radius=1.20, leakage_rate=0.03, input_scaling=0.05, static_scaling=0.10,
             ridge=1.0, innovation_scaling=0.50, block_s=1.0, delta_mode=True),
]


def expected_improvement(x: np.ndarray, gp: GaussianProcessRegressor, best: float) -> np.ndarray:
    mu, sd = gp.predict(x, return_std=True)
    sd = np.maximum(sd, 1e-9)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


class Experiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out = args.out_dir.resolve()
        for sub in ("data", "results", "figures"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.warm = int(round(WARMUP_S * RC_HZ))

    # ---- data -------------------------------------------------------------
    def cached(self, name: str, builder):
        path = self.out / "data" / f"{name}.npz"
        if path.exists():
            data = np.load(path)
            return data["proxy"], data["lam"]
        proxy, lam = builder()
        np.savez_compressed(path, proxy=proxy, lam=lam)
        return proxy, lam

    def prepare(self) -> None:
        a = self.args
        print("1/7 normal records (training / validation / calibration)", flush=True)
        self.train, self.train_lam = self.cached("training", lambda: normal_records(TRAIN_LAM, a.train_reps, a.seed + 1000))
        self.valid, self.valid_lam = self.cached("validation", lambda: normal_records(VALID_LAM, a.valid_reps, a.seed + 2000))
        self.calib, self.calib_lam = self.cached("calibration", lambda: normal_records(CALIB_LAM, a.calib_reps, a.seed + 3000))
        self.train_regime = np.repeat(np.arange(len(TRAIN_LAM)), a.train_reps)  # acquisition index only

        print("2/7 statistics, leak-free selection and conditioning path", flush=True)
        self.train_all = np.array([record_statistics(self.train[:, j]) for j in range(self.train.shape[1])])
        self.valid_all = np.array([record_statistics(self.valid[:, j]) for j in range(self.valid.shape[1])])
        self.calib_all = np.array([record_statistics(self.calib[:, j]) for j in range(self.calib.shape[1])])
        f_ratio = {}
        for k, name in enumerate(STAT_NAMES):
            groups = [self.train_all[self.train_regime == g, k] for g in range(len(TRAIN_LAM))]
            within = np.mean([np.var(x, ddof=1) for x in groups])
            between = np.var([np.mean(x) for x in groups], ddof=1)
            f_ratio[name] = float(between / max(within, 1e-18))
        self.f_ratio = f_ratio
        lead = max(STAT_NAMES, key=lambda n: f_ratio[n])
        second = max((n for n in STAT_NAMES if n not in LEVEL_FAMILY and n != lead), key=lambda n: f_ratio[n])
        self.names = (lead, second)
        self.idx = [STAT_NAMES.index(lead), STAT_NAMES.index(second)]
        print(f"  F ratios: { {k: round(v, 2) for k, v in f_ratio.items()} }", flush=True)
        print(f"  selected statistics: leading={lead}, second={second}", flush=True)

        self.train_stats = self.train_all[:, self.idx]
        self.valid_stats = self.valid_all[:, self.idx]
        self.calib_stats = self.calib_all[:, self.idx]
        self.path = fit_line(self.train_stats[:, 0], self.train_stats[:, 1])
        # Drift direction from acquisition order (not lam values).
        order_slope = np.polyfit(self.train_regime, self.train_stats[:, 0], 1)[0]
        self.direction = float(np.sign(order_slope))
        # Validation-only alignments.
        self.lead_lam = fit_line(self.calib_lam, self.calib_stats[:, 0])
        self.second_lam = fit_line(self.calib_lam, self.calib_stats[:, 1])

        values = self.train.reshape(-1, N_NODES)
        self.dyn_center = values.mean(axis=0)
        self.dyn_scale = values.std(axis=0)
        self.stat_center = self.train_stats.mean(axis=0)
        self.stat_scale = self.train_stats.std(axis=0)
        self.train_n = (self.train - self.dyn_center) / self.dyn_scale
        self.train_stat_n = (self.train_stats - self.stat_center) / self.stat_scale
        self.stat_range = np.ptp(self.train_stats, axis=0)

    def norm_stat(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.stat_center) / self.stat_scale

    def warmups(self, lead_value: float, count: int) -> tuple[np.ndarray, np.ndarray]:
        """20 s windows from the training regime nearest in the leading statistic."""
        regime_lead = np.array([self.train_stats[self.train_regime == g, 0].mean() for g in range(len(TRAIN_LAM))])
        regime = int(np.argmin(np.abs(regime_lead - lead_value)))
        members = np.flatnonzero(self.train_regime == regime)
        length = len(self.train_n)
        ends = np.linspace(length // 2, length, max(1, math.ceil(count / len(members))) + 1)[1:].astype(int)
        windows, statics = [], []
        for j in range(count):
            member = members[j % len(members)]
            end = ends[(j // len(members)) % len(ends)]
            windows.append(self.train_n[end - self.warm:end, member])
            statics.append(self.train_stat_n[member])
        return np.stack(windows, axis=1), np.asarray(statics)

    def denorm(self, pred_n: np.ndarray) -> np.ndarray:
        return pred_n * self.dyn_scale + self.dyn_center

    def build(self, params: RCParams, zero_static: bool = False) -> StaticBlockRC:
        rc = StaticBlockRC(params)
        statics = np.zeros_like(self.train_stat_n) if zero_static else self.train_stat_n
        rc.fit(self.train_n, statics, washout=5 * RC_HZ)
        return rc

    # ---- normal-only selection -------------------------------------------
    def validation_score(self, rc: StaticBlockRC, seed: int) -> dict:
        reps = self.args.valid_rollouts
        steps = int(round(VALID_ROLLOUT_S * RC_HZ))
        supplied, generated, departures, clip = [], [], 0, []
        for j, raw in enumerate(self.valid_stats):
            warm, warm_static = self.warmups(raw[0], reps)
            target = np.repeat(self.norm_stat(raw)[None], reps, axis=0)
            pred_n, c = rc.rollout(warm, warm_static, target, steps, seed + j)
            pred = self.denorm(pred_n)
            departures += int(np.sum(departure(pred)))
            clip.append(c)
            gen = np.array([record_statistics(pred[:, b])[self.idx] for b in range(reps)])
            supplied.append(raw)
            generated.append(gen.mean(axis=0))
        supplied = np.asarray(supplied)
        generated = np.asarray(generated)
        err = np.abs(generated - supplied) / self.stat_range
        stat_mae = float(np.mean(np.clip(err, 0, 10)))
        slope = float(np.polyfit(supplied[:, 0], generated[:, 0], 1)[0]) if np.ptp(supplied[:, 0]) > 0 else 0.0
        response = float(min(abs(slope - 1.0), 3.0))
        false_rate = departures / (len(self.valid_stats) * reps)
        clip_frac = float(np.mean(clip))
        score = stat_mae + response + 8.0 * false_rate + 4.0 * clip_frac
        if not np.isfinite(score):
            score = 1e3
        return {"score": score, "stat_mae": stat_mae, "response_slope": slope, "false_departure_rate": false_rate,
                "clip_fraction": clip_frac, "supplied": supplied, "generated": generated,
                "teacher_rmse": rc.teacher_rmse}

    def optimize(self) -> RCParams:
        a = self.args
        path = self.out / "data" / "bo_normal_only.csv"
        rng = np.random.default_rng(a.seed + 7000)
        units, scores, rows = [], [], []
        print(f"3/7 normal-only BO ({a.bo_calls} calls)", flush=True)
        for it in range(a.bo_calls):
            t0 = time.time()
            if it < len(STARTERS):
                unit = unit_from_params(replace(STARTERS[it], n_reservoir=a.reservoir_nodes))
            elif it < a.bo_initial:
                unit = rng.random(8)
            else:
                gp = GaussianProcessRegressor(kernel=Matern(length_scale=np.ones(8), nu=2.5) + WhiteKernel(1e-4),
                                              normalize_y=True, random_state=a.seed, n_restarts_optimizer=2)
                gp.fit(np.asarray(units), np.log1p(np.asarray(scores)))
                cand = rng.random((3000, 8))
                unit = cand[int(np.argmax(expected_improvement(cand, gp, float(np.min(np.log1p(scores))))))]
            params = params_from_unit(unit, a.reservoir_nodes, a.reservoir_seed)
            try:
                result = self.validation_score(self.build(params), a.seed + 8000)
            except Exception as error:  # noqa: BLE001 - failed candidates are scored, not hidden
                print(f"  candidate failed: {error}", flush=True)
                result = {"score": 1e3, "stat_mae": float("nan"), "response_slope": float("nan"),
                          "false_departure_rate": float("nan"), "clip_fraction": float("nan"), "teacher_rmse": float("nan")}
            units.append(unit)
            scores.append(min(float(result["score"]), 1e3))
            rows.append({"iteration": it + 1, "score": scores[-1],
                         **{k: result[k] for k in ("stat_mae", "response_slope", "false_departure_rate", "clip_fraction", "teacher_rmse")},
                         **asdict(params), "seconds": round(time.time() - t0, 1)})
            write_csv(path, rows)
            print(f"  BO {it + 1:02d}: score={scores[-1]:.4f} stat={result['stat_mae']:.3f} "
                  f"slope={result['response_slope']:.2f} false={result['false_departure_rate']:.3f} "
                  f"clip={result['clip_fraction']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        self.bo_rows = rows
        best = int(np.argmin(scores))
        self.best_iteration = best + 1
        return params_from_unit(units[best], a.reservoir_nodes, a.reservoir_seed)

    # ---- locked test ------------------------------------------------------
    def sweep_grid(self) -> np.ndarray:
        lead = self.train_stats[:, 0]
        width = np.ptp(lead)
        start = lead.mean() - self.direction * 0.25 * width
        edge = lead.max() if self.direction > 0 else lead.min()
        stop = edge + self.direction * self.args.extrapolation * width
        return np.linspace(start, stop, self.args.sweep_points)

    def sweep(self, rc: StaticBlockRC, grid: np.ndarray, seed: int, zero_static: bool = False, keep=False):
        a = self.args
        steps = int(round(ROLLOUT_S * RC_HZ))
        rows, examples = [], {}
        for g, lead in enumerate(grid):
            raw = np.array([lead, self.path[0] * lead + self.path[1]])
            warm, warm_static = self.warmups(lead, a.rc_ensemble)
            target = np.repeat(self.norm_stat(raw)[None], a.rc_ensemble, axis=0)
            if zero_static:
                warm_static = np.zeros_like(warm_static)
                target = np.zeros_like(target)
            pred_n, clip = rc.rollout(warm, warm_static, target, steps, seed + g)
            pred = self.denorm(pred_n)
            times, nodes = first_departure(pred)
            survivors = int(np.sum(~np.isfinite(times)))
            lo, hi = wilson(survivors, a.rc_ensemble)
            normal = np.isfinite(times) == False  # noqa: E712
            gen = np.array([record_statistics(pred[:, b])[self.idx] for b in np.flatnonzero(normal)]) if normal.any() else np.full((1, 2), np.nan)
            rows.append({
                "lead_input": float(lead), "second_input": float(raw[1]), "survivors": survivors,
                "ensemble": a.rc_ensemble, "survival_rate": survivors / a.rc_ensemble,
                "wilson_low": lo, "wilson_high": hi, "clip_fraction": clip,
                "median_onset_s": float(np.nanmedian(times)) if np.isfinite(times).any() else float("nan"),
                "first_node_counts": json.dumps(np.bincount(nodes[nodes >= 0], minlength=N_NODES).tolist()),
                "generated_lead_mean_survivors": float(np.nanmean(gen[:, 0])),
                "generated_second_mean_survivors": float(np.nanmean(gen[:, 1])),
            })
            if keep:
                examples[float(lead)] = pred[:, :3]
            print(f"  sweep {g + 1:02d}/{len(grid)} lead={lead:.4f} survival={survivors}/{a.rc_ensemble} clip={clip:.4f}", flush=True)
        if keep:
            rates = np.array([r["survival_rate"] for r in rows])
            chosen = {0, len(grid) // 2, len(grid) - 1, int(np.argmin(np.abs(rates - 0.5)))}
            examples = {float(grid[i]): examples[float(grid[i])] for i in sorted(chosen)}
        return rows, examples

    def threshold(self, rows: list[dict], seed: int) -> dict:
        x = np.array([r["lead_input"] for r in rows])
        s = np.array([r["survival_rate"] for r in rows])
        k = np.array([r["survivors"] for r in rows])
        c = crossing(x, s)
        ci = bootstrap_crossing(x, k, self.args.rc_ensemble, seed) if np.isfinite(c) else (float("nan"), float("nan"))
        lam = (c - self.lead_lam[1]) / self.lead_lam[0] if np.isfinite(c) else float("nan")
        return {"lead_50": c, "lead_50_ci_low": ci[0], "lead_50_ci_high": ci[1], "aligned_lam_50_validation_only": lam,
                "second_50": self.path[0] * c + self.path[1] if np.isfinite(c) else float("nan")}

    def run(self) -> None:
        a = self.args
        self.prepare()
        best = self.optimize()
        print(f"  locked parameters (BO iteration {self.best_iteration}): {asdict(best)}", flush=True)
        (self.out / "results" / "locked_parameters.json").write_text(json.dumps(
            {"bo_iteration": self.best_iteration, "parameters": asdict(best),
             "statistics": self.names, "locked_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

        print("4/7 locked validation summary", flush=True)
        rc = self.build(best)
        self.validation = self.validation_score(rc, a.seed + 8000)

        print("5/7 locked statistical sweeps (reservoir seeds)", flush=True)
        grid = self.sweep_grid()
        seeds = [a.reservoir_seed + 104729 * k for k in range(a.reservoir_repeats)]
        sweeps, thresholds, examples = {}, {}, {}
        for k, rseed in enumerate(seeds):
            model = rc if k == 0 else self.build(replace(best, seed=rseed))
            rows, ex = self.sweep(model, grid, a.seed + 300000 + 1000 * k, keep=(k == 0))
            sweeps[rseed] = rows
            thresholds[rseed] = self.threshold(rows, a.seed + 600000 + k)
            if k == 0:
                examples = ex
            write_csv(self.out / "data" / f"scdt_sweep_seed{rseed}.csv", rows)
            print(f"  seed {rseed}: {thresholds[rseed]}", flush=True)

        print("6/7 negative control and GT boundary", flush=True)
        control_rows, _ = self.sweep(self.build(best, zero_static=True), grid, a.seed + 400000, zero_static=True)
        write_csv(self.out / "data" / "condition_independent_control.csv", control_rows)
        gt_path = self.out / "data" / "gt_boundary.csv"
        if gt_path.exists():
            with gt_path.open() as handle:
                gt_rows = [{k: float(v) if k not in ("first_node_counts",) else v for k, v in r.items()} for r in csv.DictReader(handle)]
        else:
            gt_rows = gt_boundary(a.gt_ensemble, a.seed + 500000)
            write_csv(gt_path, gt_rows)
        gt_lam = np.array([r["lam"] for r in gt_rows])
        gt_s = np.array([r["survival_rate"] for r in gt_rows])
        gt_lam50 = crossing(gt_lam, gt_s)
        gt_lead50 = self.lead_lam[0] * gt_lam50 + self.lead_lam[1]

        print("7/7 summary and figures", flush=True)
        primary = thresholds[seeds[0]]
        seed_values = np.array([t["lead_50"] for t in thresholds.values()])
        finite = seed_values[np.isfinite(seed_values)]
        summary = {
            "statistics": {"leading": self.names[0], "second": self.names[1], "f_ratio": self.f_ratio},
            "path": {"slope": self.path[0], "intercept": self.path[1], "r2": self.path[2], "drift_direction": self.direction},
            "alignment_validation_only": {"lead_vs_lam": self.lead_lam, "second_vs_lam": self.second_lam},
            "locked_parameters": asdict(best), "bo_iteration": self.best_iteration,
            "normal_validation": {k: self.validation[k] for k in ("score", "stat_mae", "response_slope", "false_departure_rate", "clip_fraction", "teacher_rmse")},
            "gt": {"lam_50": gt_lam50, "lead_50_mapped": gt_lead50},
            "scdt_primary": primary,
            "scdt_seeds": {str(k): v for k, v in thresholds.items()},
            "scdt_seed_summary": {"finite": int(len(finite)), "total": len(seed_values),
                                  "min": float(finite.min()) if len(finite) else float("nan"),
                                  "max": float(finite.max()) if len(finite) else float("nan"),
                                  "mean": float(finite.mean()) if len(finite) else float("nan"),
                                  "sd": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")},
            "control_lead_50": crossing(grid, np.array([r["survival_rate"] for r in control_rows])),
            "lead_error_primary": primary["lead_50"] - gt_lead50,
            "training_lead_range": [float(self.train_stats[:, 0].min()), float(self.train_stats[:, 0].max())],
        }
        protocol = {
            "experiment": "strict three-node SCDT",
            "separation_rules": {
                "reservoir_inputs": ["three node proxies x2-x1", f"measured {self.names[0]}", f"measured {self.names[1]}"],
                "lam_use": "record generation and validation-axis alignment only",
                "statistic_selection": "between/within-regime variance ratio on training records (regime membership only)",
                "drift_direction": "sign of leading statistic over acquisition order",
                "training": "event-free fixed-lam records only",
                "hyperparameter_selection": "normal validation records only (statistic reproduction, response slope, false departure, clipping)",
                "final_test": "sweep, control and GT boundary evaluated after locking",
                "warmup": "20 s windows of the nearest training regime; no GT test signal",
            },
            "network": {"nodes": N_NODES, "offsets": OFFSETS.tolist(), "coupling": 0.7, "connectivity": "triangle [1.0, 0.45, 0.70]"},
            "event_rule": f"any node proxy <= {EVENT_THRESHOLD} after {BLANK_S} s blanking",
            "lam": {"training": TRAIN_LAM.tolist(), "validation": VALID_LAM.tolist(), "calibration": CALIB_LAM.tolist(),
                    "gt_boundary": GT_LAM.tolist(), "gt_preparation": PREP_LAM},
            "durations_s": {"record": RECORD_S, "transient": TRANSIENT_S, "warmup": WARMUP_S, "rollout": ROLLOUT_S, "validation_rollout": VALID_ROLLOUT_S},
            "bo_bounds": BOUNDS_NOTE,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        }
        (self.out / "results" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        (self.out / "results" / "protocol.json").write_text(json.dumps(protocol, indent=2, default=float))
        write_csv(self.out / "data" / "training_statistics.csv",
                  [{"record": j, "regime": int(self.train_regime[j]), "lam_validation_only": float(self.train_lam[j]),
                    **{n: float(v) for n, v in zip(STAT_NAMES, self.train_all[j])}} for j in range(len(self.train_all))])
        write_csv(self.out / "data" / "calibration_statistics.csv",
                  [{"lam": float(self.calib_lam[j]), **{n: float(v) for n, v in zip(STAT_NAMES, self.calib_all[j])}} for j in range(len(self.calib_all))])
        self.figures(grid, sweeps, seeds, thresholds, control_rows, gt_rows, gt_lam50, gt_lead50, examples)
        print(json.dumps({"gt": summary["gt"], "scdt_primary": primary, "seeds": summary["scdt_seed_summary"]}, indent=2, default=float), flush=True)

    # ---- figures ------------------------------------------------------------
    def figures(self, grid, sweeps, seeds, thresholds, control_rows, gt_rows, gt_lam50, gt_lead50, examples) -> None:
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.alpha": 0.2, "figure.facecolor": "white"})
        fig_dir = self.out / "figures"
        lead, second = self.names

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        sc = ax.scatter(self.train_stats[:, 0], self.train_stats[:, 1], c=self.train_regime, cmap="Blues", vmin=-1.5,
                        s=55, edgecolor="k", lw=0.4, label="training records (colour = regime)", zorder=3)
        ax.scatter(self.valid_stats[:, 0], self.valid_stats[:, 1], marker="D", s=35, color="#1b7837", label="normal validation")
        ax.plot(grid, self.path[0] * grid + self.path[1], "--", color="0.4", label=f"conditioning path ($R^2$={self.path[2]:.3f})")
        if np.isfinite(gt_lead50):
            ax.axvline(gt_lead50, color="k", lw=1, ls=":", label="GT 50% (mapped, validation only)")
        ax.set(xlabel=f"leading statistic: {lead}", ylabel=f"second statistic: {second}", title="Three-node conditioning path")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / "01_conditioning_path.png", dpi=220); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, k, name, fit in ((axes[0], 0, lead, self.lead_lam), (axes[1], 1, second, self.second_lam)):
            ax.scatter(self.calib_lam, self.calib_stats[:, k], s=18, color="#7b3294", alpha=0.7, label="calibration records")
            xx = np.linspace(self.calib_lam.min(), GT_LAM.max(), 100)
            ax.plot(xx, fit[0] * xx + fit[1], color="#c51b7d", label=f"fit $R^2$={fit[2]:.3f}")
            if np.isfinite(gt_lam50):
                ax.axvline(gt_lam50, color="k", ls=":", label="GT 50% λ")
            ax.set(xlabel="hidden λ (validation only)", ylabel=name)
            ax.legend(fontsize=8)
        fig.suptitle("Validation-axis alignment (never used by the SCDT)")
        fig.tight_layout(); fig.savefig(fig_dir / "02_parameter_statistic_alignment.png", dpi=220); plt.close(fig)

        v = self.validation
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, k, name in ((axes[0], 0, lead), (axes[1], 1, second)):
            lo = min(v["supplied"][:, k].min(), v["generated"][:, k].min())
            hi = max(v["supplied"][:, k].max(), v["generated"][:, k].max())
            ax.scatter(v["supplied"][:, k], v["generated"][:, k], s=45, color="#1b7837")
            ax.plot([lo, hi], [lo, hi], "--", color="0.4")
            ax.set(xlabel=f"supplied {name}", ylabel=f"reproduced {name}")
        fig.suptitle(f"Normal validation (locked): scaled MAE={v['stat_mae']:.3f}, slope={v['response_slope']:.2f}, false departure={v['false_departure_rate']:.3f}")
        fig.tight_layout(); fig.savefig(fig_dir / "03_normal_validation.png", dpi=220); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5.6))
        primary = sweeps[seeds[0]]
        x = np.array([r["lead_input"] for r in primary])
        ax.fill_between(x, [r["wilson_low"] for r in primary], [r["wilson_high"] for r in primary], color="#2166ac", alpha=0.15)
        ax.plot(x, [r["survival_rate"] for r in primary], "o-", color="#2166ac", ms=4, label=f"SCDT (seed {seeds[0]})")
        for rseed in seeds[1:]:
            ax.plot(x, [r["survival_rate"] for r in sweeps[rseed]], "-", color="#2166ac", alpha=0.35, lw=1)
        ax.plot(x, [r["survival_rate"] for r in control_rows], "--", color="#e08214", label="static-zeroed control")
        gt_x = self.lead_lam[0] * np.array([r["lam"] for r in gt_rows]) + self.lead_lam[1]
        ax.plot(gt_x, [r["survival_rate"] for r in gt_rows], "s-", color="k", ms=3, label="GT (λ mapped, validation only)")
        ax.axhline(0.5, color="0.5", ls=":")
        t = thresholds[seeds[0]]
        if np.isfinite(t["lead_50"]):
            ax.axvline(t["lead_50"], color="#2166ac", ls="--", label=f"SCDT 50%: {t['lead_50']:.4f}")
        if np.isfinite(gt_lead50):
            ax.axvline(gt_lead50, color="k", ls="--", label=f"GT 50%: {gt_lead50:.4f}")
        ax.axvspan(self.train_stats[:, 0].min(), self.train_stats[:, 0].max(), color="0.5", alpha=0.08, label="training range")
        ax.set(xlabel=f"leading statistic ({lead})", ylabel="normal-survival rate (200 s)", ylim=(-0.04, 1.04),
               title="Strict three-node SCDT: survival along the conditioning path")
        if self.direction < 0:
            ax.invert_xaxis()
        top = ax.secondary_xaxis("top", functions=(lambda s: (s - self.lead_lam[1]) / self.lead_lam[0], lambda l: self.lead_lam[0] * l + self.lead_lam[1]))
        top.set_xlabel("aligned λ (validation only)")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout(); fig.savefig(fig_dir / "04_survival_threshold.png", dpi=220); plt.close(fig)

        if examples:
            fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.4 * len(examples)), sharex=True, sharey=True)
            axes = np.atleast_1d(axes)
            for ax, (value, sig) in zip(axes, sorted(examples.items(), reverse=self.direction < 0)):
                tt = np.arange(len(sig)) / RC_HZ
                for node in range(N_NODES):
                    ax.plot(tt, sig[:, 0, node], lw=0.6, label=f"node {node}")
                ax.axhline(EVENT_THRESHOLD, color="#b2182b", ls="--", lw=0.8)
                ax.set_title(f"SCDT rollout, {lead}={value:.4f}", loc="left", fontsize=9)
            axes[0].legend(fontsize=7, ncol=3)
            axes[-1].set_xlabel("autonomous time (s)")
            fig.tight_layout(); fig.savefig(fig_dir / "05_rollout_examples.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        gen = np.array([r["generated_lead_mean_survivors"] for r in primary])
        ax.plot(x, gen, "o-", ms=3, label="reproduced (surviving rollouts)")
        ax.plot(x, x, "--", color="0.4", label="y = x")
        ax.axvspan(self.train_stats[:, 0].min(), self.train_stats[:, 0].max(), color="0.5", alpha=0.08)
        ax.set(xlabel=f"supplied {lead}", ylabel=f"reproduced {lead}", title="Supplied vs reproduced along the sweep")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / "06_sweep_self_consistency.png", dpi=220); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--train-reps", type=int, default=4)
    p.add_argument("--valid-reps", type=int, default=2)
    p.add_argument("--calib-reps", type=int, default=4)
    p.add_argument("--valid-rollouts", type=int, default=4)
    p.add_argument("--gt-ensemble", type=int, default=30)
    p.add_argument("--rc-ensemble", type=int, default=30)
    p.add_argument("--reservoir-nodes", type=int, default=1500)
    p.add_argument("--reservoir-repeats", type=int, default=1)
    p.add_argument("--bo-calls", type=int, default=40)
    p.add_argument("--bo-initial", type=int, default=12)
    p.add_argument("--sweep-points", type=int, default=36)
    p.add_argument("--extrapolation", type=float, default=2.0, help="sweep beyond training edge, in training-range widths")
    p.add_argument("--reservoir-seed", type=int, default=733900)
    p.add_argument("--seed", type=int, default=20260924)
    Experiment(p.parse_args()).run()


if __name__ == "__main__":
    main()
