"""T1 + T2 of docs/05 §3: does the time structure of the injected residuals matter?

T1 (`acf`): autocorrelation of the teacher-forced one-step residuals that the
    rollout re-injects, next to the signal's own autocorrelation, plus the
    lag-0 cross-node correlation for networks.
T2 (`rollout`): the trained readout is kept; only the innovation mode of the
    rollout changes. Members 0..M-1 of the seed ensemble, same GT trials, same
    innovation seed as the ensemble member, so the current mode reproduces the
    stored ensemble member exactly (gate).
      multi-node: block (current) | iid_fullrate | none
         block and iid_fullrate draw from the same residual pool and keep the
         same-time cross-node vector, so they differ only in time correlation.
      single node: iid (current) | block (ported, same block length rule) | none
T2 analysis (`analyze`): per mode, each member scored like the cross table
    (Brier, IBS, original MAE, lambda_c, T50 error); paired member-wise
    differences mode - current with a level-cluster bootstrap.

Run (repo root):
  .venv/bin/python survival_dist/innovation_ablation.py acf --system single_node
  .venv/bin/python survival_dist/innovation_ablation.py acf --system three_node
  .venv/bin/python survival_dist/innovation_ablation.py rollout --system three_node --members 5
  .venv/bin/python survival_dist/innovation_ablation.py rollout --system single_node --members 5
  .venv/bin/python survival_dist/innovation_ablation.py analyze
Output: outputs/innovation_ablation/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "outputs" / "innovation_ablation"
RUN_DIRS = {"two_node": HERE / "outputs" / "two_node" / "run", "three_node": HERE / "outputs" / "three_node" / "run"}
MULTI_MODES = ("block", "iid_fullrate", "none")
SINGLE_MODES = ("iid", "block", "none")
CURRENT = {"single_node": "iid", "two_node": "block", "three_node": "block"}
MAX_LAG_S = 5.0
RC_HZ = 32.0


def acf(v: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation along axis 0, averaged over the remaining axes' series."""
    v = v - v.mean(axis=0, keepdims=True)
    den = np.sum(v * v, axis=0)
    return np.array([np.mean(np.sum(v[:-k] * v[k:], axis=0) / den) for k in range(1, max_lag + 1)])


# ---- T1 ------------------------------------------------------------------------------------

def run_acf(system: str) -> None:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    L = int(MAX_LAG_S * RC_HZ)
    if system == "single_node":
        from single_node_setup import load_setup
        s = load_setup()
        rc = s.fit_member()
        res_acf = {"node 1": acf(rc.residuals[:, None], L)}
        sig_acf = {"node 1": np.mean([acf(q[:, None], L) for q in s.train_sequences], axis=0)}
        cross = None
        scale = float(np.std(rc.residuals) / np.std(np.concatenate(s.train_sequences)))
    else:
        import run_multinode_ensemble as rme
        rme._init(str(RUN_DIRS[system]), "rate", "test")
        W = rme._W
        train_n, train_raw, train_static, stable_count = W["training"]
        model, _ = W["exp"].fit_block_model(W["params"], W["block_s"], train_n, train_raw, train_static, stable_count)
        r = np.asarray(model.residual_sequences[:, :stable_count, :], dtype=float)  # (T, sources, nodes)
        sig = np.asarray(train_n[:, :stable_count, :], dtype=float)
        n_nodes = r.shape[2]
        res_acf = {f"node {i + 1}": acf(r[:, :, i], L) for i in range(n_nodes)}
        sig_acf = {f"node {i + 1}": acf(sig[:, :, i], L) for i in range(n_nodes)}
        flat = r.reshape(-1, n_nodes)
        cross = np.corrcoef(flat.T).tolist()
        scale = float(np.std(r) / np.std(sig))
    lags = np.arange(1, L + 1)
    pick = [1, 2, 4, 8, 16, 32, 64, 128]
    summary = {
        "system": system, "residual_std_over_signal_std": scale, "lag_steps": pick,
        "residual_acf": {k: [float(v[p - 1]) for p in pick] for k, v in res_acf.items()},
        "signal_acf": {k: [float(v[p - 1]) for p in pick] for k, v in sig_acf.items()},
        "residual_max_abs_acf_lag_ge_1": float(max(np.max(np.abs(v)) for v in res_acf.values())),
        "residual_cross_node_corr_lag0": cross,
    }
    n_eff = (len(rc.residuals) if system == "single_node" else r.shape[0] * r.shape[1])
    summary["white_noise_95_band_approx"] = float(1.96 / np.sqrt(n_eff))
    out = OUT / "acf"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{system}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / f"{system}.npz", lags=lags, **{f"res_{k}": v for k, v in res_acf.items()}, **{f"sig_{k}": v for k, v in sig_acf.items()})
    print(json.dumps({k: summary[k] for k in ("system", "residual_std_over_signal_std", "residual_acf", "signal_acf", "residual_cross_node_corr_lag0")}, indent=1))


# ---- T2: multi-node --------------------------------------------------------------------------

def _multi_job(args: tuple[str, str, int]) -> str:
    import run_multinode_ensemble as rme
    system, mode, j = args
    W = rme._W
    out = OUT / system / mode / f"member_{j:04d}.npz"
    if out.exists():
        return f"{system} {mode} member {j}: cached"
    exp = W["exp"]
    params = replace(W["params"], seed=int(W["params"].seed) + 10007 * j)
    train_n, train_raw, train_static, stable_count = W["training"]
    model, _ = exp.fit_block_model(params, W["block_s"], train_n, train_raw, train_static, stable_count)
    model.innovation_mode = mode  # fit_block_model sets "block"; the readout is unchanged
    metrics, prediction, rc_events = exp.evaluate_cohort(model, W["test"], W["coordinate"], W["normalizers"], W["innovation_base"] + 1000 * j)
    scalars = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))}
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, events=np.asarray(rc_events, float), nonfinite=~np.isfinite(prediction).all(axis=(0, 2)),
                        metrics_json=np.array(json.dumps(scalars)))
    os.replace(tmp, out)
    return f"{system} {mode} member {j}: done"


def _multi_init(system: str) -> None:
    import run_multinode_ensemble as rme
    rme._init(str(RUN_DIRS[system]), "rate", "fine")


def run_multi(system: str, members: int, workers: int) -> None:
    for mode in MULTI_MODES:
        (OUT / system / mode).mkdir(parents=True, exist_ok=True)
    jobs = [(system, m, j) for j in range(members) for m in MULTI_MODES]
    with ProcessPoolExecutor(max_workers=workers, initializer=_multi_init, initargs=(system,)) as pool:
        for msg in pool.map(_multi_job, jobs):
            print(msg, flush=True)


# ---- T2: single node ---------------------------------------------------------------------------

def _block_rollout(self, warmup, warmup_statics, target_statics, n_steps, seed=0, ramp_steps=0):
    """SparseStaticRC.rollout_switch_batch with block-resampled residuals (copy of the
    original loop; only the innovation draw differs). Residuals are the fit's
    one-step residuals in time order; a block may straddle two training sequences."""
    batch = warmup.shape[1]
    reservoir = np.zeros((self.params.n_reservoir, batch))
    ws, ts = warmup_statics.T, target_statics.T
    for index in range(len(warmup)):
        reservoir = self.update(reservoir, warmup[index][None, :], ws)
    current = warmup[-1][None, :]
    prediction = np.empty((n_steps, batch), dtype=float)
    rng = np.random.default_rng(seed)
    L = self.block_steps
    starts = np.zeros(batch, dtype=int)
    for index in range(n_steps):
        static = ((1.0 - (index + 1) / ramp_steps) * ws + (index + 1) / ramp_steps * ts) if (ramp_steps > 0 and index < ramp_steps) else ts
        reservoir = self.update(reservoir, current, static)
        current = self.w_out @ self.augment(reservoir, current)
        if index % L == 0:
            starts = rng.integers(0, len(self.residuals) - L + 1, size=batch)
        current = current + self.params.innovation_scaling * self.residuals[starts + index % L][None, :]
        current = np.clip(current, self.clip_low, self.clip_high)
        prediction[index] = current[0]
    return prediction


_S: dict = {}


def _single_init() -> None:
    from ensemble import load_or_build_cohort
    from run_phase2_single_node import OUT as P2, cohort_specs
    from single_node_setup import load_setup
    setup = load_setup()
    specs, _ = cohort_specs(setup)
    _S.update(setup=setup, specs=specs, cohorts={s.name: load_or_build_cohort(setup, s, P2 / "cohorts") for s in specs})


def _single_job(args: tuple[str, int, int]) -> str:
    from ensemble import _rollout_record, member_seed, rollout_seed
    mode, j, block_steps = args
    out = OUT / "single_node" / mode / f"member_{j:04d}.npz"
    if out.exists():
        return f"single {mode} member {j}: cached"
    setup = _S["setup"]
    rc = setup.fit_member(member_seed(setup, j))
    if mode == "block":
        rc.block_steps = block_steps
        rc.rollout_switch_batch = types.MethodType(_block_rollout, rc)
    elif mode == "none":
        rc.params = replace(rc.params, innovation_scaling=0.0)
    payload = {}
    for spec in _S["specs"]:
        result = setup.evaluate(rc, _S["cohorts"][spec.name], rollout_seed(setup, j, spec))
        payload.update(_rollout_record(setup, rc, result, spec.name))
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, out)
    return f"single {mode} member {j}: done"


def run_single(members: int, workers: int, block_s: float) -> None:
    for mode in SINGLE_MODES:
        (OUT / "single_node" / mode).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    steps = max(1, int(round(block_s * RC_HZ)))
    jobs = [(m, j, steps) for j in range(members) for m in SINGLE_MODES]
    with ProcessPoolExecutor(max_workers=workers, initializer=_single_init) as pool:
        for msg in pool.map(_single_job, jobs):
            print(msg, flush=True)


# ---- T2 analysis -------------------------------------------------------------------------------

def _load_mode_events(system: str, mode: str, members: int) -> np.ndarray | None:
    folder = OUT / system / mode
    files = [folder / f"member_{j:04d}.npz" for j in range(members)]
    if not all(f.exists() for f in files):
        return None
    if system == "single_node":
        return np.vstack([np.r_[np.load(f)["coarse_events"], np.load(f)["fine_events"]] for f in files])
    return np.vstack([np.load(f)["events"] for f in files])


def analyze(members: int) -> None:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import dist_metrics as dm
    from critical_values import empirical_x50, survival_curve
    from cross_table import level_survival_curves, multi_node_inputs, single_node_inputs
    from critical_values import pool_single_node
    from plotting import SERIES, TEXT, TEXT_SECONDARY, apply_style, plt

    rng = np.random.default_rng(2028)
    systems = {"single_node": (pool_single_node(single_node_inputs()), SINGLE_MODES)}
    for label, name in (("two_node", "2-node"), ("three_node", "3-node")):
        inp = multi_node_inputs(label, name, suffix="_fine")
        if inp is not None:
            systems[label] = (inp, MULTI_MODES)
    report, md = {}, ["# T2: innovation-mode ablation (generated by `innovation_ablation.py analyze`; read with docs/05 §3)", ""]
    for system, (inp, modes) in systems.items():
        ev = {m: _load_mode_events(system, m, members) for m in modes}
        if any(v is None for v in ev.values()):
            print(f"{system}: rollouts missing, skipped", flush=True)
            continue
        x, gt, H = inp["x"], inp["gt"], inp["horizon"]
        grid = np.unique(x)
        gi = np.searchsorted(grid, x)
        cur = CURRENT[system]
        stored = inp["members"][:members]
        gate = bool(np.allclose(ev[cur], stored, equal_nan=True))
        o = (np.isfinite(gt) & (gt <= H)).astype(float)
        o_freq = np.array([o[x == v].mean() for v in grid])
        times = np.arange(max(1.0, inp["blanking_s"]), H - 1.0 + 1e-9, 0.5)
        alive = (np.where(np.isfinite(gt), gt, np.inf)[:, None] > times).astype(float)
        gt_t50 = _t50_levels(x, gt, grid, times)

        per = {}
        for m in modes:
            rows = []
            for j in range(members):
                e = ev[m][j]
                p_level = 1.0 - survival_curve(x, e, H, grid)
                curves = level_survival_curves(x, e, grid, times)
                t50 = _t50_levels(x, e, grid, times)
                both = np.isfinite(t50) & np.isfinite(gt_t50)
                rows.append({
                    "se": (p_level[gi] - o) ** 2,
                    "ibs_trial": np.trapezoid((alive - curves[gi]) ** 2, times, axis=1) / (times[-1] - times[0]),
                    "ae_level": np.abs(p_level - o_freq),
                    "lambda_c_H": empirical_x50(grid, survival_curve(x, e, H, grid))[0],
                    "lambda_c_60": empirical_x50(grid, survival_curve(x, e, 60.0, grid))[0],
                    "t50_abs_err": float(np.median(np.abs(t50 - gt_t50)[both])) if both.any() else float("nan"),
                    "departures": int(np.isfinite(e).sum()),
                })
            per[m] = rows
        gt_lc_H = empirical_x50(grid, survival_curve(x, gt, H, grid))[0]
        gt_lc_60 = empirical_x50(grid, survival_curve(x, gt, 60.0, grid))[0]
        res = {"system": system, "cohort": inp["cohort"], "members": members, "gate_current_mode_equals_stored_members": gate,
               "gt_lambda_c": {"H": gt_lc_H, "60": gt_lc_60}, "modes": {}, "paired_vs_current": {}}
        idx_by_level = [np.flatnonzero(x == v) for v in grid]
        boot = rng.integers(0, len(grid), size=(2000, len(grid)))
        for m in modes:
            r = per[m]
            res["modes"][m] = {
                "brier_H_members": [float(q["se"].mean()) for q in r],
                "ibs_members": [float(q["ibs_trial"].mean()) for q in r],
                "mae_original_members": [float(q["ae_level"].mean()) for q in r],
                "lambda_c_H_members": [q["lambda_c_H"] for q in r],
                "lambda_c_60_members": [q["lambda_c_60"] for q in r],
                "t50_abs_err_members_s": [q["t50_abs_err"] for q in r],
                "departures_members": [q["departures"] for q in r],
            }
            if m == cur:
                continue
            d = {}
            for key, field in (("brier_H", "se"), ("ibs", "ibs_trial")):
                diff = np.mean([per[m][j][field] - per[cur][j][field] for j in range(members)], axis=0)  # per trial
                bs = np.array([np.concatenate([diff[idx_by_level[g]] for g in lv]).mean() for lv in boot])
                d[key] = {"mean": float(diff.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
            diff = np.mean([per[m][j]["ae_level"] - per[cur][j]["ae_level"] for j in range(members)], axis=0)
            bs = diff[boot].mean(axis=1)
            d["mae_original"] = {"mean": float(diff.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
            for key in ("lambda_c_H", "lambda_c_60", "t50_abs_err"):
                dv = np.array([per[m][j][key] - per[cur][j][key] for j in range(members)], dtype=float)
                d[key] = {"member_diffs": dv.tolist(), "median": float(np.nanmedian(dv)) if np.isfinite(dv).any() else float("nan")}
            res["paired_vs_current"][f"{m} - {cur}"] = d
        report[system] = res
        md += _md(res, modes, cur)
        _plot(system, inp, ev, modes, grid, x, gt, H, res, apply_style, plt, SERIES, TEXT, TEXT_SECONDARY)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t2_summary.json").write_text(json.dumps(report, indent=2, default=float))
    (OUT / "t2_table.md").write_text("\n".join(md))
    print("\n".join(md))


def _t50_levels(x, events, grid, times):
    out = []
    ev = np.where(np.isfinite(events), events, np.inf)
    for v in grid:
        s = (ev[x == v][:, None] > times).mean(axis=0)
        k = np.flatnonzero(s <= 0.5)
        out.append(times[k[0]] if k.size else np.nan)
    return np.array(out)


def _fmt(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _md(res, modes, cur):
    lines = [f"## {res['system']} — {res['cohort']} (members 0–{res['members'] - 1}; gate current == stored: {res['gate_current_mode_equals_stored_members']})", "",
             f"GT λc: H {_fmt(res['gt_lambda_c']['H'], 4)}, 60 s {_fmt(res['gt_lambda_c']['60'], 4)}", "",
             "| mode | Brier(H) mean [min–max] | IBS mean | MAE mean | λc(H) median [min–max] | λc(60) median | T50 |err| median (s) |",
             "|---|---|---|---|---|---|---|"]
    for m in modes:
        r = res["modes"][m]
        b, lc = np.array(r["brier_H_members"]), np.array(r["lambda_c_H_members"], dtype=float)
        lines.append(f"| {m}{' (current)' if m == cur else ''} | {b.mean():.4f} [{b.min():.4f}–{b.max():.4f}] | {np.mean(r['ibs_members']):.4f} | "
                     f"{np.mean(r['mae_original_members']):.4f} | {_fmt(np.nanmedian(lc), 4)} [{_fmt(np.nanmin(lc), 4)}–{_fmt(np.nanmax(lc), 4)}] | "
                     f"{_fmt(np.nanmedian(np.array(r['lambda_c_60_members'], dtype=float)), 4)} | {_fmt(np.nanmedian(np.array(r['t50_abs_err_members_s'], dtype=float)), 1)} |")
    lines += ["", "| paired (same reservoir, same trials) | ΔBrier(H) [95% CI] | ΔIBS [95% CI] | ΔMAE [95% CI] | Δλc(H) median | Δ T50 err median (s) |", "|---|---|---|---|---|---|"]
    for k, d in res["paired_vs_current"].items():
        lines.append(f"| {k} | {d['brier_H']['mean']:+.4f} [{d['brier_H']['ci95'][0]:+.4f}, {d['brier_H']['ci95'][1]:+.4f}] | "
                     f"{d['ibs']['mean']:+.4f} [{d['ibs']['ci95'][0]:+.4f}, {d['ibs']['ci95'][1]:+.4f}] | "
                     f"{d['mae_original']['mean']:+.4f} [{d['mae_original']['ci95'][0]:+.4f}, {d['mae_original']['ci95'][1]:+.4f}] | "
                     f"{_fmt(d['lambda_c_H']['median'], 4)} | {_fmt(d['t50_abs_err']['median'], 1)} |")
    return lines + [""]


def _plot(system, inp, ev, modes, grid, x, gt, H, res, apply_style, plt, SERIES, TEXT, TEXT_SECONDARY):
    """Survival at H and T50 per mode (member mean), GT for reference. Modes are told apart
    by line style + marker (one hue family), so identity is not color-alone."""
    from critical_values import survival_curve
    apply_style()
    styles = {modes[0]: ("-", "s"), modes[1]: ("--", "^"), modes[2]: (":", "v")}
    rc_c, gt_c = SERIES["rc"]["color"], SERIES["gt"]["color"]
    times = np.arange(max(1.0, inp["blanking_s"]), H - 1.0 + 1e-9, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for m in modes:
        ls, mk = styles[m]
        s = np.mean([survival_curve(x, e, H, grid) for e in ev[m]], axis=0)
        axes[0].plot(grid, s, color=rc_c, ls=ls, marker=mk, ms=3.5, lw=1.6, label=f"RC, {m}")
        t50 = np.nanmedian(np.vstack([_t50_levels(x, e, grid, times) for e in ev[m]]), axis=0)
        axes[1].plot(grid, t50, color=rc_c, ls=ls, marker=mk, ms=3.5, lw=1.6, label=f"RC, {m}")
    axes[0].plot(grid, survival_curve(x, gt, H, grid), color=gt_c, marker="o", ms=3.5, label="GT")
    axes[1].plot(grid, _t50_levels(x, gt, grid, times), color=gt_c, marker="o", ms=3.5, label="GT")
    axes[0].set(xlabel=inp["x_label"], ylabel=f"survival to {H:.0f} s (mean over members)", ylim=(-0.05, 1.05))
    axes[1].set(xlabel=inp["x_label"], ylabel="T50 (s), median over members", ylim=(0, H))
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"{system}: innovation mode only changed (same readout, same trials, members 0–{len(ev[modes[0]]) - 1})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"t2_{system}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("acf", "rollout", "analyze"))
    ap.add_argument("--system", choices=("single_node", "two_node", "three_node"))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--single-block-s", type=float, default=2.454, help="block length for the single-node port (3-node rate model's)")
    a = ap.parse_args()
    if a.command == "acf":
        run_acf(a.system)
    elif a.command == "rollout":
        if a.system == "single_node":
            run_single(a.members, a.workers, a.single_block_s)
        else:
            run_multi(a.system, a.members, a.workers)
    else:
        analyze(a.members)


if __name__ == "__main__":
    main()
