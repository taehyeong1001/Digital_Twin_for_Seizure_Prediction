"""Fair-comparison cross table: predictors x metrics on identical GT trials (docs/04).

Arms (all paired with the same GT trials, same event definition, same horizon):
  A  member 0, 1 rollout per trial            (the original single RC)
  B  member-0 reservoir, K innovation rollouts (same rollout count as E)
  C  each member alone, 1 rollout             (reported as a distribution over members)
  E  ensemble, J members x 1 rollout
  F  ensemble without member 0 (J - 1 members; removes the BO home advantage)
  oracle       independent-seed GT frequency (attainable floor)
  climatology  overall event rate (Brier reference)

Every arm's forecast for a trial is its departure frequency at that trial's
x level, pooled over its rollouts and over the level's trials, which is what
the original `grouped_survival` and our phase-2 metrics do. Arms differ only in
which rollouts are pooled, so E vs B isolates reservoir diversity at equal
rollout count.

Metrics per horizon h: original survival MAE (levels weighted equally), trial
Brier / BSS / Murphy REL & RES, x50(h); over time: IBS and D-calibration.
Paired differences use a cluster bootstrap over x levels; the forecasts are
held fixed, so the interval reflects which levels/trials were sampled, not the
rollout noise inside each forecast.

Inputs:
  single node: outputs/single_node/phase2/{cohorts, member_*.npz, armB/}
  multi node : outputs/<label>/cross_inputs_<model>.npz (written by
               run_multinode_ensemble.py --arm-b K) + its member / arm-B dirs
Run:   .venv/bin/python survival_dist/cross_table.py
Output: outputs/cross_table/{cross_table.json, cross_table.csv, cross_table.md}
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

import dist_metrics as dm
from ensemble import load_arm_b, load_members

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "cross_table"
N_BOOT = 2000
PAIRS = (("E", "B"), ("E", "A"), ("B", "A"), ("F", "A"), ("F", "B"), ("E", "F"))


# ---- inputs -------------------------------------------------------------------

def _cohort_npz(cache: Path, name: str) -> dict:
    paths = sorted(p for p in cache.glob(f"cohort_{name}_*.npz") if re.fullmatch(rf"cohort_{name}_[0-9a-f]{{12}}\.npz", p.name))
    if len(paths) != 1:
        raise RuntimeError(f"expected one cached cohort '{name}' in {cache}, found {len(paths)}")
    with np.load(paths[0]) as d:
        return {"x": d["x0"], "gt": d["gt_events"]}


def single_node_inputs(n_members: int = 20, n_rollouts: int = 20) -> list[dict]:
    base = HERE / "outputs" / "single_node" / "phase2"
    out = []
    for name in ("coarse", "fine"):
        gt = _cohort_npz(base / "cohorts", name)
        oracle = _cohort_npz(base / "cohorts", f"{name}_oracle")
        members = load_members(base, n_members, name)
        arm_b = load_arm_b(base / "armB", n_rollouts, name)
        out.append({
            "system": "single node", "cohort": name, "x_label": "x0",
            "x": gt["x"], "gt": gt["gt"], "oracle_x": oracle["x"], "oracle": oracle["gt"],
            "members": members["events"], "members_first_clip": members["first_clip_s"],
            "arm_b": arm_b["events"], "arm_b_first_clip": arm_b["first_clip_s"], "arm_b_nonfinite": arm_b["nonfinite"],
            "horizon": 140.0, "blanking_s": 0.0, "horizons": (60.0, 90.0, 140.0),
        })
    return out


def multi_node_inputs(label: str, system: str, model: str = "rate", n_members: int = 20, n_rollouts: int = 20,
                      suffix: str = "") -> dict | None:
    """suffix "" = the pipeline's fresh test cohort, "_fine" = dense transition cohort (docs/05 §1b)."""
    path = HERE / "outputs" / label / f"cross_inputs_{model}{suffix}.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        inp = {k: d[k] for k in d.files}
    ens, arm_b = Path(str(inp["ensemble_dir"])), Path(str(inp["arm_b_dir"]))
    if not (arm_b / f"rollout_{n_rollouts - 1:04d}.npz").exists():
        return None

    def stack(folder: Path, prefix: str, n: int, key: str) -> np.ndarray:
        return np.vstack([np.load(folder / f"{prefix}_{i:04d}.npz")[key] for i in range(n)])

    return {
        "system": system, "cohort": f"{'dense transition' if suffix else 'fresh test'} ({model} model)", "x_label": "max x0",
        "x": inp["x_trial"], "gt": inp["gt_events"], "oracle_x": inp["oracle_x"], "oracle": inp["oracle_events"],
        "members": stack(ens, "member", n_members, "events"),
        "members_first_clip": stack(ens, "member", n_members, "first_clip_s"),
        "arm_b": stack(arm_b, "rollout", n_rollouts, "events"),
        "arm_b_first_clip": stack(arm_b, "rollout", n_rollouts, "first_clip_s"),
        "arm_b_nonfinite": stack(arm_b, "rollout", n_rollouts, "nonfinite"),
        "horizon": float(inp["horizon"]), "blanking_s": float(inp["blanking_s"]), "horizons": (60.0, 90.0, float(inp["horizon"])),
    }


# ---- arm forecasts --------------------------------------------------------------

def level_freq(x: np.ndarray, events: np.ndarray, h: float, grid: np.ndarray) -> np.ndarray:
    """Departure frequency at or before h per level, pooled over rows (rollouts) and trials."""
    events = np.atleast_2d(events)
    hit = np.isfinite(events) & (events <= h)
    return np.array([hit[:, x == v].mean() for v in grid])


def level_survival_curves(x: np.ndarray, events: np.ndarray, grid: np.ndarray, times: np.ndarray) -> np.ndarray:
    """S(t | level) pooled over rows and trials. With only administrative
    censoring at H this equals the Kaplan-Meier curve (and the mean of the
    per-rollout KM curves, since every rollout has the same trials)."""
    events = np.atleast_2d(events)
    ev = np.where(np.isfinite(events), events, np.inf)
    return np.vstack([(ev[:, x == v][..., None] > times).mean(axis=(0, 1)) for v in grid])


def arms(inp: dict) -> dict[str, np.ndarray]:
    m = inp["members"]
    return {"A": m[:1], "B": inp["arm_b"], "E": m, "F": m[1:]}


# ---- scoring --------------------------------------------------------------------

def score(inp: dict, rng: np.random.Generator) -> dict:
    x, gt = inp["x"], inp["gt"]
    grid = np.unique(x)
    gi = np.searchsorted(grid, x)
    H = inp["horizon"]
    arm_events = arms(inp)
    J = inp["members"].shape[0]
    res: dict = {
        "system": inp["system"], "cohort": inp["cohort"], "n_trials": int(len(x)), "n_levels": int(len(grid)),
        "rollouts_per_trial": {k: int(v.shape[0]) for k, v in arm_events.items()},
        "gates": {
            "B_rollout0_equals_member0": bool(np.allclose(inp["arm_b"][0], inp["members"][0], equal_nan=True)),
        },
        "arm_b_health": _health(inp["arm_b"], inp["arm_b_first_clip"], inp["arm_b_nonfinite"]),
        "horizons": {},
    }

    for h in inp["horizons"]:
        o = (np.isfinite(gt) & (gt <= h)).astype(float)
        o_freq = np.array([o[x == v].mean() for v in grid])
        n_per = np.array([np.sum(x == v) for v in grid])
        k_gt = o_freq * n_per
        fit_gt = dm.fit_logistic_survival(grid, k_gt, n_per)
        p_level = {k: level_freq(x, ev, h, grid) for k, ev in arm_events.items()}
        p_level["oracle"] = level_freq(inp["oracle_x"], inp["oracle"], h, grid)
        p_trial = {k: v[gi] for k, v in p_level.items()}
        p_trial["climatology"] = dm.climatology(o)

        table = {}
        for name, p in p_trial.items():
            row = {"brier": dm.brier_score(p, o), "bss": dm.brier_skill_score(p, o, p_trial["climatology"])}
            mur = dm.murphy_decomposition(p, o, n_bins=None)
            row["rel"], row["res"] = float(mur["rel"]), float(mur["res"])
            if name in p_level:
                row["mae_original"] = float(np.mean(np.abs(p_level[name] - o_freq)))
                fit = dm.fit_logistic_survival(grid, p_level[name] * n_per, n_per)
                row["x50"] = fit["x50"]
            table[name] = row

        # C: each member alone (distribution over members)
        member_brier = np.array([dm.brier_score(level_freq(x, inp["members"][j], h, grid)[gi], o) for j in range(J)])
        member_mae = np.array([np.mean(np.abs(level_freq(x, inp["members"][j], h, grid) - o_freq)) for j in range(J)])
        table["C"] = {
            "brier_median": float(np.median(member_brier)), "brier_5_95": _q(member_brier),
            "brier_min_max": [float(member_brier.min()), float(member_brier.max())],
            "mae_original_median": float(np.median(member_mae)), "mae_original_5_95": _q(member_mae),
            "member0_brier_rank": int(np.sum(member_brier < member_brier[0])) + 1,
        }

        # paired differences, cluster bootstrap over levels
        se = {k: (p_trial[k] - o) ** 2 for k in ("A", "B", "E", "F")}
        ae = {k: np.abs(p_level[k] - o_freq) for k in ("A", "B", "E", "F")}
        idx_by_level = [np.flatnonzero(x == v) for v in grid]
        boot_levels = rng.integers(0, len(grid), size=(N_BOOT, len(grid)))
        diffs = {}
        for a, b in PAIRS:
            d_trial = se[a] - se[b]
            d_level = ae[a] - ae[b]
            b_brier = np.array([np.concatenate([d_trial[idx_by_level[g]] for g in lv]).mean() for lv in boot_levels])
            b_mae = d_level[boot_levels].mean(axis=1)
            diffs[f"{a}-{b}"] = {
                "brier": float(d_trial.mean()), "brier_95": _q(b_brier, (0.025, 0.975)),
                "mae_original": float(d_level.mean()), "mae_original_95": _q(b_mae, (0.025, 0.975)),
            }
        res["horizons"][str(int(h))] = {
            "event_rate": float(o.mean()), "gt_x50": fit_gt["x50"], "arms": table, "paired_differences": diffs,
        }

    # time dimension (over the full horizon)
    times = np.linspace(max(1.0, inp["blanking_s"]), H - 1.0, 120)
    dur = np.where(np.isfinite(gt), np.minimum(gt, H), H)
    obs = np.isfinite(gt) & (gt <= H)
    time_rows = {}
    curve_sets = {k: level_survival_curves(x, ev, grid, times) for k, ev in arm_events.items()}
    curve_sets["oracle"] = level_survival_curves(inp["oracle_x"], inp["oracle"], grid, times)
    for name, curves in curve_sets.items():
        s_at = np.array([np.interp(t, times, curves[g], left=1.0, right=curves[g, -1]) for t, g in zip(dur, gi)])
        time_rows[name] = {
            "ibs": dm.integrated_brier_score(dur, obs, curves[gi], times),
            "d_calibration_p": dm.d_calibration(s_at, s_at, obs, rng=np.random.default_rng(12))["p_value"],
        }
    # IBS paired differences: with only administrative censoring at H and
    # times < H, IPCW weights are 1 and IBS = mean over trials of each trial's
    # time-integrated squared error, so it splits per trial (checked below).
    def trial_ibs(curves: np.ndarray) -> np.ndarray:
        alive = (np.where(np.isfinite(gt), gt, np.inf)[:, None] > times).astype(float)
        return np.trapezoid((alive - curves[gi]) ** 2, times, axis=1) / (times[-1] - times[0])

    per_trial = {k: trial_ibs(curve_sets[k]) for k in ("A", "B", "E", "F")}
    for k in per_trial:
        if not np.isclose(per_trial[k].mean(), time_rows[k]["ibs"], rtol=1e-6, atol=1e-9):
            raise RuntimeError(f"per-trial IBS split does not reproduce IBS for arm {k}")
    idx_by_level = [np.flatnonzero(x == v) for v in grid]
    boot_levels = rng.integers(0, len(grid), size=(N_BOOT, len(grid)))
    time_rows["paired_differences_ibs"] = {}
    for a, b in PAIRS:
        d = per_trial[a] - per_trial[b]
        boot = np.array([np.concatenate([d[idx_by_level[g]] for g in lv]).mean() for lv in boot_levels])
        time_rows["paired_differences_ibs"][f"{a}-{b}"] = {"ibs": float(d.mean()), "ibs_95": _q(boot, (0.025, 0.975))}
    member_ibs = np.array([
        dm.integrated_brier_score(dur, obs, level_survival_curves(x, inp["members"][j], grid, times)[gi], times) for j in range(J)
    ])
    time_rows["C"] = {"ibs_median": float(np.median(member_ibs)), "ibs_5_95": _q(member_ibs)}
    res["time"] = time_rows
    return res


def _health(events: np.ndarray, first_clip: np.ndarray, nonfinite: np.ndarray) -> dict:
    departed = np.isfinite(events)
    saturation = departed & np.isfinite(first_clip) & (first_clip <= events + 0.5)
    return {
        "nonfinite_trials": int(nonfinite.sum()),
        "departures": int(departed.sum()),
        "saturation_share": float(saturation.sum() / max(departed.sum(), 1)),
    }


def _q(v: np.ndarray, qs=(0.05, 0.95)) -> list[float]:
    return [float(np.quantile(v, q)) for q in qs]


# ---- report -----------------------------------------------------------------------

def _f(v: float, nd: int = 3) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def markdown(results: list[dict]) -> str:
    lines = ["# Cross table (generated by `survival_dist/cross_table.py`; read with docs/04)", ""]
    for r in results:
        H = max(r["horizons"], key=float)
        lines += [f"## {r['system']} — {r['cohort']}", "",
                  f"trials {r['n_trials']}, levels {r['n_levels']}, rollouts per trial {r['rollouts_per_trial']}, "
                  f"gate B[0] == member 0: {r['gates']['B_rollout0_equals_member0']}, arm-B health {r['arm_b_health']}", ""]
        for h, hr in r["horizons"].items():
            a = hr["arms"]
            lines += [f"### h = {h} s (event rate {hr['event_rate']:.3f}, GT x50 {_f(hr['gt_x50'], 4)})" + ("  ← H" if h == H else ""), "",
                      "| arm | MAE (original) | Brier | BSS | REL | RES | x50 |", "|---|---|---|---|---|---|---|"]
            for k in ("A", "B", "E", "F", "oracle", "climatology"):
                v = a[k]
                lines.append(f"| {k} | {_f(v.get('mae_original'))} | {_f(v['brier'])} | {_f(v['bss'], 2)} | {_f(v['rel'])} | {_f(v['res'])} | {_f(v.get('x50'), 4)} |")
            c = a["C"]
            lines.append(f"| C (members, median [5–95%]) | {_f(c['mae_original_median'])} [{_f(c['mae_original_5_95'][0])}, {_f(c['mae_original_5_95'][1])}] "
                         f"| {_f(c['brier_median'])} [{_f(c['brier_5_95'][0])}, {_f(c['brier_5_95'][1])}] | | | | member 0 rank {c['member0_brier_rank']} |")
            lines += ["", "| paired difference | ΔBrier [95% cluster CI] | ΔMAE [95% cluster CI] |", "|---|---|---|"]
            for k, d in hr["paired_differences"].items():
                lines.append(f"| {k} | {d['brier']:+.4f} [{d['brier_95'][0]:+.4f}, {d['brier_95'][1]:+.4f}] "
                             f"| {d['mae_original']:+.4f} [{d['mae_original_95'][0]:+.4f}, {d['mae_original_95'][1]:+.4f}] |")
            lines.append("")
        t = r["time"]
        lines += [f"### time (IBS over the horizon, D-calibration)", "", "| arm | IBS | D-cal p |", "|---|---|---|"]
        for k in ("A", "B", "E", "F", "oracle"):
            lines.append(f"| {k} | {_f(t[k]['ibs'])} | {t[k]['d_calibration_p']:.2g} |")
        lines += [f"| C (median [5–95%]) | {_f(t['C']['ibs_median'])} [{_f(t['C']['ibs_5_95'][0])}, {_f(t['C']['ibs_5_95'][1])}] | |", "",
                  "| paired difference | ΔIBS [95% cluster CI] |", "|---|---|"]
        for k, d in t["paired_differences_ibs"].items():
            lines.append(f"| {k} | {d['ibs']:+.4f} [{d['ibs_95'][0]:+.4f}, {d['ibs_95'][1]:+.4f}] |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    inputs = single_node_inputs()
    for label, system in (("two_node", "2-node"), ("three_node", "3-node")):
        inp = multi_node_inputs(label, system)
        if inp is None:
            print(f"{system}: arm B or cross inputs missing, skipped", flush=True)
        else:
            inputs.append(inp)
    results = [score(inp, rng) for inp in inputs]
    (OUT / "cross_table.json").write_text(json.dumps(results, indent=2, default=float))
    (OUT / "cross_table.md").write_text(markdown(results))
    rows = []
    for r in results:
        for h, hr in r["horizons"].items():
            for k in ("A", "B", "E", "F", "oracle", "climatology"):
                rows.append({"system": r["system"], "cohort": r["cohort"], "h": h, "arm": k, **{
                    m: hr["arms"][k].get(m, float("nan")) for m in ("mae_original", "brier", "bss", "rel", "res", "x50")}})
    with (OUT / "cross_table.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in results:
        print(r["system"], r["cohort"], "gate:", r["gates"], flush=True)


if __name__ == "__main__":
    main()
