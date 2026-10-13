"""Seed-wise critical values: one survival curve per reservoir -> one critical value per seed.

Each seed-ensemble member is a complete single-RC experiment in the original
sense: its trials give a survival curve S_j(h | lambda) over the control
parameter lambda (x0, or max x0 for networks), and that curve gives a critical
value lambda_c,j where it first crosses 50 %. The ensemble is then used as the
distribution of lambda_c over reservoir seeds, shown as a histogram against GT.

References drawn on the same axis:
  GT          lambda_c from the GT curve, and its bootstrap distribution
              (trials resampled within each level; finite-sample noise only)
  oracle      lambda_c from an independent-seed GT cohort
  arm B       lambda_c of the member-0 reservoir from each of K innovation
              rollouts: the spread a single reservoir shows from rollout noise
              alone, at the same trial count as a member (docs/04)
Spread beyond arm B is attributable to the reservoir seed.

Critical value = the original code's `empirical_x50`
(experiment_three_node_scdt_empirical_boundary_bo.py): the first level where
S = 0.5, else linear interpolation at the first 0.5 crossing. A curve that never
crosses 0.5 inside the scanned range is kept, not dropped (no refill, docs/03):
it is censored, "< range" (already below 50 % at the lowest lambda) or
"> range" (still above 50 % at the highest lambda).

The time-axis views stay alongside (docs/05 V1): S(t | lambda) heatmaps and the
median departure time T50(lambda).

Run:   .venv/bin/python survival_dist/critical_values.py
Output: outputs/critical_values/<system>/ (summary.json, per_seed.csv, figures)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dist_metrics as dm  # noqa: E402
import survival as sv  # noqa: E402
from cross_table import multi_node_inputs, single_node_inputs  # noqa: E402
from plotting import SERIES, TEXT, TEXT_SECONDARY, apply_style, plt  # noqa: E402

OUT = HERE / "outputs" / "critical_values"
N_BOOT = 2000
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])


# ---- critical value of one curve ------------------------------------------------------

def empirical_x50(x: np.ndarray, s: np.ndarray) -> tuple[float, str, int]:
    """(lambda_c, status, number of 0.5 crossings). Same rule as the original code;
    status is 'ok', 'below_range' (S < 0.5 everywhere) or 'above_range' (S > 0.5 everywhere)."""
    order = np.argsort(x)
    x, s = np.asarray(x)[order], np.asarray(s)[order]
    n_cross = int(np.sum((s[:-1] - 0.5) * (s[1:] - 0.5) < 0.0) + np.sum(np.isclose(s, 0.5)))
    exact = np.flatnonzero(np.isclose(s, 0.5))
    if exact.size:
        return float(x[exact[0]]), "ok", n_cross
    crossing = np.flatnonzero((s[:-1] - 0.5) * (s[1:] - 0.5) < 0.0)
    if crossing.size:
        i = int(crossing[0])
        return float(x[i] + (0.5 - s[i]) * (x[i + 1] - x[i]) / (s[i + 1] - s[i])), "ok", n_cross
    return float("nan"), ("below_range" if np.all(s < 0.5) else "above_range"), 0


def survival_curve(x: np.ndarray, events: np.ndarray, h: float, grid: np.ndarray) -> np.ndarray:
    """S(h | level) from one row of events (one reservoir, one rollout per trial)."""
    hit = np.isfinite(events) & (events <= h)
    return np.array([1.0 - hit[x == v].mean() for v in grid])


def critical_values(x: np.ndarray, rows: np.ndarray, h: float, grid: np.ndarray) -> dict:
    vals, status, ncross, logit = [], [], [], []
    n_per = np.array([np.sum(x == v) for v in grid])
    for r in np.atleast_2d(rows):
        s = survival_curve(x, r, h, grid)
        v, st, nc = empirical_x50(grid, s)
        vals.append(v), status.append(st), ncross.append(nc)
        fit = dm.fit_logistic_survival(grid, (1 - s) * n_per, n_per)
        lv = fit["x50"]
        logit.append(lv if np.isfinite(lv) and grid.min() <= lv <= grid.max() else np.nan)
    return {"x50": np.array(vals), "status": np.array(status), "n_crossings": np.array(ncross), "x50_logistic": np.array(logit)}


def gt_bootstrap(x: np.ndarray, gt: np.ndarray, h: float, grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """lambda_c of GT curves with trials resampled within each level."""
    idx = [np.flatnonzero(x == v) for v in grid]
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = np.array([1.0 - np.mean(np.isfinite(gt[i]) & (gt[i] <= h)) for i in (rng.choice(ii, len(ii)) for ii in idx)])
        out[b] = empirical_x50(grid, s)[0]
    return out


# ---- per system -------------------------------------------------------------------------

def pool_single_node(parts: list[dict]) -> dict:
    """Coarse + fine cohorts as one trial set (each member ran both; levels do not coincide)."""
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    hcat = lambda k: np.hstack([p[k] for p in parts])  # noqa: E731
    out = dict(parts[0])
    out.update(cohort="coarse + fine", x=cat("x"), gt=cat("gt"), oracle_x=cat("oracle_x"), oracle=cat("oracle"),
               members=hcat("members"), arm_b=hcat("arm_b"))
    if len(np.unique(out["x"])) != sum(len(np.unique(p["x"])) for p in parts):
        raise RuntimeError("coarse and fine levels coincide; pooling would merge trial counts")
    return out


def analyze(inp: dict, label: str, rng: np.random.Generator) -> dict:
    out_dir = OUT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    x, gt = inp["x"], inp["gt"]
    grid = np.unique(x)
    res = {"system": inp["system"], "cohort": inp["cohort"], "n_levels": int(len(grid)),
           "lambda_range": [float(grid.min()), float(grid.max())],
           "trials_per_level": sorted({int(np.sum(x == v)) for v in grid}),
           "n_members": int(inp["members"].shape[0]), "n_arm_b": int(inp["arm_b"].shape[0]), "horizons": {}}
    store, rows = {}, []
    for h in inp["horizons"]:
        mem = critical_values(x, inp["members"], h, grid)
        armb = critical_values(x, inp["arm_b"], h, grid)
        gt_v = empirical_x50(grid, survival_curve(x, gt, h, grid))
        if len(inp["oracle_x"]):
            or_grid = np.unique(inp["oracle_x"])
            or_v = empirical_x50(or_grid, survival_curve(inp["oracle_x"], inp["oracle"], h, or_grid))
        else:  # dense cohort has no independent-seed GT cohort
            or_v = (float("nan"), "none", 0)
        boot = gt_bootstrap(x, gt, h, grid, rng)
        n_per = np.array([np.sum(x == v) for v in grid])
        gt_fit = dm.fit_logistic_survival(grid, (1 - survival_curve(x, gt, h, grid)) * n_per, n_per)["x50"]
        store[h] = {"members": mem, "arm_b": armb, "gt": gt_v[0], "oracle": or_v[0], "gt_boot": boot, "gt_logistic": gt_fit}
        res["horizons"][str(int(h))] = _summary(mem, armb, gt_v, or_v, boot)
        res["horizons"][str(int(h))]["logistic"] = _logistic_summary(mem, armb, gt_fit)
        for j in range(len(mem["x50"])):
            rows.append({"horizon": h, "source": "member", "index": j, "x50": mem["x50"][j], "status": mem["status"][j],
                         "n_crossings": int(mem["n_crossings"][j]), "x50_logistic": mem["x50_logistic"][j]})
        for k in range(len(armb["x50"])):
            rows.append({"horizon": h, "source": "arm_b", "index": k, "x50": armb["x50"][k], "status": armb["status"][k],
                         "n_crossings": int(armb["n_crossings"][k]), "x50_logistic": armb["x50_logistic"][k]})
    res["time_view"] = _time_view(inp, grid, out_dir)
    with (out_dir / "per_seed.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    _plot_hist(store, inp, grid, out_dir)
    _plot_hist(store, inp, grid, out_dir, key="x50_logistic")
    _plot_curves(inp, grid, store, out_dir)
    return res


def _summary(mem: dict, armb: dict, gt_v, or_v, boot: np.ndarray) -> dict:
    v, st = mem["x50"], mem["status"]
    ok = st == "ok"
    vb = armb["x50"][armb["status"] == "ok"]
    gt = gt_v[0]
    lo, hi = np.nanquantile(boot, [0.025, 0.975]) if np.isfinite(boot).any() else (np.nan, np.nan)
    d = {
        "gt": gt, "gt_status": gt_v[1], "gt_boot_95": [float(lo), float(hi)], "gt_boot_sd": float(np.nanstd(boot)),
        "gates": {"arm_b_rollout0_equals_member0": bool(np.allclose(armb["x50"][0], mem["x50"][0], equal_nan=True))},
        "oracle": or_v[0],
        "members": {
            "identifiable": int(ok.sum()), "below_range": int(np.sum(st == "below_range")), "above_range": int(np.sum(st == "above_range")),
            "multi_crossing": int(np.sum(mem["n_crossings"] > 1)),
            "median": _nanq(v[ok], 0.5), "q05_q95": [_nanq(v[ok], 0.05), _nanq(v[ok], 0.95)], "iqr": [_nanq(v[ok], 0.25), _nanq(v[ok], 0.75)],
            "sd": float(np.std(v[ok], ddof=1)) if ok.sum() > 1 else float("nan"),
            "member0": float(v[0]), "member0_status": str(st[0]),
            "logistic_vs_empirical_median_abs_diff": _nanq(np.abs(mem["x50_logistic"][ok] - v[ok]), 0.5),
        },
        "arm_b": {
            "identifiable": int(len(vb)), "median": _nanq(vb, 0.5), "q05_q95": [_nanq(vb, 0.05), _nanq(vb, 0.95)],
            "sd": float(np.std(vb, ddof=1)) if len(vb) > 1 else float("nan"),
        },
    }
    m = d["members"]
    if np.isfinite(gt):
        # censored members count on their side of GT
        below = np.sum(v[ok] < gt) + np.sum(st == "below_range")
        d["gt_percentile_in_members"] = float(below / len(v))
        d["members_inside_gt_boot_95"] = int(np.sum((v[ok] >= lo) & (v[ok] <= hi)))
        d["member_bias_median_minus_gt"] = m["median"] - gt if np.isfinite(m["median"]) else float("nan")
        d["crps_identifiable_members_vs_gt"] = dm.crps_ensemble(v[ok], gt) if ok.any() else float("nan")
    d["spread_ratio_members_over_arm_b_sd"] = m["sd"] / d["arm_b"]["sd"] if d["arm_b"]["sd"] and np.isfinite(d["arm_b"]["sd"]) and d["arm_b"]["sd"] > 0 else float("nan")
    return d


def _logistic_summary(mem: dict, armb: dict, gt_fit: float) -> dict:
    """Same summary with the binomial-logistic x50: uses the whole curve, so it is far
    less sensitive than the first crossing to wiggles around 0.5 on dense grids."""
    m, b = mem["x50_logistic"], armb["x50_logistic"]
    mf, bf = m[np.isfinite(m)], b[np.isfinite(b)]
    sd_m = float(np.std(mf, ddof=1)) if mf.size > 1 else float("nan")
    sd_b = float(np.std(bf, ddof=1)) if bf.size > 1 else float("nan")
    return {
        "gt": float(gt_fit), "members_identifiable": int(mf.size), "members_median": _nanq(mf, 0.5),
        "members_q05_q95": [_nanq(mf, 0.05), _nanq(mf, 0.95)], "members_sd": sd_m, "member0": float(m[0]),
        "arm_b_median": _nanq(bf, 0.5), "arm_b_sd": sd_b,
        "seed_share_of_variance": float(1 - (sd_b / sd_m) ** 2) if np.isfinite(sd_m) and sd_m > 0 else float("nan"),
        "gt_percentile_in_members": float(np.mean(mf < gt_fit)) if mf.size and np.isfinite(gt_fit) else float("nan"),
    }


def _nanq(v: np.ndarray, q: float) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if v.size else float("nan")


# ---- time view (V1) ------------------------------------------------------------------------

def _level_surv_t(x, events, grid, times):
    events = np.atleast_2d(events)
    ev = np.where(np.isfinite(events), events, np.inf)
    return np.vstack([(ev[:, x == v][..., None] > times).mean(axis=(0, 1)) for v in grid])


def _t50(curve: np.ndarray, times: np.ndarray) -> float:
    below = np.flatnonzero(curve <= 0.5)
    return float(times[below[0]]) if below.size else float("nan")


def _time_view(inp: dict, grid: np.ndarray, out_dir: Path) -> dict:
    x, H = inp["x"], inp["horizon"]
    times = np.arange(max(0.5, inp["blanking_s"]), H + 1e-9, 0.5)
    gt_c = _level_surv_t(x, inp["gt"], grid, times)
    ens_c = _level_surv_t(x, inp["members"], grid, times)
    m0_c = _level_surv_t(x, inp["members"][:1], grid, times)
    mem_t50 = np.array([[_t50(c, times) for c in _level_surv_t(x, inp["members"][j:j + 1], grid, times)] for j in range(inp["members"].shape[0])])
    t50 = {"gt": [_t50(c, times) for c in gt_c], "member0": [_t50(c, times) for c in m0_c], "ensemble": [_t50(c, times) for c in ens_c]}
    _plot_time(inp, grid, times, gt_c, ens_c, m0_c, t50, mem_t50, out_dir)
    dep = np.isfinite(t50["gt"]) & np.isfinite(t50["ensemble"])
    return {"levels_with_gt_t50": int(np.isfinite(t50["gt"]).sum()),
            "median_abs_t50_error_ensemble_s": _nanq(np.abs(np.array(t50["ensemble"]) - np.array(t50["gt"]))[dep], 0.5),
            "median_abs_t50_error_member0_s": _nanq(np.abs(np.array(t50["member0"]) - np.array(t50["gt"])), 0.5)}


# ---- plots -----------------------------------------------------------------------------------

def _edges(grid: np.ndarray) -> np.ndarray:
    mid = (grid[:-1] + grid[1:]) / 2
    return np.r_[grid[0] - (mid[0] - grid[0]), mid, grid[-1] + (grid[-1] - mid[-1])]


def _plot_hist(store: dict, inp: dict, grid: np.ndarray, out_dir: Path, key: str = "x50") -> None:
    """key "x50" = original first-crossing rule; "x50_logistic" = binomial-logistic fit."""
    logistic = key == "x50_logistic"
    gt_key = "gt_logistic" if logistic else "gt"
    apply_style()
    hs = list(store)
    rc_c, gt_c, or_c = SERIES["rc"]["color"], SERIES["gt"]["color"], SERIES["oracle"]["color"]
    finite = np.concatenate([np.r_[d["members"][key], d["arm_b"][key], d[gt_key], d["oracle"]] for d in store.values()])
    finite = finite[np.isfinite(finite)]
    lo, hi = finite.min(), finite.max()
    pad = 0.08 * (hi - lo + 1e-3)
    width = max((hi - lo + 2 * pad) / 40, 1e-4)
    bins = np.arange(lo - pad, hi + pad + width, width)
    fig, axes = plt.subplots(len(hs), 1, figsize=(9.5, 2.5 * len(hs) + 0.6), sharex=True)
    for ax, h in zip(np.atleast_1d(axes), hs):
        d = store[h]
        mv, ms = d["members"][key], d["members"]["status"]
        bv = d["arm_b"][key]
        b_lo, b_hi = np.nanquantile(d["gt_boot"], [0.025, 0.975]) if (not logistic and np.isfinite(d["gt_boot"]).any()) else (np.nan, np.nan)
        if np.isfinite(b_lo):
            ax.axvspan(b_lo, b_hi, color=gt_c, alpha=0.14, lw=0, label="GT λc, 95% bootstrap")
        ax.hist(mv[np.isfinite(mv)], bins=bins, color=rc_c, alpha=0.85, edgecolor="white", linewidth=1.0,
                label=f"RC λc per reservoir seed (n = {np.isfinite(mv).sum()})")
        ax.hist(bv[np.isfinite(bv)], bins=bins, histtype="step", color=TEXT_SECONDARY, linewidth=1.6, linestyle="--",
                label="member-0 reservoir, innovation noise only (arm B)")
        if np.isfinite(d[gt_key]):
            ax.axvline(d[gt_key], color=gt_c, lw=2.2, label="GT λc")
        if not logistic and np.isfinite(d["oracle"]):
            ax.axvline(d["oracle"], color=or_c, lw=1.6, ls=":", label="GT λc, independent seeds")
        if np.isfinite(mv[0]):
            ax.plot(mv[0], 0, marker="D", color=TEXT, ms=7, clip_on=False, zorder=5, lw=0, label="member 0 (original RC)")
        n_below, n_above = int(np.sum(ms == "below_range")), int(np.sum(ms == "above_range"))
        note = []
        if n_below:
            note.append(f"{n_below} seed(s): survival < 50% across the whole range (λc < {grid.min():.3f})")
        if n_above:
            note.append(f"{n_above} seed(s): survival > 50% across the whole range (λc > {grid.max():.3f})")
        if logistic:
            n_out = int(np.sum(~np.isfinite(mv)))
            note = [f"{n_out} seed(s): logistic 50% point outside the range"] if n_out else []
        if note:
            ax.set_title("; ".join(note), loc="right", fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(f"survival to {h:.0f} s", loc="left")
        ax.set_ylabel("count (seeds or rollouts)")
    rule = "50% point of a binomial-logistic fit to the whole curve" if logistic else "where survival first crosses 50%, original rule"
    np.atleast_1d(axes)[-1].set_xlabel(f"critical value λc ({inp['x_label']}; {rule})")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"{inp['system']} ({inp['cohort']}): critical value per reservoir seed vs GT", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / ("critical_value_hist_logistic.png" if logistic else "critical_value_hist.png"), dpi=200)
    plt.close(fig)


def _plot_curves(inp: dict, grid: np.ndarray, store: dict, out_dir: Path) -> None:
    """V2: S_j(H | lambda) per seed, with GT."""
    apply_style()
    rc_c, gt_c = SERIES["rc"]["color"], SERIES["gt"]["color"]
    x, H = inp["x"], inp["horizon"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for j, r in enumerate(inp["members"]):
        ax.plot(grid, survival_curve(x, r, H, grid), color=rc_c, lw=1.0, alpha=0.35, label="RC, one curve per reservoir seed" if j == 0 else None)
    ax.plot(grid, survival_curve(x, inp["members"][0], H, grid), color=TEXT, lw=1.6, ls="--", label="member 0 (original RC)")
    n = np.array([np.sum(x == v) for v in grid])
    gs = survival_curve(x, inp["gt"], H, grid)
    lo, hi = sv.jeffreys_interval(np.round(gs * n), n)
    ax.errorbar(grid, gs, yerr=[gs - lo, hi - gs], color=gt_c, marker="o", ms=4, lw=0, elinewidth=0.9, label="GT (95% Jeffreys)")
    mv = store[H]["members"]["x50"]
    ax.plot(mv[np.isfinite(mv)], np.full(np.isfinite(mv).sum(), -0.04), "|", color=rc_c, ms=10, mew=1.5, label="λc per seed")
    if np.isfinite(store[H]["gt"]):
        ax.axvline(store[H]["gt"], color=gt_c, lw=1.4, ls=":")
    ax.axhline(0.5, color=TEXT_SECONDARY, lw=0.8)
    ax.set(xlabel=inp["x_label"], ylabel=f"survival to {H:.0f} s", ylim=(-0.08, 1.05))
    ax.set_title(f"{inp['system']} ({inp['cohort']}): survival at the horizon, one curve per seed", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "survival_at_horizon_per_seed.png", dpi=200)
    plt.close(fig)


def _plot_time(inp, grid, times, gt_c, ens_c, m0_c, t50, mem_t50, out_dir):
    """V1: S(t | lambda) heatmaps and T50(lambda)."""
    apply_style()
    xe = _edges(grid)
    te = np.r_[times - 0.25, times[-1] + 0.25]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, c, title in ((axes[0], gt_c, "GT"), (axes[1], ens_c, "RC seed ensemble (mixture)")):
        m = ax.pcolormesh(xe, te, c.T, cmap=SEQUENTIAL.reversed(), vmin=0, vmax=1, shading="flat", rasterized=True)
        ax.set_title(f"{title}: S(t | λ)", loc="left")
        ax.set_xlabel(inp["x_label"])
        ax.grid(False)
    fig.colorbar(m, ax=axes[:2], fraction=0.03, pad=0.01, label="survival probability")
    d = ax = axes[2]
    dm_ = ax.pcolormesh(xe, te, (ens_c - gt_c).T, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1), shading="flat", rasterized=True)
    ax.set_title("RC − GT (red: RC survives longer)", loc="left")
    ax.set_xlabel(inp["x_label"])
    ax.grid(False)
    fig.colorbar(dm_, ax=d, fraction=0.05, pad=0.01, label="difference in survival")
    axes[0].set_ylabel("time since parameter switch (s)")
    fig.suptitle(f"{inp['system']} ({inp['cohort']}): survival over time and λ", fontsize=11)
    fig.savefig(out_dir / "survival_time_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rc_c, gt_c_ = SERIES["rc"]["color"], SERIES["gt"]["color"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    lo = np.nanquantile(mem_t50, 0.05, axis=0)
    hi = np.nanquantile(mem_t50, 0.95, axis=0)
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(grid[ok], lo[ok], hi[ok], color=rc_c, alpha=0.2, lw=0, label="RC seeds 5–95% (seeds with T50 ≤ H)")
    ax.plot(grid, t50["ensemble"], color=rc_c, marker="s", ms=4, label="RC seed ensemble")
    ax.plot(grid, t50["member0"], color=TEXT, ls="--", lw=1.4, label="member 0 (original RC)")
    ax.plot(grid, t50["gt"], color=gt_c_, marker="o", ms=4, label="GT")
    ax.set(xlabel=inp["x_label"], ylabel="T50: median time to departure (s)", ylim=(0, inp["horizon"]))
    ax.set_title(f"{inp['system']} ({inp['cohort']}): median departure time vs λ (missing = fewer than half depart by {inp['horizon']:.0f} s)",
                 loc="left", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "t50_vs_lambda.png", dpi=200)
    plt.close(fig)


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rng = np.random.default_rng(2027)
    jobs = [(pool_single_node(single_node_inputs()), "single_node")]
    for label, system in (("two_node", "2-node"), ("three_node", "3-node")):
        for suffix in ("", "_fine"):
            inp = multi_node_inputs(label, system, suffix=suffix)
            if inp is not None:
                jobs.append((inp, label + suffix))
    results = [analyze(inp, label, rng) for inp, label in jobs]
    (OUT / "summary_all.json").write_text(json.dumps(results, indent=2, default=float))
    for r in results:
        H = max(r["horizons"], key=float)
        s = r["horizons"][H]
        print(r["system"], r["cohort"], f"H={H}", "GT", round(s["gt"], 4), "members", s["members"]["identifiable"], "ok, median",
              round(s["members"]["median"], 4), "sd", round(s["members"]["sd"], 4), "| arm B sd", round(s["arm_b"]["sd"], 4), flush=True)


if __name__ == "__main__":
    main()
