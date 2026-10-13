"""Generic distributional report for a seed ensemble vs GT (any node count).

Inputs are plain arrays, so single- and multi-node experiments share one
analysis (the single-node phase-2 script predates this module and keeps its
own copy of the same computations):

    x_trial      (N,)    grouping coordinate per trial (x0, or max(x0) for networks)
    gt_events    (N,)    GT departure time, s from rollout start, NaN = none
    rc_events    (J, N)  member departure times on the same (paired) trials
    first_clip   (J, N)  first time each RC rollout touched the clip bound (NaN = never)
    oracle_x     (M,)    independent-seed GT cohort, same grid
    oracle_events(M,)
Metric definitions: docs/02_metrics.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import dist_metrics as dm
import survival as sv
from plotting import SERIES, TEXT_SECONDARY, apply_style, plt


def _freq(x_trial, events, h, grid):
    return np.array([np.mean(np.isfinite(events[x_trial == v]) & (events[x_trial == v] <= h)) for v in grid])


def _to_trials(x_trial, grid, values):
    return values[np.searchsorted(grid, x_trial)]


def ensemble_report(
    *,
    x_trial: np.ndarray,
    gt_events: np.ndarray,
    rc_events: np.ndarray,
    first_clip: np.ndarray,
    nonfinite: np.ndarray,
    oracle_x: np.ndarray,
    oracle_events: np.ndarray,
    horizon: float,
    horizons: tuple[float, ...],
    out_dir: Path,
    x_label: str,
    title: str,
    blanking_s: float = 0.0,
    n_boot: int = 1000,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    J, N = rc_events.shape
    grid = np.unique(x_trial)
    n_per = np.array([np.sum(x_trial == v) for v in grid])
    rng = np.random.default_rng(11)

    departed = np.isfinite(rc_events)
    saturation = departed & np.isfinite(first_clip) & (first_clip <= rc_events + 0.5)
    dyn = departed & ~saturation
    gt_dep = np.isfinite(gt_events)
    results: dict = {
        "n_members": J,
        "n_trials": N,
        "n_levels": int(len(grid)),
        "health": {"nonfinite_trials_total": int(nonfinite.sum()), "trials_ever_clipped": int(np.isfinite(first_clip).sum())},
        "departure_types": {
            "rc_departures_total": int(departed.sum()),
            "saturation_departures": int(saturation.sum()),
            "dynamic_departures": int(dyn.sum()),
            "saturation_share": float(saturation.sum() / max(departed.sum(), 1)),
            "dynamic_departure_time_median_s": float(np.median(rc_events[dyn])) if dyn.any() else float("nan"),
            "gt_departure_time_median_s": float(np.median(gt_events[gt_dep])) if gt_dep.any() else float("nan"),
        },
        "horizons": {},
    }

    rows, x50_store = [], {}
    for h in horizons:
        o = (gt_dep & (gt_events <= h)).astype(float)
        p_m = np.vstack([_freq(x_trial, rc_events[j], h, grid) for j in range(J)])
        p_bar = p_m.mean(axis=0)
        p_or = _freq(oracle_x, oracle_events, h, grid)
        fc = {
            "ensemble": _to_trials(x_trial, grid, p_bar),
            "member0": _to_trials(x_trial, grid, p_m[0]),
            "oracle": _to_trials(x_trial, grid, p_or),
            "climatology": dm.climatology(o),
        }
        binary = {
            k: {"brier": dm.brier_score(p, o), "bss": dm.brier_skill_score(p, o, fc["climatology"]),
                **{f"murphy_{a}": b for a, b in dm.murphy_decomposition(p, o, n_bins=10).items()}}
            for k, p in fc.items()
        }
        member_brier = np.array([dm.brier_score(_to_trials(x_trial, grid, p_m[j]), o) for j in range(J)])
        dec = dm.uncertainty_decomposition(p_m)
        within = np.mean(p_m * (1 - p_m), axis=0)
        epi_corr = dec["epistemic"] - within / np.maximum(n_per - 1, 1)
        cov, lo, hi = dm.band_coverage(p_m, p_or)

        k_gt = np.array([o[x_trial == v].sum() for v in grid])
        fit_gt = dm.fit_logistic_survival(grid, k_gt, n_per)
        boot = dm.bootstrap_logistic(grid, k_gt, n_per, n_boot, rng)
        fits = [dm.fit_logistic_survival(grid, np.round(p_m[j] * n_per), n_per) for j in range(J)]
        x50 = np.array([f["x50"] for f in fits])
        s50 = np.array([f["s"] for f in fits])
        ident = np.isfinite(x50) & (x50 >= grid.min()) & (x50 <= grid.max()) & (s50 < np.ptp(grid))
        xv = x50[ident]
        x50_store[h] = {"members": np.where(ident, x50, np.nan), "gt": fit_gt["x50"], "gt_boot": boot[:, 0]}
        gl, gh = np.nanquantile(boot[:, 0], [0.025, 0.975])
        results["horizons"][str(int(h))] = {
            "event_rate": float(o.mean()),
            "binary": binary,
            "member_brier_min_median_max": [float(member_brier.min()), float(np.median(member_brier)), float(member_brier.max())],
            "member_brier_all": member_brier.tolist(),
            "band_coverage_5_95": cov,
            "band_mean_width_5_95": float(np.mean(hi - lo)),
            "x50": {
                "gt": fit_gt["x50"], "gt_s": fit_gt["s"], "gt_boot_95": [float(gl), float(gh)],
                "gt_n_mixed_levels": int(np.count_nonzero((k_gt > 0) & (k_gt < n_per))),
                "members_identifiable": int(ident.sum()),
                "members_median": float(np.median(xv)) if xv.size else float("nan"),
                "members_5_95": [float(np.quantile(xv, 0.05)), float(np.quantile(xv, 0.95))] if xv.size else [float("nan")] * 2,
                "crps_identifiable_members_vs_gt": dm.crps_ensemble(xv, fit_gt["x50"]),
                "member0": float(x50[0]) if ident[0] else float("nan"),
            },
            "uncertainty_mean": {"total": float(dec["total"].mean()), "aleatoric": float(dec["aleatoric"].mean()),
                                 "epistemic": float(dec["epistemic"].mean()), "epistemic_corrected": float(epi_corr.mean())},
        }
        for g, v in enumerate(grid):
            rows.append({"horizon": h, "x": float(v), "n": int(n_per[g]), "gt_event_freq": float(k_gt[g] / n_per[g]),
                         "oracle_event_freq": float(p_or[g]), "ens_mean": float(p_bar[g]), "ens_q05": float(lo[g]),
                         "ens_q95": float(hi[g]), "member0": float(p_m[0, g]), "aleatoric": float(dec["aleatoric"][g]),
                         "epistemic_corrected": float(epi_corr[g])})

    # time dimension
    times = np.linspace(max(1.0, blanking_s), horizon - 1.0, 120)
    curves = np.empty((J, len(grid), len(times)))
    for j in range(J):
        for g, v in enumerate(grid):
            curves[j, g] = sv.kaplan_meier(*sv.censor(rc_events[j, x_trial == v], horizon)).at(times)
    s_bar = curves.mean(axis=0)
    gi = np.searchsorted(grid, x_trial)
    dur, obs = sv.censor(gt_events, horizon)
    or_curves = np.vstack([sv.kaplan_meier(*sv.censor(oracle_events[oracle_x == v], horizon)).at(times) for v in grid])
    pooled = sv.kaplan_meier(*sv.censor(oracle_events, horizon)).at(times)
    member_ibs = np.array([dm.integrated_brier_score(dur, obs, curves[j][gi], times) for j in range(J)])
    s_at = np.array([np.interp(t, times, s_bar[g], left=1.0, right=s_bar[g, -1]) for t, g in zip(dur, gi)])
    dcal = dm.d_calibration(s_at, s_at, obs, rng=np.random.default_rng(12))
    results["time"] = {
        "ibs": {
            "ensemble_mixture": dm.integrated_brier_score(dur, obs, s_bar[gi], times),
            "member0": float(member_ibs[0]),
            "oracle": dm.integrated_brier_score(dur, obs, or_curves[gi], times),
            "pooled_km_reference": dm.integrated_brier_score(dur, obs, np.tile(pooled, (N, 1)), times),
            "member_min_median_max": [float(member_ibs.min()), float(np.median(member_ibs)), float(member_ibs.max())],
        },
        "d_calibration_mixture_p": dcal["p_value"],
    }

    # per-member table
    member_rows = []
    h_last = horizons[-1]
    o_last = (gt_dep & (gt_events <= h_last)).astype(float)
    for j in range(J):
        d = departed[j]
        member_rows.append({
            "member": j,
            "false_departure_rate": float(d[~gt_dep].mean()) if (~gt_dep).any() else float("nan"),
            "miss_rate": float((~d[gt_dep]).mean()) if gt_dep.any() else float("nan"),
            "saturation_share": float(saturation[j].sum() / max(d.sum(), 1)),
            "dynamic_departures": int(dyn[j].sum()),
            "median_dynamic_departure_s": float(np.median(rc_events[j][dyn[j]])) if dyn[j].any() else float("nan"),
            f"brier_{int(h_last)}": float(dm.brier_score(_to_trials(x_trial, grid, _freq(x_trial, rc_events[j], h_last, grid)), o_last)),
            "ibs": float(member_ibs[j]),
        })
    _write_csv(out_dir / "per_member.csv", member_rows)
    _write_csv(out_dir / "per_level.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, default=float))
    _plots(out_dir, rows, x50_store, curves, s_bar, grid, x_trial, gt_events, times, horizon, horizons, x_label, title)
    return results


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plots(out, rows, x50_store, curves, s_bar, grid, x_trial, gt_events, times, horizon, horizons, x_label, title):
    apply_style()
    rc_c, gt_c = SERIES["rc"]["color"], SERIES["gt"]["color"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True, sharey=True)
    for ax, h in zip(axes.flat, horizons):
        r = [q for q in rows if q["horizon"] == h]
        x = np.array([q["x"] for q in r])
        n = np.array([q["n"] for q in r])
        gs = 1 - np.array([q["gt_event_freq"] for q in r])
        lo, hi = sv.jeffreys_interval(np.round(gs * n), n)
        ax.fill_between(x, 1 - np.array([q["ens_q95"] for q in r]), 1 - np.array([q["ens_q05"] for q in r]),
                        color=rc_c, alpha=0.22, lw=0, label="RC ensemble 5–95 %")
        ax.plot(x, 1 - np.array([q["ens_mean"] for q in r]), color=rc_c, lw=2, label="RC ensemble mean")
        ax.plot(x, 1 - np.array([q["member0"] for q in r]), color=rc_c, lw=1, ls="--", label="member 0 (original RC)")
        ax.errorbar(x, gs, yerr=[gs - lo, hi - gs], color=gt_c, marker="o", ms=3.5, lw=0, elinewidth=0.8, label="GT (95 % Jeffreys)")
        ax.set_title(f"survival to {h:.0f} s")
        ax.set_ylim(-0.05, 1.05)
    for ax in axes[-1]:
        ax.set_xlabel(x_label)
    for ax in axes[:, 0]:
        ax.set_ylabel("survival probability")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"{title}: survival vs {x_label}, seed ensemble vs GT", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "survival_vs_x_horizons.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    jit = np.random.default_rng(13)
    for i, h in enumerate(horizons):
        d = x50_store[h]
        lo, hi = np.nanquantile(d["gt_boot"], [0.025, 0.975])
        ax.plot([lo, hi], [i - 0.18] * 2, color=gt_c, lw=4, solid_capstyle="round", label="GT x50, 95 % bootstrap" if i == 0 else None)
        ax.plot(d["gt"], i - 0.18, "o", color=gt_c, ms=7)
        ax.scatter(d["members"], i + 0.18 + jit.uniform(-0.07, 0.07, len(d["members"])), s=20, color=rc_c, marker="s",
                   edgecolors="white", linewidths=0.5, label="RC members (identifiable)" if i == 0 else None)
        ax.plot(d["members"][0], i + 0.18, marker="D", color=TEXT_SECONDARY, ms=6, lw=0, label="member 0 (original RC)" if i == 0 else None)
    ax.set_yticks(range(len(horizons)), [f"{h:.0f} s" for h in horizons])
    ax.set_ylabel("horizon")
    ax.set_xlabel(f"x50: {x_label} where half of the trials have departed")
    ax.set_title(f"{title}: critical-point location per horizon")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "x50_distribution.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, h in zip(axes, (horizons[1], horizons[-1])):
        r = [q for q in rows if q["horizon"] == h]
        x = np.array([q["x"] for q in r])
        ax.plot(x, [q["aleatoric"] for q in r], color=gt_c, marker="o", ms=3, label="aleatoric  E[p(1−p)]")
        ax.plot(x, np.clip([q["epistemic_corrected"] for q in r], 0, None), color=rc_c, marker="s", ms=3, label="epistemic  Var_j(p), corrected")
        ax.set_title(f"horizon {h:.0f} s")
        ax.set_xlabel(x_label)
    axes[0].set_ylabel("variance of the departure outcome")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{title}: where does the ensemble's uncertainty come from?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "uncertainty_decomposition.png", dpi=200)
    plt.close(fig)

    gt_dep_levels = [v for v in grid if np.isfinite(gt_events[x_trial == v]).any()]
    picks = list(np.unique(np.round(np.linspace(0, len(gt_dep_levels) - 1, 6)).astype(int))) if gt_dep_levels else []
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, picks):
        v = gt_dep_levels[k]
        g = int(np.flatnonzero(grid == v)[0])
        lo, hi = np.quantile(curves[:, g], [0.05, 0.95], axis=0)
        ax.fill_between(times, lo, hi, color=rc_c, alpha=0.22, lw=0, step="post", label="RC members 5–95 %")
        ax.step(times, s_bar[g], where="post", color=rc_c, lw=2, label="RC ensemble mixture")
        km = sv.kaplan_meier(*sv.censor(gt_events[x_trial == v], horizon))
        ax.step(np.r_[km.time, horizon], np.r_[km.survival, km.survival[-1]], where="post", color=gt_c, lw=2, label="GT KM")
        ax.set_title(f"{x_label} = {v:.4f} (n = {np.sum(x_trial == v)})", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
    for ax in axes.flat[len(picks):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time since rollout start (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("S(t)")
    axes.flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"{title}: departure-time distribution, RC ensemble vs GT", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "km_ensemble.png", dpi=200)
    plt.close(fig)
