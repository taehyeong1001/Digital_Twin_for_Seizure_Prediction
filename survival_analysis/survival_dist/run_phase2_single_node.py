"""Phase 2: seed ensemble of the locked single-node SCDT-RC.

Cohorts (GT Epileptor, paired warmups, one RC rollout per GT trial and member):
  coarse : the original test grid (26 x0 x 10 seeds) -> member 0 = original result
  fine   : x0 in [-2.080, -2.020] step 0.002 (31 x0 x 20 seeds), covering the GT
           transition (~-2.0613) and the single RC's transition (~-2.043)
GT-only cohorts with independent seeds give the "oracle" forecast.

Because GT survival at a single horizon is a step in x0 (docs/03), every
binary metric is evaluated at several horizons h in HORIZONS: shorter h turns
the spread of departure times into genuinely probabilistic survival.

Run:   .venv/bin/python survival_dist/run_phase2_single_node.py [--members 20] [--workers 6] [--analyze-only]
Output: survival_dist/outputs/single_node/phase2/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dist_metrics as dm  # noqa: E402
import survival as sv  # noqa: E402
from ensemble import CohortSpec, load_members, load_or_build_cohort, train_ensemble_checkpointed  # noqa: E402
from plotting import SERIES, TEXT_SECONDARY, apply_style, plt  # noqa: E402
from single_node_setup import load_setup  # noqa: E402

OUT = HERE / "outputs" / "single_node" / "phase2"
HORIZONS = (60.0, 90.0, 120.0, 140.0)
X0_GT_STEP = -2.06133  # GT 140-s survival step located in the bifurcation probe (docs/03)
N_BOOT = 1000


def cohort_specs(setup) -> tuple[list[CohortSpec], list[CohortSpec]]:
    a = setup.args
    coarse_x0 = tuple(float(v) for v in setup.test_x0())
    fine_x0 = tuple(float(v) for v in np.round(np.linspace(-2.080, -2.020, 31), 4))
    rc_specs = [
        CohortSpec("coarse", coarse_x0, a.survival_seeds, a.test_seed_base, 0),
        CohortSpec("fine", fine_x0, 20, a.test_seed_base + 2_000_000, 500),
    ]
    oracle_specs = [
        CohortSpec("coarse_oracle", coarse_x0, a.survival_seeds, a.test_seed_base + 700_000, 0),
        CohortSpec("fine_oracle", fine_x0, 20, a.test_seed_base + 2_500_000, 0),
    ]
    return rc_specs, oracle_specs


# ---------------------------------------------------------------------------

def event_freq(x0_trial, events, h, grid):
    """Per-x0 event fraction (event at or before h) on `grid`."""
    out = np.empty(len(grid))
    for i, v in enumerate(grid):
        m = x0_trial == v
        out[i] = np.mean(np.isfinite(events[m]) & (events[m] <= h))
    return out


def to_trials(x0_trial, grid, values):
    return values[np.searchsorted(grid, x0_trial)]


def mixture_survival(member_events, x0_trial, grid, times, horizon):
    """S_bar(t|x0) = mean_j KM_j(t|x0); also each member's curves. Shapes (J, G, T)."""
    J = member_events.shape[0]
    curves = np.empty((J, len(grid), len(times)))
    for j in range(J):
        for g, v in enumerate(grid):
            m = x0_trial == v
            curves[j, g] = sv.kaplan_meier(*sv.censor(member_events[j, m], horizon)).at(times)
    return curves


def analyze(n_members: int) -> None:
    setup = load_setup()
    horizon = setup.horizon
    rc_specs, oracle_specs = cohort_specs(setup)
    cache = OUT / "cohorts"
    gt = {s.name: load_or_build_cohort(setup, s, cache) for s in rc_specs + oracle_specs}
    mem = {s.name: load_members(OUT, n_members, s.name) for s in rc_specs}

    # pooled trial arrays (coarse + fine)
    x0_trial = np.concatenate([gt[s.name].x0 for s in rc_specs])
    gt_events = np.concatenate([gt[s.name].gt_events for s in rc_specs])
    cohort_id = np.concatenate([np.full(len(gt[s.name].x0), s.name) for s in rc_specs])
    rc_events = np.hstack([mem[s.name]["events"] for s in rc_specs])  # (J, N)
    or_x0 = np.concatenate([gt[s.name].x0 for s in oracle_specs])
    or_events = np.concatenate([gt[s.name].gt_events for s in oracle_specs])
    grid = np.unique(x0_trial)

    health = {
        s.name: {
            "nonfinite_trials_per_member": mem[s.name]["nonfinite"].sum(axis=1).tolist(),
            "max_clip_fraction": float(mem[s.name]["clip_fraction"].max()),
        }
        for s in rc_specs
    }
    print("health:", {k: (sum(v["nonfinite_trials_per_member"]), v["max_clip_fraction"]) for k, v in health.items()}, flush=True)
    # Departure type: a "saturation" departure reaches the clip boundary at or
    # before the detected event (+0.5 s), i.e. the rollout ran away and was held
    # by np.clip; a "dynamic" departure crosses the event criterion first.
    first_clip = np.hstack([mem[s.name]["first_clip_s"] for s in rc_specs])
    departed = np.isfinite(rc_events)
    saturation = departed & np.isfinite(first_clip) & (first_clip <= rc_events + 0.5)
    departure_types = {
        "rc_departures_total": int(departed.sum()),
        "saturation_departures": int(saturation.sum()),
        "dynamic_departures": int((departed & ~saturation).sum()),
        "saturation_share": float(saturation.sum() / max(departed.sum(), 1)),
        "trials_ever_clipped_without_departure": int((~departed & np.isfinite(first_clip)).sum()),
    }
    print("departure types:", departure_types, flush=True)

    # member 0 must reproduce the original single RC on the coarse cohort
    p1 = json.loads((HERE / "outputs" / "single_node" / "phase1" / "summary.json").read_text())
    ph1 = np.load(HERE / "outputs" / "single_node" / "phase1" / "trial_events.npz")
    gate_member0 = bool(np.array_equal(np.isfinite(mem["coarse"]["events"][0]), np.isfinite(ph1["rc_events"]))
                        and np.allclose(mem["coarse"]["events"][0], ph1["rc_events"], equal_nan=True))
    print("gate member0 == phase-1 single RC:", gate_member0, flush=True)

    rng = np.random.default_rng(11)
    results = {
        "n_members": n_members,
        "gate_member0_equals_single_rc": gate_member0,
        "health": health,
        "departure_types": departure_types,
        "saturation_share_by_x0": {
            f"{v:.4f}": float(saturation[:, x0_trial == v].sum() / max(departed[:, x0_trial == v].sum(), 1))
            for v in np.unique(x0_trial) if departed[:, x0_trial == v].any()
        },
        "horizons": {},
    }
    per_x0_rows = []
    x50_store = {}
    for h in HORIZONS:
        o = (np.isfinite(gt_events) & (gt_events <= h)).astype(float)
        p_members = np.vstack([event_freq(x0_trial, rc_events[j], h, grid) for j in range(n_members)])  # (J, G)
        p_bar = p_members.mean(axis=0)
        p_oracle = event_freq(or_x0, or_events, h, grid)
        n_per_x0 = np.array([np.sum(x0_trial == v) for v in grid])
        forecasts = {
            "ensemble": to_trials(x0_trial, grid, p_bar),
            "member0": to_trials(x0_trial, grid, p_members[0]),
            "oracle": to_trials(x0_trial, grid, p_oracle),
            "climatology": dm.climatology(o),
        }
        binary = {}
        for name, p in forecasts.items():
            row = {"brier": dm.brier_score(p, o), "bss": dm.brier_skill_score(p, o, forecasts["climatology"])}
            for c in ("coarse", "fine"):
                m = cohort_id == c
                row[f"brier_{c}"] = dm.brier_score(p[m], o[m])
            row.update({f"murphy_{k}": v for k, v in dm.murphy_decomposition(p, o, n_bins=10).items()})
            binary[name] = row
        member_brier = [dm.brier_score(to_trials(x0_trial, grid, p_members[j]), o) for j in range(n_members)]
        decomp = dm.uncertainty_decomposition(p_members, replicates=None)
        # replicate-noise correction per x0 (trials per x0 differ between grids)
        within = np.mean(p_members * (1 - p_members), axis=0)
        epi_corr = decomp["epistemic"] - within / np.maximum(n_per_x0 - 1, 1)
        cov, band_lo, band_hi = dm.band_coverage(p_members, p_oracle)

        # dose-response x50(h): members vs GT bootstrap
        k_gt = np.array([np.sum(o[x0_trial == v]) for v in grid])
        fit_gt = dm.fit_logistic_survival(grid, k_gt, n_per_x0)
        boot_gt = dm.bootstrap_logistic(grid, k_gt, n_per_x0, N_BOOT, rng)
        member_fits = [dm.fit_logistic_survival(grid, np.round(p_members[j] * n_per_x0), n_per_x0) for j in range(n_members)]
        x50_members = np.array([f["x50"] for f in member_fits])
        s_members = np.array([f["s"] for f in member_fits])
        # A member whose departures never switch on/off inside the scanned
        # range has no identifiable x50 (the MLE runs off the grid): flag it.
        identifiable = np.isfinite(x50_members) & (x50_members >= grid.min()) & (x50_members <= grid.max()) & (s_members < np.ptp(grid))
        x50_valid = x50_members[identifiable]
        gt_lo, gt_hi = np.nanquantile(boot_gt[:, 0], [0.025, 0.975])
        x50_store[h] = {"members": np.where(identifiable, x50_members, np.nan), "gt": fit_gt["x50"], "gt_boot": boot_gt[:, 0]}
        results["horizons"][str(int(h))] = {
            "event_rate": float(o.mean()),
            "binary": binary,
            "member_brier_min_median_max": [float(np.min(member_brier)), float(np.median(member_brier)), float(np.max(member_brier))],
            "band_coverage_5_95": cov,
            "band_mean_width_5_95": float(np.mean(band_hi - band_lo)),
            "x50": {
                "gt": fit_gt["x50"], "gt_s": fit_gt["s"], "gt_boot_95": [float(gt_lo), float(gt_hi)],
                "gt_n_mixed_x0": int(np.count_nonzero((k_gt > 0) & (k_gt < n_per_x0))),
                "members_identifiable": int(identifiable.sum()),
                "members_median": float(np.median(x50_valid)) if x50_valid.size else float("nan"),
                "members_5_95": [float(np.quantile(x50_valid, 0.05)), float(np.quantile(x50_valid, 0.95))] if x50_valid.size else [float("nan")] * 2,
                "members_s_median": float(np.median(s_members[identifiable])) if x50_valid.size else float("nan"),
                "crps_identifiable_members_vs_gt": dm.crps_ensemble(x50_valid, fit_gt["x50"]),
                "member0": float(x50_members[0]) if identifiable[0] else float("nan"),
            },
            "uncertainty_mean_over_x0": {
                "total": float(decomp["total"].mean()),
                "aleatoric": float(decomp["aleatoric"].mean()),
                "epistemic": float(decomp["epistemic"].mean()),
                "epistemic_corrected": float(epi_corr.mean()),
            },
        }
        for g, v in enumerate(grid):
            per_x0_rows.append({
                "horizon": h, "x0": float(v), "n": int(n_per_x0[g]),
                "gt_event_freq": float(k_gt[g] / n_per_x0[g]), "oracle_event_freq": float(p_oracle[g]),
                "ens_mean": float(p_bar[g]), "ens_q05": float(band_lo[g]), "ens_q95": float(band_hi[g]),
                "member0": float(p_members[0, g]),
                "aleatoric": float(decomp["aleatoric"][g]), "epistemic": float(decomp["epistemic"][g]),
                "epistemic_corrected": float(epi_corr[g]),
            })

    # ---- time dimension: mixture survival curves ---------------------------
    times = np.linspace(1.0, horizon - 1.0, 139)
    curves = mixture_survival(rc_events, x0_trial, grid, times, horizon)  # (J, G, T)
    s_bar = curves.mean(axis=0)
    durations, observed = sv.censor(gt_events, horizon)
    gi = np.searchsorted(grid, x0_trial)
    oracle_curves = np.vstack([sv.kaplan_meier(*sv.censor(or_events[or_x0 == v], horizon)).at(times) for v in grid])
    pooled = sv.kaplan_meier(*sv.censor(or_events, horizon)).at(times)
    ibs = {
        "ensemble_mixture": dm.integrated_brier_score(durations, observed, s_bar[gi], times),
        "member0": dm.integrated_brier_score(durations, observed, curves[0][gi], times),
        "oracle": dm.integrated_brier_score(durations, observed, oracle_curves[gi], times),
        "pooled_km_reference": dm.integrated_brier_score(durations, observed, np.tile(pooled, (len(gi), 1)), times),
    }
    member_ibs = [dm.integrated_brier_score(durations, observed, curves[j][gi], times) for j in range(n_members)]
    ibs["member_min_median_max"] = [float(np.min(member_ibs)), float(np.median(member_ibs)), float(np.max(member_ibs))]
    # D-calibration of the mixture (continuous-ish in t: interpolate on the grid)
    s_at = np.array([np.interp(t, times, s_bar[g], left=1.0, right=s_bar[g, -1]) for t, g in zip(durations, gi)])
    dcal = dm.d_calibration(s_at, s_at, observed, rng=np.random.default_rng(12))
    results["time"] = {"ibs": ibs, "d_calibration_mixture": {"chi2": dcal["chi2"], "p_value": dcal["p_value"], "mass": dcal["mass"].tolist()}}

    gt_dep = np.isfinite(gt_events)
    member_rows = []
    for j in range(n_members):
        d = departed[j]
        below = x0_trial < X0_GT_STEP
        p140 = to_trials(x0_trial, grid, event_freq(x0_trial, rc_events[j], horizon, grid))
        dyn = d & ~saturation[j]
        member_rows.append({
            "member": j,
            "false_departure_rate_below_gt_step": float(d[below].mean()),
            "miss_rate_above_gt_step": float((~d[~below]).mean()),
            "departure_rate_in_training_range": float(d[x0_trial <= -2.15].mean()),
            "saturation_share": float(saturation[j].sum() / max(d.sum(), 1)),
            "dynamic_departures": int(dyn.sum()),
            "median_dynamic_departure_s": float(np.median(rc_events[j][dyn])) if dyn.any() else float("nan"),
            "brier_140": dm.brier_score(p140, gt_dep.astype(float)),
            "ibs": float(member_ibs[j]),
        })
    with (OUT / "per_member.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(member_rows[0].keys()))
        writer.writeheader()
        writer.writerows(member_rows)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2, default=float))
    with (OUT / "per_x0.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_x0_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_x0_rows)
    print(json.dumps(results, indent=2, default=float))
    plot_all(per_x0_rows, x50_store, curves, s_bar, grid, x0_trial, gt_events, times, horizon)


def plot_all(rows, x50_store, curves, s_bar, grid, x0_trial, gt_events, times, horizon) -> None:
    apply_style()
    rc_c, gt_c, or_c = SERIES["rc"]["color"], SERIES["gt"]["color"], SERIES["oracle"]["color"]

    # (1) survival vs x0 at each horizon, ensemble band
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True, sharey=True)
    for ax, h in zip(axes.flat, HORIZONS):
        r = [x for x in rows if x["horizon"] == h]
        x = np.array([q["x0"] for q in r])
        n = np.array([q["n"] for q in r])
        gt_s = 1 - np.array([q["gt_event_freq"] for q in r])
        lo, hi = sv.jeffreys_interval(np.round(gt_s * n), n)
        ax.fill_between(x, 1 - np.array([q["ens_q95"] for q in r]), 1 - np.array([q["ens_q05"] for q in r]),
                        color=rc_c, alpha=0.22, lw=0, label="RC ensemble 5–95 %")
        ax.plot(x, 1 - np.array([q["ens_mean"] for q in r]), color=rc_c, lw=2, label="RC ensemble mean")
        ax.plot(x, 1 - np.array([q["member0"] for q in r]), color=rc_c, lw=1, ls="--", label="member 0 (original RC)")
        ax.errorbar(x, gt_s, yerr=[gt_s - lo, hi - gt_s], color=gt_c, marker="o", ms=3.5, lw=0, elinewidth=0.8,
                    label="GT (95 % Jeffreys)")
        ax.set_title(f"survival to {h:.0f} s")
        ax.set_xlim(-2.20, -1.95)
        ax.set_ylim(-0.05, 1.05)
    for ax in axes[-1]:
        ax.set_xlabel("target x0")
    for ax in axes[:, 0]:
        ax.set_ylabel("survival probability")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Survival vs x0: seed-ensemble distribution vs GT (x-range zoomed to the transition)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "survival_vs_x0_horizons.png", dpi=200)
    plt.close(fig)

    # (2) x50(h) distributions
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    rng = np.random.default_rng(13)
    for i, h in enumerate(HORIZONS):
        d = x50_store[h]
        lo, hi = np.nanquantile(d["gt_boot"], [0.025, 0.975])
        ax.plot([lo, hi], [i - 0.18, i - 0.18], color=gt_c, lw=4, solid_capstyle="round",
                label="GT x50, 95 % bootstrap" if i == 0 else None)
        ax.plot(d["gt"], i - 0.18, "o", color=gt_c, ms=7)
        ax.scatter(d["members"], i + 0.18 + rng.uniform(-0.07, 0.07, len(d["members"])), s=20, color=rc_c,
                   marker="s", edgecolors="white", linewidths=0.5, label="RC members" if i == 0 else None)
        ax.plot(d["members"][0], i + 0.18, marker="D", color=TEXT_SECONDARY, ms=6, lw=0,
                label="member 0 (original RC)" if i == 0 else None)
    ax.set_yticks(range(len(HORIZONS)), [f"{h:.0f} s" for h in HORIZONS])
    ax.set_ylabel("horizon")
    ax.set_xlabel("x50: x0 where half of the trials have departed")
    ax.set_title("Critical-point location per horizon: GT vs RC seed ensemble")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "x50_distribution.png", dpi=200)
    plt.close(fig)

    # (3) uncertainty decomposition vs x0 at two horizons
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, h in zip(axes, (90.0, 140.0)):
        r = [x for x in rows if x["horizon"] == h]
        x = np.array([q["x0"] for q in r])
        ax.plot(x, [q["aleatoric"] for q in r], color=SERIES["gt"]["color"], marker="o", ms=3, label="aleatoric  E[p(1−p)]")
        ax.plot(x, np.clip([q["epistemic_corrected"] for q in r], 0, None), color=rc_c, marker="s", ms=3,
                label="epistemic  Var_j(p), corrected")
        ax.set_title(f"horizon {h:.0f} s")
        ax.set_xlim(-2.20, -1.95)
        ax.set_xlabel("target x0")
    axes[0].set_ylabel("variance of the departure outcome")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Where does the RC ensemble's uncertainty come from?", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "uncertainty_decomposition.png", dpi=200)
    plt.close(fig)

    # (4) KM: ensemble band vs GT at selected x0
    picks = [v for v in (-2.060, -2.044, -2.030, -2.020, -2.0005, -1.9567) if np.any(np.isclose(grid, v))]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    for ax, v in zip(axes.flat, picks):
        g = int(np.flatnonzero(np.isclose(grid, v))[0])
        lo, hi = np.quantile(curves[:, g], [0.05, 0.95], axis=0)
        ax.fill_between(times, lo, hi, color=rc_c, alpha=0.22, lw=0, step="post", label="RC members 5–95 %")
        ax.step(times, s_bar[g], where="post", color=rc_c, lw=2, label="RC ensemble mixture")
        km = sv.kaplan_meier(*sv.censor(gt_events[x0_trial == grid[g]], horizon))
        ax.step(np.r_[km.time, horizon], np.r_[km.survival, km.survival[-1]], where="post", color=gt_c, lw=2, label="GT KM")
        ax.set_title(f"x0 = {grid[g]:.4f} (n = {np.sum(x0_trial == grid[g])})", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
    for ax in axes[-1]:
        ax.set_xlabel("time since rollout start (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("S(t)")
    axes.flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Departure-time distribution: RC ensemble vs GT", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "km_ensemble.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        setup = load_setup()
        rc_specs, oracle_specs = cohort_specs(setup)
        for spec in oracle_specs:
            load_or_build_cohort(setup, spec, OUT / "cohorts")
        train_ensemble_checkpointed(args.members, rc_specs, OUT, workers=args.workers)
    analyze(args.members)


if __name__ == "__main__":
    main()
