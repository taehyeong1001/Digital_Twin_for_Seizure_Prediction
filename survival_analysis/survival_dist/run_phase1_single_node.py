"""Phase 1: distributional metrics for the existing single-node SCDT-RC.

1. Rebuild the locked single RC from the original run's protocol and
   re-score the original test cohort -> trial-level GT / RC event times.
   Gate: the per-x0 survival must equal the original `survival_rate.csv`,
   and KM S(H) must equal the original `grouped_survival`.
2. Simulate an independent GT cohort (new seeds, same x0 grid): its per-x0
   event frequency is the "oracle" forecast (best achievable from finite GT).
3. Score the single RC with the new metrics (docs/02_metrics.md) and plot.

Outputs: survival_dist/outputs/single_node/phase1/
Run:     .venv/bin/python survival_dist/run_phase1_single_node.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dist_metrics as dm  # noqa: E402
import survival as sv  # noqa: E402
from plotting import SERIES, TEXT_SECONDARY, apply_style, plt  # noqa: E402
from single_node_setup import DEFAULT_PROTOCOL, load_setup  # noqa: E402

OUT = HERE / "outputs" / "single_node" / "phase1"
ORACLE_SEED_OFFSET = 700000  # independent of every seed base used by the original run
N_BOOT = 1000


def per_x0_frequency(x0_trial: np.ndarray, events: np.ndarray, horizon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sv.grouped_event_counts(x0_trial, events, horizon)


def forecast_for_trials(x0_trial: np.ndarray, x0_grid: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
    return p_grid[np.searchsorted(x0_grid, x0_trial)]


def km_by_x0(x0_trial: np.ndarray, events: np.ndarray, horizon: float) -> dict[float, sv.KMCurve]:
    out = {}
    for value in np.unique(x0_trial):
        mask = x0_trial == value
        out[float(value)] = sv.kaplan_meier(*sv.censor(events[mask], horizon))
    return out


def survival_matrix(curves: dict[float, sv.KMCurve], x0_trial: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.vstack([curves[float(v)].at(times) for v in x0_trial])


def d_calibration_inputs(curves, x0_trial, durations, observed):
    s_at = np.array([curves[float(v)].at(t) for v, t in zip(x0_trial, durations)], dtype=float)
    s_before = np.array([curves[float(v)].at(np.nextafter(t, -np.inf)) for v, t in zip(x0_trial, durations)], dtype=float)
    return s_at, s_before


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup = load_setup()
    args = setup.args
    horizon = setup.horizon
    x0_grid = setup.test_x0()

    # ---- 1. reproduce the original test result at trial level -------------
    rc = setup.fit_member()  # locked parameters, original reservoir seed
    test = setup.cohort(x0_grid, args.survival_seeds, args.test_seed_base)
    rc_events = setup.rc_events(rc, test, args.seed + 90000)
    gt_events = np.asarray(test.gt_events, dtype=float)
    x0_trial = np.asarray(test.x0, dtype=float)

    original_csv = DEFAULT_PROTOCOL.parent / "survival_rate.csv"
    with original_csv.open() as handle:
        original = list(csv.DictReader(handle))
    orig_gt = np.array([float(r["gt_survival"]) for r in original])
    orig_rc = np.array([float(r["rc_survival"]) for r in original])
    _, k_gt, n_gt = per_x0_frequency(x0_trial, gt_events, horizon)
    _, k_rc, n_rc = per_x0_frequency(x0_trial, rc_events, horizon)
    km_gt = km_by_x0(x0_trial, gt_events, horizon)
    km_rc = km_by_x0(x0_trial, rc_events, horizon)
    gate = {
        "gt_matches_original_csv": bool(np.array_equal(1 - k_gt / n_gt, orig_gt)),
        "rc_matches_original_csv": bool(np.array_equal(1 - k_rc / n_rc, orig_rc)),
        "km_at_horizon_equals_grouped_survival": bool(
            np.allclose([km_gt[float(v)].at(horizon) for v in x0_grid], orig_gt)
            and np.allclose([km_rc[float(v)].at(horizon) for v in x0_grid], orig_rc)
        ),
    }
    print("gate:", gate, flush=True)

    # ---- 2. independent GT cohort -> oracle forecast -----------------------
    oracle = setup.cohort(x0_grid, args.survival_seeds, args.test_seed_base + ORACLE_SEED_OFFSET)
    oracle_events = np.asarray(oracle.gt_events, dtype=float)
    _, k_or, n_or = per_x0_frequency(np.asarray(oracle.x0), oracle_events, horizon)
    km_or = km_by_x0(np.asarray(oracle.x0), oracle_events, horizon)

    np.savez_compressed(
        OUT / "trial_events.npz",
        x0_grid=x0_grid,
        x0_trial=x0_trial,
        gt_events=gt_events,
        rc_events=rc_events,
        oracle_x0_trial=np.asarray(oracle.x0),
        oracle_events=oracle_events,
        horizon=horizon,
    )

    # ---- 3a. binary event within the horizon (Brier family) ---------------
    _, observed = sv.censor(gt_events, horizon)
    o = observed.astype(float)
    p_rc = forecast_for_trials(x0_trial, x0_grid, k_rc / n_rc)
    p_or = forecast_for_trials(x0_trial, x0_grid, k_or / n_or)
    p_clim = dm.climatology(o)
    binary = {}
    for name, p in [("rc", p_rc), ("oracle", p_or), ("climatology", p_clim)]:
        binary[name] = {
            "brier": dm.brier_score(p, o),
            "bss_vs_climatology": dm.brier_skill_score(p, o, p_clim),
            **{f"murphy_{k}": v for k, v in dm.murphy_decomposition(p, o).items()},
        }
    reliability = dm.reliability_curve(p_rc, o, n_bins=10, rng=np.random.default_rng(1))

    # ---- 3b. dose-response summary ----------------------------------------
    rng = np.random.default_rng(2)
    dose = {}
    for name, k, n in [("gt", k_gt, n_gt), ("rc", k_rc, n_rc), ("oracle", k_or, n_or)]:
        fit = dm.fit_logistic_survival(x0_grid, k, n)
        boot = dm.bootstrap_logistic(x0_grid, k, n, N_BOOT, rng)
        lo, hi = np.nanquantile(boot[:, 0], [0.025, 0.975])
        mixed = int(np.count_nonzero((k > 0) & (k < n)))
        dose[name] = {**fit, "x50_boot_lo": float(lo), "x50_boot_hi": float(hi), "n_mixed_x0": mixed}

    # ---- 3c. time dimension ------------------------------------------------
    durations, observed_t = sv.censor(gt_events, horizon)
    times = np.linspace(1.0, horizon - 1.0, 139)
    pooled = sv.kaplan_meier(*sv.censor(oracle_events, horizon))  # covariate-free reference
    ibs = {
        "rc": dm.integrated_brier_score(durations, observed_t, survival_matrix(km_rc, x0_trial, times), times),
        "oracle": dm.integrated_brier_score(durations, observed_t, survival_matrix(km_or, x0_trial, times), times),
        "pooled_km_reference": dm.integrated_brier_score(
            durations, observed_t, np.tile(pooled.at(times), (len(durations), 1)), times
        ),
    }
    dcal = {}
    for name, curves in [("rc", km_rc), ("oracle", km_or)]:
        s_at, s_before = d_calibration_inputs(curves, x0_trial, durations, observed_t)
        res = dm.d_calibration(s_at, s_before, observed_t, rng=np.random.default_rng(3))
        dcal[name] = {"chi2": res["chi2"], "p_value": res["p_value"], "mass": res["mass"].tolist()}

    per_x0 = []
    lo_gt, hi_gt = sv.jeffreys_interval(n_gt - k_gt, n_gt)
    lo_rc, hi_rc = sv.jeffreys_interval(n_rc - k_rc, n_rc)
    for i, value in enumerate(x0_grid):
        mask = x0_trial == value
        kap_gt = sv.exponential_escape_rate(*sv.censor(gt_events[mask], horizon))
        kap_rc = sv.exponential_escape_rate(*sv.censor(rc_events[mask], horizon))
        per_x0.append(
            {
                "x0": float(value),
                "n": int(n_gt[i]),
                "gt_survival": float(1 - k_gt[i] / n_gt[i]),
                "gt_jeffreys_lo": float(lo_gt[i]),
                "gt_jeffreys_hi": float(hi_gt[i]),
                "rc_survival": float(1 - k_rc[i] / n_rc[i]),
                "rc_jeffreys_lo": float(lo_rc[i]),
                "rc_jeffreys_hi": float(hi_rc[i]),
                "oracle_survival": float(1 - k_or[i] / n_or[i]),
                "gt_median_event_s": float(np.nanmedian(gt_events[mask])) if np.isfinite(gt_events[mask]).any() else float("nan"),
                "rc_median_event_s": float(np.nanmedian(rc_events[mask])) if np.isfinite(rc_events[mask]).any() else float("nan"),
                "gt_kappa_per_s": kap_gt[0],
                "rc_kappa_per_s": kap_rc[0],
            }
        )
    with (OUT / "per_x0.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_x0[0].keys()))
        writer.writeheader()
        writer.writerows(per_x0)

    summary = {
        "gate": gate,
        "horizon_s": horizon,
        "n_trials": int(len(o)),
        "n_x0": int(len(x0_grid)),
        "trials_per_x0": int(args.survival_seeds),
        "event_rate": float(o.mean()),
        "binary": binary,
        "dose_response": dose,
        "ibs": ibs,
        "d_calibration": dcal,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))

    plot_all(x0_grid, per_x0, reliability, km_gt, km_rc, x0_trial, gt_events, rc_events, horizon, dose)


def plot_all(x0_grid, per_x0, reliability, km_gt, km_rc, x0_trial, gt_events, rc_events, horizon, dose) -> None:
    apply_style()
    rows = {k: np.array([r[k] for r in per_x0]) for k in per_x0[0]}

    # (a) survival vs x0 with Jeffreys intervals
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    offset = 0.0018
    for key, shift, lo, hi in [("gt", -offset, "gt_jeffreys_lo", "gt_jeffreys_hi"), ("rc", offset, "rc_jeffreys_lo", "rc_jeffreys_hi")]:
        style = SERIES[key]
        y = rows[f"{key}_survival"]
        ax.errorbar(
            x0_grid + shift, y, yerr=[y - rows[lo], rows[hi] - y], color=style["color"], marker=style["marker"],
            lw=1.4, capsize=0, elinewidth=1.0, label=f"{style['label']} (95 % Jeffreys)",
        )
    ax.plot(x0_grid, rows["oracle_survival"], color=SERIES["oracle"]["color"], marker=SERIES["oracle"]["marker"],
            lw=0, label=SERIES["oracle"]["label"])
    ax.set_xlabel("target x0")
    ax.set_ylabel(f"survival to {horizon:.0f} s")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Survival per x0 with sampling uncertainty (10 trials per x0)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "survival_vs_x0.png", dpi=200)
    plt.close(fig)

    # (b) reliability diagram for the single RC
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    filled = reliability["count"] > 0
    centers = reliability["mean_forecast"]  # bars sit at each bin's mean forecast
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, lw=1.0, ls="--", label="perfect reliability")
    ax.vlines(centers[filled], reliability["consistency_lower"][filled], reliability["consistency_upper"][filled],
              color=TEXT_SECONDARY, lw=6, alpha=0.25, label="90 % consistency bar (if reliable)")
    ax.plot(reliability["mean_forecast"][filled], reliability["observed"][filled], color=SERIES["rc"]["color"],
            marker=SERIES["rc"]["marker"], lw=1.4, label=SERIES["rc"]["label"])
    for c, f, n in zip(reliability["mean_forecast"][filled], reliability["observed"][filled], reliability["count"][filled]):
        ax.annotate(f"n={n}", (c, f), textcoords="offset points", xytext=(6, -10), fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("forecast departure probability")
    ax.set_ylabel("observed GT departure frequency")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Reliability of the single RC")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "reliability_rc.png", dpi=200)
    plt.close(fig)

    # (c) KM curves where either GT or RC has events
    active = [v for v in x0_grid if km_gt[float(v)].survival[-1] < 1 or km_rc[float(v)].survival[-1] < 1]
    ncol = 4
    nrow = int(np.ceil(len(active) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, value in zip(axes.flat, active):
        for key, curves in [("gt", km_gt), ("rc", km_rc)]:
            c = curves[float(value)]
            t = np.r_[c.time, horizon]
            s = np.r_[c.survival, c.survival[-1]]
            ax.step(t, s, where="post", color=SERIES[key]["color"], lw=1.8, label=SERIES[key]["label"])
        ax.set_title(f"x0 = {value:.4f}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
    for ax in axes.flat[len(active):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time since rollout start (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("S(t)")
    axes.flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Kaplan-Meier survival per x0 (10 trials each)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "km_per_x0.png", dpi=200)
    plt.close(fig)

    # (d) event-time strip plot
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    rng = np.random.default_rng(4)
    for key, events, shift in [("gt", gt_events, -0.0025), ("rc", rc_events, 0.0025)]:
        mask = np.isfinite(events)
        jitter = rng.uniform(-0.0012, 0.0012, mask.sum())
        ax.scatter(x0_trial[mask] + shift + jitter, events[mask], s=22, color=SERIES[key]["color"],
                   marker=SERIES[key]["marker"], edgecolors="white", linewidths=0.6, label=SERIES[key]["label"])
    ax.set_xlabel("target x0")
    ax.set_ylabel("departure time (s since rollout start)")
    ax.set_ylim(0, horizon)
    ax.set_title("Departure times of individual trials (no marker = survived)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "event_times.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
