"""Self-test for survival.py and dist_metrics.py.

Checks against an independent implementation (scikit-survival) where one
exists, otherwise against a known synthetic ground truth or an exact identity.
Run:  .venv/bin/python survival_dist/selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dist_metrics as dm  # noqa: E402
import survival as sv  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def synthetic_survival(rng: np.random.Generator, n: int, rate: float, horizon: float):
    t = rng.exponential(1.0 / rate, size=n)
    t[t > horizon] = np.nan
    return t


def test_km_against_sksurv(rng: np.random.Generator) -> None:
    from sksurv.nonparametric import kaplan_meier_estimator

    times = synthetic_survival(rng, 300, 0.01, 120.0)
    # random censoring in addition to the horizon
    durations, observed = sv.censor(times, 120.0)
    random_c = rng.uniform(20.0, 200.0, size=len(durations))
    early = random_c < durations
    durations = np.where(early, random_c, durations)
    observed = observed & ~early
    km = sv.kaplan_meier(durations, observed)
    t_ref, s_ref, (lo_ref, hi_ref) = kaplan_meier_estimator(
        observed, durations, conf_type="log-log"
    )
    s_ours = km.at(t_ref)
    check("KM matches sksurv", np.allclose(s_ours, s_ref), f"max|d|={np.max(np.abs(s_ours - s_ref)):.2e}")
    # sksurv evaluates the band at every unique time; compare at event times.
    mask = np.isin(t_ref, km.time[1:]) & (s_ref > 0) & (s_ref < 1)
    idx = np.searchsorted(km.time, t_ref[mask])
    check(
        "Greenwood log-log band matches sksurv",
        np.allclose(km.lower[idx], lo_ref[mask], atol=1e-8) and np.allclose(km.upper[idx], hi_ref[mask], atol=1e-8),
        f"max|d|={max(np.max(np.abs(km.lower[idx] - lo_ref[mask])), np.max(np.abs(km.upper[idx] - hi_ref[mask]))):.2e}",
    )


def test_horizon_equals_grouped_survival(rng: np.random.Generator) -> None:
    times = synthetic_survival(rng, 50, 0.02, 140.0)
    original = float(np.mean(~np.isfinite(times)))  # final_code grouped_survival
    durations, observed = sv.censor(times, 140.0)
    km = sv.kaplan_meier(durations, observed)
    check(
        "S_KM(H) == survival_at_horizon == original grouped_survival",
        np.isclose(km.at(140.0), original) and np.isclose(sv.survival_at_horizon(times, 140.0), original),
    )


def test_jeffreys() -> None:
    lo, hi = sv.jeffreys_interval(np.array([0, 3, 10]), np.array([10, 10, 10]))
    ok = lo[0] == 0.0 and hi[2] == 1.0 and np.isclose(lo[1], beta.ppf(0.025, 3.5, 7.5))
    check("Jeffreys interval endpoints / quantiles", ok)
    # coverage close to nominal for moderate n
    rng = np.random.default_rng(5)
    p, n = 0.3, 40
    k = rng.binomial(n, p, size=20000)
    lo, hi = sv.jeffreys_interval(k, np.full_like(k, n))
    cov = np.mean((lo <= p) & (p <= hi))
    check("Jeffreys coverage ~ 95 %", 0.92 < cov < 0.98, f"coverage={cov:.3f}")


def test_escape_rate(rng: np.random.Generator) -> None:
    rate = 0.015
    times = synthetic_survival(rng, 4000, rate, 140.0)
    kappa, lo, hi = sv.exponential_escape_rate(*sv.censor(times, 140.0))
    check("exponential escape-rate MLE recovers kappa", lo < rate < hi, f"kappa={kappa:.4f} [{lo:.4f},{hi:.4f}]")


def test_brier_family(rng: np.random.Generator) -> None:
    p = rng.choice([0.1, 0.3, 0.6, 0.9], size=1000)
    o = (rng.random(1000) < p).astype(float)
    d = dm.murphy_decomposition(p, o)
    check("Murphy identity exact on distinct forecasts", abs(d["residual"]) < 1e-12, f"residual={d['residual']:.1e}")
    check("BSS(climatology) == 0", np.isclose(dm.brier_skill_score(dm.climatology(o), o, dm.climatology(o)), 0.0))
    rc = dm.reliability_curve(p, o, n_bins=10, n_resample=500, rng=rng)
    filled = rc["count"] > 0
    inside = (rc["observed"][filled] >= rc["consistency_lower"][filled] - 1e-12) & (
        rc["observed"][filled] <= rc["consistency_upper"][filled] + 1e-12
    )
    check("reliable forecast stays inside consistency bars", inside.mean() >= 0.75, f"{inside.sum()}/{filled.sum()} bins")


def test_ibs_against_sksurv(rng: np.random.Generator) -> None:
    from sksurv.metrics import brier_score as sk_brier, integrated_brier_score as sk_ibs
    from sksurv.util import Surv

    horizon = 120.0
    rate = 0.012
    t_train = synthetic_survival(rng, 400, rate, horizon)
    t_test = synthetic_survival(rng, 300, rate, horizon)
    random_c = rng.uniform(10.0, 300.0, size=len(t_test))
    d_te, e_te = sv.censor(t_test, horizon)
    early = random_c < d_te
    d_te, e_te = np.where(early, random_c, d_te), e_te & ~early
    d_tr, e_tr = sv.censor(t_train, horizon)
    times = np.linspace(5.0, min(d_te.max(), d_tr.max()) - 1.0, 40)
    pred = np.exp(-rate * 1.1 * times)[None, :].repeat(len(d_te), axis=0)
    y_tr = Surv.from_arrays(e_tr, d_tr)
    y_te = Surv.from_arrays(e_te, d_te)
    # sksurv estimates G from the *training* set; feed ours the same data by
    # comparing on train == test.
    _, bs_ref = sk_brier(y_te, y_te, pred, times)
    bs_ours = dm.brier_score_over_time(d_te, e_te, pred, times)
    check("BS(t) matches sksurv (G from same set)", np.allclose(bs_ours, bs_ref, atol=1e-10), f"max|d|={np.max(np.abs(bs_ours - bs_ref)):.1e}")
    ibs_ref = sk_ibs(y_te, y_te, pred, times)
    ibs_ours = dm.integrated_brier_score(d_te, e_te, pred, times)
    check("IBS matches sksurv", np.isclose(ibs_ours, ibs_ref, atol=1e-10), f"{ibs_ours:.6f} vs {ibs_ref:.6f}")


def test_d_calibration(rng: np.random.Generator) -> None:
    horizon, rate, n = 140.0, 0.01, 3000
    times = synthetic_survival(rng, n, rate, horizon)
    durations, observed = sv.censor(times, horizon)

    def run(pred_rate: float) -> float:
        s = np.exp(-pred_rate * durations)
        return dm.d_calibration(s, s, observed, rng=rng)["p_value"]

    p_good, p_bad = run(rate), run(rate * 2.0)
    check("D-calibration accepts the true model", p_good > 0.01, f"p={p_good:.3f}")
    check("D-calibration rejects a 2x-hazard model", p_bad < 1e-3, f"p={p_bad:.1e}")


def test_logistic(rng: np.random.Generator) -> None:
    x = np.linspace(-2.3, -1.9, 26)
    true_x50, true_s = -2.08, 0.02
    n = np.full_like(x, 200)
    k = rng.binomial(n.astype(int), 1.0 - dm.logistic_survival(x, true_x50, true_s))
    fit = dm.fit_logistic_survival(x, k, n)
    ok = abs(fit["x50"] - true_x50) < 0.005 and abs(fit["s"] - true_s) < 0.006
    check("logistic dose-response fit recovers (x50, s)", ok, f"x50={fit['x50']:.4f} s={fit['s']:.4f}")
    boot = dm.bootstrap_logistic(x, k, n, 200, rng)
    lo, hi = np.nanquantile(boot[:, 0], [0.025, 0.975])
    check("bootstrap x50 interval covers the truth", lo < true_x50 < hi, f"[{lo:.4f},{hi:.4f}]")


def test_crps_and_decomposition(rng: np.random.Generator) -> None:
    x = rng.normal(size=37)
    y = 0.3
    brute = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    check("CRPS sorted identity == brute force", np.isclose(dm.crps_ensemble(x, y), brute))
    p = rng.beta(2, 5, size=(30, 12))
    d = dm.uncertainty_decomposition(p, replicates=20)
    check("total = aleatoric + epistemic (exact)", np.allclose(d["total"], d["aleatoric"] + d["epistemic"]))
    # replicate-noise correction: identical members observed through R draws
    true_p = np.full((200, 5), 0.4)
    R = 25
    p_hat = rng.binomial(R, true_p) / R
    d = dm.uncertainty_decomposition(p_hat, replicates=R)
    check(
        "replicate-noise correction removes spurious epistemic variance",
        np.all(np.abs(d["epistemic_corrected"]) < 0.004) and np.all(d["epistemic"] > 0.005),
        f"raw={d['epistemic'].mean():.4f} corrected={d['epistemic_corrected'].mean():.4f}",
    )


def main() -> None:
    rng = np.random.default_rng(20261012)
    test_km_against_sksurv(rng)
    test_horizon_equals_grouped_survival(rng)
    test_jeffreys()
    test_escape_rate(rng)
    test_brier_family(rng)
    test_ibs_against_sksurv(rng)
    test_d_calibration(rng)
    test_logistic(rng)
    test_crps_and_decomposition(rng)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    out = Path(__file__).resolve().parent / "outputs" / "selftest_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"{'PASS' if ok else 'FAIL'}\t{n}\t{d}" for n, ok, d in RESULTS) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
