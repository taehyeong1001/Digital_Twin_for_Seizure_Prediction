"""Scores for probabilistic predictions of normal departure.

Each function is documented with its definition; the full rationale and the
literature are in `docs/02_metrics.md`.

Conventions
-----------
- `p` is a predicted *event* (departure) probability within the horizon,
  `o` the observed outcome (1 = departed, 0 = survived). Survival = 1 - p.
- Survival-curve predictions are arrays `S[i, j] = S_i(t_j)` on a time grid.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

from survival import kaplan_meier


# ---------------------------------------------------------------------------
# 1. Binary-event probability scores (Brier family)
# ---------------------------------------------------------------------------

def brier_score(p: np.ndarray, o: np.ndarray) -> float:
    """BS = mean (p_i - o_i)^2. Strictly proper; 0 is perfect."""
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    return float(np.mean((p - o) ** 2))


def brier_skill_score(p: np.ndarray, o: np.ndarray, p_reference: np.ndarray) -> float:
    """BSS = 1 - BS(p) / BS(reference). 1 perfect, 0 no gain, < 0 worse."""
    reference = brier_score(p_reference, o)
    return float(1.0 - brier_score(p, o) / reference) if reference > 0 else float("nan")


def climatology(o: np.ndarray) -> np.ndarray:
    """Constant forecast equal to the overall event rate (reference forecast)."""
    o = np.asarray(o, dtype=float)
    return np.full_like(o, o.mean())


def murphy_decomposition(
    p: np.ndarray, o: np.ndarray, n_bins: int | None = None
) -> dict[str, float]:
    """Murphy (1973): BS = REL - RES + UNC.

    REL = sum_k n_k (pbar_k - obar_k)^2 / N   (miscalibration, lower is better)
    RES = sum_k n_k (obar_k - obar)^2 / N     (discrimination, higher is better)
    UNC = obar (1 - obar)                      (event uncertainty, data only)

    With `n_bins=None` forecasts are grouped by their distinct values and the
    identity is exact. With bins, the within-bin forecast variance makes the
    identity approximate; the leftover is returned as `residual`.
    """
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    if n_bins is None:
        _, labels = np.unique(p, return_inverse=True)
    else:
        labels = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    n = len(p)
    obar = o.mean()
    rel = res = 0.0
    for label in np.unique(labels):
        mask = labels == label
        nk = mask.sum()
        rel += nk * (p[mask].mean() - o[mask].mean()) ** 2
        res += nk * (o[mask].mean() - obar) ** 2
    rel /= n
    res /= n
    unc = obar * (1.0 - obar)
    bs = brier_score(p, o)
    return {"bs": bs, "rel": rel, "res": res, "unc": unc, "residual": bs - (rel - res + unc)}


def reliability_curve(
    p: np.ndarray,
    o: np.ndarray,
    n_bins: int = 10,
    n_resample: int = 2000,
    level: float = 0.90,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Reliability diagram data with Broecker & Smith (2007) consistency bars.

    For each equal-width bin: mean forecast, observed frequency, count.
    Consistency bars: resample forecasts (with replacement) from the observed
    forecast set, draw outcomes o* ~ Bernoulli(p*) (i.e. assume the forecast
    IS reliable), and take quantiles of the per-bin observed frequency. A bin
    whose observed frequency falls outside its bar is evidence of miscalibration
    beyond sampling noise.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    labels = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    count = np.bincount(labels, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_forecast = np.bincount(labels, weights=p, minlength=n_bins) / count
        observed = np.bincount(labels, weights=o, minlength=n_bins) / count
    resampled = np.full((n_resample, n_bins), np.nan)
    for r in range(n_resample):
        index = rng.integers(0, len(p), size=len(p))
        p_star = p[index]
        o_star = rng.random(len(p)) < p_star
        lab = labels[index]
        c = np.bincount(lab, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            resampled[r] = np.bincount(lab, weights=o_star, minlength=n_bins) / c
    tail = (1.0 - level) / 2.0
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty bins -> NaN
        lower = np.nanquantile(resampled, tail, axis=0)
        upper = np.nanquantile(resampled, 1.0 - tail, axis=0)
    return {
        "bin_edges": np.linspace(0.0, 1.0, n_bins + 1),
        "mean_forecast": mean_forecast,
        "observed": observed,
        "count": count,
        "consistency_lower": lower,
        "consistency_upper": upper,
    }


# ---------------------------------------------------------------------------
# 2. Survival-curve scores (time dimension)
# ---------------------------------------------------------------------------

def _censoring_survival(durations: np.ndarray, observed: np.ndarray):
    """KM estimate G(t) of the censoring distribution (censoring = event)."""
    return kaplan_meier(durations, ~np.asarray(observed, dtype=bool))


def brier_score_over_time(
    durations: np.ndarray,
    observed: np.ndarray,
    survival_pred: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Graf et al. (1999) censoring-weighted Brier score BS(t) at each time.

    BS(t) = mean_i [ S_i(t)^2 * 1(T_i <= t, event) / G(T_i)
                    + (1 - S_i(t))^2 * 1(T_i > t) / G(t) ]
    G = Kaplan-Meier of the censoring times (right-continuous, as in
    scikit-survival). Under purely administrative censoring at the horizon,
    G = 1 on [0, H) and BS(t) is the plain Brier score of the event "T <= t".
    """
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    survival_pred = np.asarray(survival_pred, dtype=float)
    times = np.asarray(times, dtype=float)
    g = _censoring_survival(durations, observed)
    g_at_t = g.at(times)
    g_at_y = g.at(durations)
    scores = np.empty(len(times))
    for j, t in enumerate(times):
        died = (durations <= t) & observed
        alive = durations > t
        with np.errstate(invalid="ignore", divide="ignore"):
            term_died = np.where(died, survival_pred[:, j] ** 2 / g_at_y, 0.0)
            term_alive = np.where(alive, (1.0 - survival_pred[:, j]) ** 2 / g_at_t[j], 0.0)
        scores[j] = np.mean(term_died + term_alive)
    return scores


def integrated_brier_score(
    durations: np.ndarray,
    observed: np.ndarray,
    survival_pred: np.ndarray,
    times: np.ndarray,
) -> float:
    """IBS = trapezoid integral of BS(t) over `times`, divided by its span.

    Same convention as `sksurv.metrics.integrated_brier_score`.
    """
    times = np.asarray(times, dtype=float)
    scores = brier_score_over_time(durations, observed, survival_pred, times)
    return float(np.trapezoid(scores, times) / (times[-1] - times[0]))


def d_calibration(
    survival_at_time: np.ndarray,
    survival_before_time: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray | float]:
    """Haider et al. (2020) D-calibration: is S_i(T_i) ~ Uniform(0, 1)?

    - Event at T_i: u_i = S_i(T_i). Step-function predictions put mass on
      a jump, so u_i is randomized uniformly in [S_i(T_i), S_i(T_i-)]
      (randomized PIT); with continuous S the two coincide.
    - Censored at C_i with c = S_i(C_i): the true S_i(T_i) is below c, so the
      unit mass is spread uniformly over [0, c): bins fully below c get
      width / c, the bin containing c gets (c - left edge) / c.
    Returns the bin masses (sum = N), and the chi-square uniformity p-value.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    s_at = np.asarray(survival_at_time, dtype=float)
    s_before = np.asarray(survival_before_time, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mass = np.zeros(n_bins)
    for i in range(len(s_at)):
        if observed[i]:
            u = s_at[i] + rng.random() * (s_before[i] - s_at[i])
            mass[min(int(u * n_bins), n_bins - 1)] += 1.0
        else:
            c = s_at[i]
            if c <= 0.0:
                mass[0] += 1.0
                continue
            for b in range(n_bins):
                lo, hi = edges[b], edges[b + 1]
                if hi <= c:
                    mass[b] += (hi - lo) / c
                elif lo < c:
                    mass[b] += (c - lo) / c
    expected = len(s_at) / n_bins
    statistic = float(np.sum((mass - expected) ** 2 / expected))
    return {
        "bin_edges": edges,
        "mass": mass,
        "chi2": statistic,
        "p_value": float(chi2.sf(statistic, n_bins - 1)),
    }


# ---------------------------------------------------------------------------
# 3. Dose-response summary of the survival-vs-x0 curve
# ---------------------------------------------------------------------------

def fit_logistic_survival(
    x: np.ndarray, k: np.ndarray, n: np.ndarray
) -> dict[str, float]:
    """Binomial MLE of S(x) = 1 / (1 + exp((x - x50) / s)), s > 0.

    x50 = location where survival is 50 % (the LD50 analogue),
    s   = transition width (smaller = sharper transition).
    `k` = events (departures), `n` = trials per x. Returns NaN with
    `degenerate=True` when no group has a mixed outcome on both sides,
    i.e. the transition is not inside the scanned range.
    """
    x = np.asarray(x, dtype=float)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    survivors = n - k
    if k.sum() == 0 or survivors.sum() == 0:
        return {"x50": np.nan, "s": np.nan, "degenerate": True, "nll": np.nan}
    frac = survivors / n
    order = np.argsort(x)
    crossing = np.flatnonzero(np.diff(np.sign(frac[order] - 0.5)) != 0)
    x50_init = float(x[order][crossing[0]]) if crossing.size else float(np.median(x))
    span = float(np.ptp(x)) or 1.0

    def nll(theta: np.ndarray) -> float:
        x50, log_s = theta
        z = (x - x50) / np.exp(log_s)
        log_surv = -np.logaddexp(0.0, z)
        log_event = -np.logaddexp(0.0, -z)
        return -float(np.sum(survivors * log_surv + k * log_event))

    result = minimize(
        nll,
        x0=np.array([x50_init, np.log(span / 10.0)]),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000},
    )
    x50, log_s = result.x
    # A step function (perfectly separated data) drives s -> 0; report it.
    return {
        "x50": float(x50),
        "s": float(np.exp(log_s)),
        "degenerate": False,
        "nll": float(result.fun),
    }


def logistic_survival(x: np.ndarray, x50: float, s: float) -> np.ndarray:
    return expit(-(np.asarray(x, dtype=float) - x50) / s)


def bootstrap_logistic(
    x: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Nonparametric bootstrap of (x50, s): resample trials within each x.

    Resampling n Bernoulli outcomes with replacement inside a group is the
    same as drawing k* ~ Binomial(n, k/n). Returns an (n_boot, 2) array.
    """
    out = np.full((n_boot, 2), np.nan)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n).astype(int)
    for b in range(n_boot):
        k_star = rng.binomial(n, k / n)
        fit = fit_logistic_survival(x, k_star, n)
        out[b] = fit["x50"], fit["s"]
    return out


# ---------------------------------------------------------------------------
# 4. Ensemble distribution scores
# ---------------------------------------------------------------------------

def crps_ensemble(members: np.ndarray, observation: float, fair: bool = False) -> float:
    """CRPS of an ensemble for a scalar observation.

    CRPS = E|X - y| - 1/2 E|X - X'|. `fair=True` uses the m(m-1) estimator of
    E|X - X'| (unbiased for the underlying distribution, Ferro 2014).
    Computed in O(m log m) via the sorted-sample identity.
    """
    x = np.sort(np.asarray(members, dtype=float))
    x = x[np.isfinite(x)]
    m = len(x)
    if m == 0:
        return float("nan")
    term1 = np.mean(np.abs(x - observation))
    weights = 2.0 * np.arange(1, m + 1) - m - 1
    pair_sum = 2.0 * np.sum(weights * x)  # = sum_{i,j} |x_i - x_j|
    denom = m * (m - 1) if (fair and m > 1) else m * m
    return float(term1 - 0.5 * pair_sum / denom)


def uncertainty_decomposition(p_members: np.ndarray, replicates: int | None = None) -> dict[str, np.ndarray]:
    """Split predictive variance of a Bernoulli outcome (law of total variance).

    p_members: (n_members, n_groups) event probabilities per member.
        total     = pbar (1 - pbar)
        aleatoric = mean_j p_j (1 - p_j)   (system randomness, within member)
        epistemic = var_j p_j               (disagreement between members)
    total = aleatoric + epistemic holds exactly.

    If each p_j is itself a frequency over R replicates, var_j(p_hat_j)
    contains binomial noise p(1-p)/R. With `replicates=R` the corrected
    estimates are also returned:
        epistemic_corrected = var_j(p_hat) - mean_j p_hat(1-p_hat) / (R - 1)
        aleatoric_corrected = mean_j p_hat(1-p_hat) * R / (R - 1)
    The entropy version (mutual information) is returned as well:
        H(pbar) = mean_j H(p_j) + MI.
    """
    p = np.asarray(p_members, dtype=float)
    pbar = p.mean(axis=0)
    total = pbar * (1.0 - pbar)
    aleatoric = np.mean(p * (1.0 - p), axis=0)
    epistemic = p.var(axis=0)
    out = {"pbar": pbar, "total": total, "aleatoric": aleatoric, "epistemic": epistemic}
    if replicates is not None and replicates > 1:
        r = float(replicates)
        within = np.mean(p * (1.0 - p), axis=0)
        out["epistemic_corrected"] = epistemic - within / (r - 1.0)
        out["aleatoric_corrected"] = within * r / (r - 1.0)

    def entropy(q: np.ndarray) -> np.ndarray:
        q = np.clip(q, 1e-12, 1.0 - 1e-12)
        return -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))

    out["entropy_total"] = entropy(pbar)
    out["entropy_aleatoric"] = np.mean(entropy(p), axis=0)
    out["mutual_information"] = out["entropy_total"] - out["entropy_aleatoric"]
    return out


def band_coverage(
    p_members: np.ndarray, p_truth: np.ndarray, lower_q: float = 0.05, upper_q: float = 0.95
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fraction of groups whose truth probability lies in the ensemble band."""
    p = np.asarray(p_members, dtype=float)
    lower = np.quantile(p, lower_q, axis=0)
    upper = np.quantile(p, upper_q, axis=0)
    inside = (p_truth >= lower) & (p_truth <= upper)
    return float(inside.mean()), lower, upper
