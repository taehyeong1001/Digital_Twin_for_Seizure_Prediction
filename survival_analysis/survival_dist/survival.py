"""Survival-analysis primitives for normal-departure (seizure-onset) times.

Ported in spirit from EESN `R_experiments/R0/R0c_metrics_and_plots.jl`
(`kaplan_meier`, `km_half_life`) and extended for this project:

- event times are given as seconds from rollout start, NaN = no event;
- every trial is administratively censored at the horizon H;
- Greenwood variance (log-log interval), Jeffreys interval for a proportion,
  exponential escape-rate MLE.

Definitions are documented in `docs/02_metrics.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, chi2, norm


@dataclass(frozen=True)
class KMCurve:
    """Right-continuous Kaplan-Meier step function.

    `time[k]` is the k-th distinct event time; S(t) = `survival[k]` for
    time[k] <= t < time[k+1]. `time[0] = 0`, `survival[0] = 1`.
    """

    time: np.ndarray
    survival: np.ndarray
    variance: np.ndarray  # Greenwood variance of S(t)
    lower: np.ndarray
    upper: np.ndarray
    n: int

    def at(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        index = np.searchsorted(self.time, t, side="right") - 1
        return self.survival[np.clip(index, 0, None)]

    def half_life(self) -> float:
        """First t with S(t) <= 0.5 (NaN if never reached)."""
        hits = np.flatnonzero(self.survival <= 0.5)
        return float(self.time[hits[0]]) if hits.size else float("nan")


def censor(event_times: np.ndarray, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """(durations, observed) with administrative censoring at `horizon`.

    An event exactly at the horizon counts as observed, consistent with the
    original pipeline's `onset <= horizon` survival definition.
    """
    times = np.asarray(event_times, dtype=float)
    observed = np.isfinite(times) & (times <= horizon)
    durations = np.where(observed, times, horizon)
    return durations, observed


def kaplan_meier(
    durations: np.ndarray,
    observed: np.ndarray,
    level: float = 0.95,
) -> KMCurve:
    """Kaplan-Meier estimate with Greenwood variance and log(-log) interval."""
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    event_times = np.unique(durations[observed])
    s, greenwood_sum = 1.0, 0.0
    times, survival, variance, lower, upper = [0.0], [1.0], [0.0], [1.0], [1.0]
    z = norm.ppf(0.5 + level / 2.0)
    for t in event_times:
        at_risk = np.count_nonzero(durations >= t)
        deaths = np.count_nonzero((durations == t) & observed)
        s *= 1.0 - deaths / at_risk
        if at_risk > deaths:
            greenwood_sum += deaths / (at_risk * (at_risk - deaths))
        else:
            greenwood_sum = np.inf
        var = s * s * greenwood_sum if np.isfinite(greenwood_sum) else np.nan
        if 0.0 < s < 1.0 and np.isfinite(greenwood_sum):
            # Kalbfleisch-Prentice log(-log) transform keeps the band in [0, 1].
            se = np.sqrt(greenwood_sum) / abs(np.log(s))
            lo, hi = s ** np.exp(z * se), s ** np.exp(-z * se)
        else:
            lo, hi = (0.0, 0.0) if s == 0.0 else (np.nan, np.nan)
        times.append(t)
        survival.append(s)
        variance.append(var)
        lower.append(lo)
        upper.append(hi)
    return KMCurve(
        time=np.asarray(times),
        survival=np.asarray(survival),
        variance=np.asarray(variance),
        lower=np.asarray(lower),
        upper=np.asarray(upper),
        n=len(durations),
    )


def survival_at_horizon(event_times: np.ndarray, horizon: float) -> float:
    """Fraction of trials without an event in [0, horizon].

    With no censoring before the horizon this equals the KM value at the
    horizon and the original `grouped_survival`.
    """
    _, observed = censor(event_times, horizon)
    return float(1.0 - observed.mean())


def jeffreys_interval(k: np.ndarray, n: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Equal-tailed Jeffreys interval for a binomial proportion k/n.

    Posterior Beta(k + 1/2, n - k + 1/2); Brown, Cai & DasGupta (2001)
    endpoint convention: lower = 0 when k = 0, upper = 1 when k = n.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    alpha = 1.0 - level
    lower = np.where(k > 0, beta.ppf(alpha / 2.0, k + 0.5, n - k + 0.5), 0.0)
    upper = np.where(k < n, beta.ppf(1.0 - alpha / 2.0, k + 0.5, n - k + 0.5), 1.0)
    return lower, upper


def jeffreys_posterior_samples(
    k: int, n: int, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draws from the Jeffreys posterior Beta(k + 1/2, n - k + 1/2)."""
    return rng.beta(k + 0.5, n - k + 0.5, size=size)


def exponential_escape_rate(
    durations: np.ndarray, observed: np.ndarray, level: float = 0.95
) -> tuple[float, float, float]:
    """MLE of a constant hazard kappa with right censoring, plus exact CI.

    kappa_hat = d / sum(durations). The interval uses the chi-square pivot
    for the total exposure (exact for type-II, standard approximation here).
    Returns (kappa, lower, upper) in 1/s.
    """
    durations = np.asarray(durations, dtype=float)
    d = int(np.count_nonzero(observed))
    exposure = float(durations.sum())
    if exposure <= 0.0:
        return float("nan"), float("nan"), float("nan")
    alpha = 1.0 - level
    kappa = d / exposure
    lower = chi2.ppf(alpha / 2.0, 2 * d) / (2.0 * exposure) if d > 0 else 0.0
    upper = chi2.ppf(1.0 - alpha / 2.0, 2 * d + 2) / (2.0 * exposure)
    return float(kappa), float(lower), float(upper)


def grouped(keys: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sorted unique keys and the trial indices of each group."""
    keys = np.asarray(keys)
    unique = np.unique(keys)
    return unique, [np.flatnonzero(keys == value) for value in unique]


def grouped_event_counts(
    keys: np.ndarray, event_times: np.ndarray, horizon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(unique keys, events k, trials n) per group, events counted up to the horizon."""
    unique, groups = grouped(keys)
    _, observed = censor(event_times, horizon)
    k = np.asarray([observed[g].sum() for g in groups])
    n = np.asarray([len(g) for g in groups])
    return unique, k, n
