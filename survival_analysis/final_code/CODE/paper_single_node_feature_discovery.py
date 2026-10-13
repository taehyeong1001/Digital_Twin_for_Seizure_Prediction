#!/usr/bin/env python3
"""Discover causal warmup features for single-node Epileptor warning.

Feature selection uses only a fresh development cohort. Coordinates and seed
ranges reserved for the later audit are written to the protocol but never
simulated by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert, welch
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from paper_single_node_scrc import onset_for_signal, simulate_batch


DEVELOPMENT_X0 = np.asarray(
    [
        -2.2875,
        -2.2675,
        -2.2475,
        -2.2275,
        -2.2075,
        -2.1875,
        -2.1675,
        -2.1475,
        -2.1275,
        -2.1075,
        -2.0875,
        -2.0725,
        -2.0625,
        -2.0525,
        -2.0425,
        -2.0325,
        -2.0125,
        -1.9925,
        -1.9725,
        -1.9525,
        -1.9325,
    ]
)

RESERVED_AUDIT_X0 = np.asarray(
    [
        -2.2975,
        -2.2775,
        -2.2575,
        -2.2375,
        -2.2175,
        -2.1975,
        -2.1775,
        -2.1575,
        -2.1375,
        -2.1175,
        -2.0975,
        -2.0775,
        -2.0675,
        -2.0575,
        -2.0475,
        -2.0375,
        -2.0275,
        -2.0075,
        -1.9875,
        -1.9675,
        -1.9475,
        -1.9275,
    ]
)


FEATURE_CATALOG = {
    "mean": ("amplitude", "sample mean"),
    "std": ("amplitude", "sample standard deviation"),
    "variance": ("amplitude", "sample variance"),
    "mad": ("amplitude", "1.4826 * median absolute deviation"),
    "iqr": ("amplitude", "75th percentile - 25th percentile"),
    "rms": ("amplitude", "root mean square"),
    "skewness": ("distribution", "standardized third central moment"),
    "kurtosis": ("distribution", "Fisher excess kurtosis"),
    "p05": ("distribution", "5th percentile"),
    "p95": ("distribution", "95th percentile"),
    "peak_to_peak": ("distribution", "maximum - minimum"),
    "line_length": ("temporal", "mean absolute first difference"),
    "zero_crossing_rate": ("temporal", "centered sign changes per sample"),
    "ac_lag1": ("temporal", "lag-1 autocorrelation"),
    "ac_250ms": ("temporal", "autocorrelation at 250 ms"),
    "hjorth_mobility": ("temporal", "sqrt(var(dx) / var(x))"),
    "hjorth_complexity": ("temporal", "mobility(dx) / mobility(x)"),
    "envelope_mean": ("envelope", "mean Hilbert-envelope amplitude"),
    "envelope_std": ("envelope", "standard deviation of Hilbert envelope"),
    "envelope_p95": ("envelope", "95th percentile of Hilbert envelope"),
    "dominant_frequency": ("spectral", "frequency of maximum Welch PSD"),
    "spectral_centroid": ("spectral", "PSD-weighted mean frequency"),
    "spectral_entropy": ("spectral", "normalized Shannon entropy of Welch PSD"),
    "relative_power_0p1_1": ("spectral", "relative Welch power from 0.1 to 1 Hz"),
    "relative_power_1_4": ("spectral", "relative Welch power from 1 to 4 Hz"),
    "relative_power_4_8": ("spectral", "relative Welch power from 4 to 8 Hz"),
    "relative_power_8_16": ("spectral", "relative Welch power from 8 to 16 Hz"),
    "relative_power_16_30": ("spectral", "relative Welch power from 16 to 30 Hz"),
    "permutation_entropy": ("nonlinear", "normalized order-3 ordinal-pattern entropy"),
    "late_outlier_fraction": (
        "burst",
        "fraction after first 10 s beyond 3 robust first-10-s deviations",
    ),
    "burst_rate": ("burst", "3-robust-deviation excursion onsets per second"),
    "max_abs_robust_z": ("burst", "maximum absolute first-10-s robust z-score"),
    "window_std_slope": ("trend", "linear slope of four consecutive 10-s std values"),
    "window_ac1_slope": ("trend", "linear slope of four consecutive 10-s AC(1) values"),
    "window_line_length_slope": (
        "trend",
        "linear slope of four consecutive 10-s line-length values",
    ),
    "window_spectral_entropy_slope": (
        "trend",
        "linear slope of four consecutive 10-s spectral entropies",
    ),
    "std_last_first_ratio": ("multiscale", "last-10-s std / first-10-s std"),
    "ac1_last_first_difference": ("multiscale", "last-10-s AC(1) - first-10-s AC(1)"),
    "line_length_last_first_ratio": (
        "multiscale",
        "last-10-s line length / first-10-s line length",
    ),
    "window_std_cv": ("multiscale", "coefficient of variation across four 10-s std values"),
    "window_ac1_range": ("multiscale", "range across four 10-s AC(1) values"),
}

# These depend on the arbitrary DC level of the latent source proxy and are not
# expected to survive common EEG referencing/high-pass preprocessing unchanged.
OFFSET_SENSITIVE_FEATURES = {"mean", "p05", "p95", "rms"}


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-12))


def autocorrelation(signal: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(signal) <= lag:
        return np.nan
    left = signal[:-lag] - np.mean(signal[:-lag])
    right = signal[lag:] - np.mean(signal[lag:])
    denominator = np.sqrt(np.sum(left**2) * np.sum(right**2))
    return float(np.sum(left * right) / denominator) if denominator > 1e-12 else 0.0


def spectral_features(signal: np.ndarray, sample_hz: float) -> dict[str, float]:
    frequencies, power = welch(
        signal - np.mean(signal),
        fs=sample_hz,
        nperseg=min(len(signal), 512),
        noverlap=min(len(signal) // 2, 256),
    )
    positive = frequencies >= 0.1
    frequencies = frequencies[positive]
    power = np.maximum(power[positive], 0.0)
    total = float(np.sum(power)) + 1e-15
    probability = power / total
    entropy = -np.sum(probability * np.log(probability + 1e-15))
    entropy /= max(np.log(len(probability)), 1e-12)

    def relative_power(low: float, high: float) -> float:
        band = (frequencies >= low) & (frequencies < high)
        return float(np.sum(power[band]) / total)

    return {
        "dominant_frequency": float(frequencies[int(np.argmax(power))]),
        "spectral_centroid": float(np.sum(frequencies * power) / total),
        "spectral_entropy": float(entropy),
        "relative_power_0p1_1": relative_power(0.1, 1.0),
        "relative_power_1_4": relative_power(1.0, 4.0),
        "relative_power_4_8": relative_power(4.0, 8.0),
        "relative_power_8_16": relative_power(8.0, 16.0),
        "relative_power_16_30": relative_power(16.0, 30.0),
    }


def permutation_entropy(signal: np.ndarray, order: int = 3) -> float:
    if len(signal) < order + 1:
        return np.nan
    patterns: dict[tuple[int, ...], int] = {}
    for start in range(len(signal) - order + 1):
        pattern = tuple(np.argsort(signal[start : start + order], kind="mergesort"))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    counts = np.asarray(list(patterns.values()), dtype=float)
    probability = counts / np.sum(counts)
    entropy = -np.sum(probability * np.log(probability + 1e-15))
    return float(entropy / np.log(math.factorial(order)))


def segment_statistics(signal: np.ndarray, sample_hz: float) -> dict[str, float]:
    centered = signal - np.mean(signal)
    first_difference = np.diff(signal)
    second_difference = np.diff(first_difference)
    variance = float(np.var(signal))
    mobility = np.sqrt(np.var(first_difference) / max(variance, 1e-15))
    derivative_mobility = np.sqrt(
        np.var(second_difference) / max(np.var(first_difference), 1e-15)
    )
    envelope = np.abs(hilbert(centered))
    values = {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "variance": variance,
        "mad": float(1.4826 * np.median(np.abs(signal - np.median(signal)))),
        "iqr": float(np.quantile(signal, 0.75) - np.quantile(signal, 0.25)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "skewness": float(skew(signal, bias=False)),
        "kurtosis": float(kurtosis(signal, fisher=True, bias=False)),
        "p05": float(np.quantile(signal, 0.05)),
        "p95": float(np.quantile(signal, 0.95)),
        "peak_to_peak": float(np.ptp(signal)),
        "line_length": float(np.mean(np.abs(first_difference))),
        "zero_crossing_rate": float(np.mean(centered[:-1] * centered[1:] < 0.0)),
        "ac_lag1": autocorrelation(signal, 1),
        "ac_250ms": autocorrelation(signal, max(1, int(round(0.25 * sample_hz)))),
        "hjorth_mobility": float(mobility),
        "hjorth_complexity": float(derivative_mobility / max(mobility, 1e-15)),
        "envelope_mean": float(np.mean(envelope)),
        "envelope_std": float(np.std(envelope)),
        "envelope_p95": float(np.quantile(envelope, 0.95)),
    }
    values.update(spectral_features(signal, sample_hz))
    return values


def extract_features(signal: np.ndarray, sample_hz: float) -> dict[str, float]:
    values = segment_statistics(signal, sample_hz)
    downsampled = signal[:: max(1, int(round(sample_hz / 16.0)))]
    values["permutation_entropy"] = permutation_entropy(downsampled)

    baseline_samples = int(round(10.0 * sample_hz))
    baseline = signal[:baseline_samples]
    center = float(np.median(baseline))
    scale = float(1.4826 * np.median(np.abs(baseline - center)))
    if scale < 1e-12:
        scale = max(float(np.std(baseline)), 1.0)
    robust_z = (signal - center) / scale
    excursions = np.abs(robust_z) > 3.0
    late = excursions[baseline_samples:]
    onsets = np.flatnonzero(excursions & ~np.r_[False, excursions[:-1]])
    values["late_outlier_fraction"] = float(np.mean(late)) if late.size else 0.0
    values["burst_rate"] = float(len(onsets) / (len(signal) / sample_hz))
    values["max_abs_robust_z"] = float(np.max(np.abs(robust_z)))

    segment_samples = int(round(10.0 * sample_hz))
    segments = [signal[start : start + segment_samples] for start in range(0, len(signal), segment_samples)]
    segments = [segment for segment in segments if len(segment) == segment_samples][:4]
    if len(segments) != 4:
        raise ValueError("Warmup must contain four complete 10-second segments")
    segment_values = [segment_statistics(segment, sample_hz) for segment in segments]
    centers = np.asarray([5.0, 15.0, 25.0, 35.0])

    def series(name: str) -> np.ndarray:
        return np.asarray([segment[name] for segment in segment_values])

    std_values = series("std")
    ac_values = series("ac_lag1")
    line_values = series("line_length")
    entropy_values = series("spectral_entropy")
    values["window_std_slope"] = float(np.polyfit(centers, std_values, 1)[0])
    values["window_ac1_slope"] = float(np.polyfit(centers, ac_values, 1)[0])
    values["window_line_length_slope"] = float(np.polyfit(centers, line_values, 1)[0])
    values["window_spectral_entropy_slope"] = float(
        np.polyfit(centers, entropy_values, 1)[0]
    )
    values["std_last_first_ratio"] = safe_ratio(std_values[-1], std_values[0])
    values["ac1_last_first_difference"] = float(ac_values[-1] - ac_values[0])
    values["line_length_last_first_ratio"] = safe_ratio(line_values[-1], line_values[0])
    values["window_std_cv"] = safe_ratio(np.std(std_values), np.mean(std_values))
    values["window_ac1_range"] = float(np.ptp(ac_values))

    if set(values) != set(FEATURE_CATALOG):
        missing = set(FEATURE_CATALOG) - set(values)
        extra = set(values) - set(FEATURE_CATALOG)
        raise RuntimeError(f"Feature catalog mismatch: missing={missing}, extra={extra}")
    return values


def simulate_development_cohort(args: argparse.Namespace) -> pd.DataFrame:
    sample_hz = 256.0 / args.rc_stride
    warmup_samples = int(round(args.warmup_s * sample_hz))
    horizon_end = args.warmup_s + args.rollout_s
    rows: list[dict[str, float | int | str]] = []
    for x0_index, x0 in enumerate(DEVELOPMENT_X0):
        simulation_seed = args.seed + 100000 + 1000 * x0_index
        time_s, proxy = simulate_batch(
            np.full(args.trajectories_per_x0, x0), args.duration_s, simulation_seed
        )
        sampled = proxy[:: args.rc_stride]
        for trajectory in range(args.trajectories_per_x0):
            onset = onset_for_signal(time_s, proxy[:, trajectory])
            eligible = onset is None or onset > args.warmup_s
            if not eligible:
                continue
            event = onset is not None and onset <= horizon_end
            row: dict[str, float | int | str] = {
                "x0": float(x0),
                "simulation_seed": simulation_seed,
                "trajectory": trajectory,
                "gt_onset_absolute_s": "" if onset is None else onset,
                "event_within_rollout": int(event),
                "time_to_event_after_warmup_s": "" if not event else onset - args.warmup_s,
            }
            row.update(extract_features(sampled[:warmup_samples, trajectory], sample_hz))
            rows.append(row)
        print(f"feature cohort x0={x0:.4f}: eligible={sum(r['x0'] == float(x0) for r in rows)}", flush=True)
    return pd.DataFrame(rows)


def grouped_auc(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    splits: int,
    seed: int,
) -> tuple[float, float]:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, matrix, labels, groups=groups, cv=cv, scoring="roc_auc")
    return float(np.mean(scores)), float(np.std(scores))


def rank_features(
    frame: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    labels = frame["event_within_rollout"].to_numpy(dtype=int)
    groups = frame["x0"].to_numpy(dtype=float)
    positive = labels == 1
    rows = []
    for name, (category, definition) in FEATURE_CATALOG.items():
        values = frame[name].to_numpy(dtype=float)
        finite = np.isfinite(values)
        raw_auc = roc_auc_score(labels[finite], values[finite])
        predictive_auc = max(raw_auc, 1.0 - raw_auc)
        direction = "higher_risk" if raw_auc >= 0.5 else "lower_risk"
        cv_mean, cv_std = grouped_auc(
            values[:, None], labels, groups, args.cv_splits, args.seed
        )
        pooled = np.sqrt(
            0.5 * (np.var(values[positive]) + np.var(values[~positive]))
        )
        effect = (
            float((np.mean(values[positive]) - np.mean(values[~positive])) / pooled)
            if pooled > 1e-12
            else 0.0
        )
        x0_spearman = spearmanr(values, groups, nan_policy="omit").statistic
        positive_time = frame.loc[positive, "time_to_event_after_warmup_s"].to_numpy(dtype=float)
        time_spearman = spearmanr(values[positive], positive_time, nan_policy="omit").statistic
        rows.append(
            {
                "feature": name,
                "category": category,
                "definition": definition,
                "predictive_auc_all": predictive_auc,
                "direction": direction,
                "group_cv_auc_mean": cv_mean,
                "group_cv_auc_std": cv_std,
                "standardized_effect": effect,
                "spearman_x0": x0_spearman,
                "spearman_time_to_event": time_spearman,
            }
        )
    return pd.DataFrame(rows).sort_values("group_cv_auc_mean", ascending=False).reset_index(drop=True)


def select_nonredundant_features(
    frame: pd.DataFrame,
    ranking: pd.DataFrame,
    correlation_limit: float,
    maximum: int,
) -> list[str]:
    selected: list[str] = []
    for name in ranking["feature"]:
        if not selected:
            selected.append(name)
        else:
            correlations = frame[selected + [name]].corr().abs().loc[name, selected]
            if float(np.nanmax(correlations)) < correlation_limit:
                selected.append(name)
        if len(selected) >= maximum:
            break
    return selected


def multivariate_curve(
    frame: pd.DataFrame,
    candidates: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    labels = frame["event_within_rollout"].to_numpy(dtype=int)
    groups = frame["x0"].to_numpy(dtype=float)
    rows = []
    for count in range(1, len(candidates) + 1):
        names = candidates[:count]
        mean_auc, std_auc = grouped_auc(
            frame[names].to_numpy(dtype=float),
            labels,
            groups,
            args.cv_splits,
            args.seed,
        )
        rows.append(
            {
                "feature_count": count,
                "features": ";".join(names),
                "group_cv_auc_mean": mean_auc,
                "group_cv_auc_std": std_auc,
            }
        )
    return pd.DataFrame(rows)


def choose_feature_count(curve: pd.DataFrame, tolerance: float = 0.01) -> int:
    best = float(curve["group_cv_auc_mean"].max())
    eligible = curve[curve["group_cv_auc_mean"] >= best - tolerance]
    return int(eligible["feature_count"].min())


def plot_discovery(
    output: Path,
    frame: pd.DataFrame,
    ranking: pd.DataFrame,
    candidates: list[str],
    curve: pd.DataFrame,
    selected_count: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    selected = candidates[:selected_count]
    grouped = frame.groupby("x0")[selected].mean()
    standardized = (grouped - grouped.mean()) / grouped.std().replace(0.0, 1.0)
    image = axes[0, 0].imshow(
        standardized.T,
        aspect="auto",
        origin="lower",
        extent=[grouped.index.min(), grouped.index.max(), -0.5, len(selected) - 0.5],
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=2.5,
        interpolation="nearest",
    )
    axes[0, 0].set_yticks(np.arange(len(selected)), selected)
    axes[0, 0].axvline(-2.1, color="#111111", ls="--", lw=1.2)
    axes[0, 0].set_xlabel("development x0")
    axes[0, 0].set_title("(a) Selected warmup-feature trends", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[0, 0], pad=0.01, label="z-scored x0-group mean")

    top = ranking.head(15).iloc[::-1]
    axes[0, 1].barh(
        top["feature"], top["group_cv_auc_mean"], xerr=top["group_cv_auc_std"],
        color="#3182bd", alpha=0.9, capsize=2
    )
    axes[0, 1].axvline(0.5, color="#555555", ls=":", lw=1.1)
    axes[0, 1].set_xlim(0.45, 1.0)
    axes[0, 1].set_xlabel("x0-grouped logistic CV AUC")
    axes[0, 1].set_title("(b) Univariate information", loc="left", fontweight="bold")

    correlation = frame[candidates].corr().to_numpy()
    correlation_image = axes[1, 0].imshow(
        correlation, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
    )
    axes[1, 0].set_xticks(np.arange(len(candidates)), candidates, rotation=55, ha="right")
    axes[1, 0].set_yticks(np.arange(len(candidates)), candidates)
    axes[1, 0].set_title("(c) Nonredundant-candidate correlation", loc="left", fontweight="bold")
    fig.colorbar(correlation_image, ax=axes[1, 0], pad=0.01, label="Pearson r")

    axes[1, 1].errorbar(
        curve["feature_count"], curve["group_cv_auc_mean"],
        yerr=curve["group_cv_auc_std"], marker="o", capsize=3, lw=1.7
    )
    axes[1, 1].axvline(selected_count, color="#d62728", ls="--", lw=1.3,
                       label=f"selected count = {selected_count}")
    axes[1, 1].axhline(0.5, color="#555555", ls=":", lw=1.1)
    axes[1, 1].set_xlabel("number of ranked nonredundant features")
    axes[1, 1].set_ylabel("x0-grouped logistic CV AUC")
    axes[1, 1].set_ylim(0.45, 1.0)
    axes[1, 1].set_title("(d) Incremental multivariate performance", loc="left", fontweight="bold")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.16)
    fig.suptitle("Causal warmup feature discovery, single-node Epileptor", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=230)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def write_catalog(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["feature", "category", "definition", "causal_window", "offset_robust"]
        )
        for name, (category, definition) in FEATURE_CATALOG.items():
            writer.writerow(
                [
                    name,
                    category,
                    definition,
                    "first 40 s only",
                    int(name not in OFFSET_SENSITIVE_FEATURES),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--duration-s", type=float, default=160.0)
    parser.add_argument("--warmup-s", type=float, default=40.0)
    parser.add_argument("--rollout-s", type=float, default=120.0)
    parser.add_argument("--rc-stride", type=int, default=4)
    parser.add_argument("--trajectories-per-x0", type=int, default=12)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--correlation-limit", type=float, default=0.92)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=310831)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse the saved development feature CSV without resimulation.",
    )
    args = parser.parse_args()

    if args.warmup_s != 40.0:
        raise ValueError("Current trend catalog requires a 40 s warmup")
    if args.warmup_s + args.rollout_s > args.duration_s:
        raise ValueError("warmup + rollout must not exceed duration")
    if set(DEVELOPMENT_X0.tolist()) & set(RESERVED_AUDIT_X0.tolist()):
        raise RuntimeError("Development and reserved audit x0 overlap")

    data_dir = args.out_dir / "data" / "paper_single_node_feature_discovery"
    figure_dir = args.out_dir / "figures" / "RC" / "paper_single_node_feature_discovery"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    feature_path = data_dir / "development_warmup_features.csv"
    if args.reuse_features:
        if not feature_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing {feature_path}")
        frame = pd.read_csv(feature_path)
    else:
        frame = simulate_development_cohort(args)
    ranking = rank_features(frame, args)
    candidates = select_nonredundant_features(
        frame, ranking, args.correlation_limit, args.max_candidates
    )
    curve = multivariate_curve(frame, candidates, args)
    selected_count = choose_feature_count(curve)
    selected = candidates[:selected_count]
    offset_robust_ranking = ranking[
        ~ranking["feature"].isin(OFFSET_SENSITIVE_FEATURES)
    ].reset_index(drop=True)
    offset_robust_candidates = select_nonredundant_features(
        frame,
        offset_robust_ranking,
        args.correlation_limit,
        args.max_candidates,
    )
    offset_robust_curve = multivariate_curve(
        frame, offset_robust_candidates, args
    )
    offset_robust_count = choose_feature_count(offset_robust_curve)
    offset_robust_selected = offset_robust_candidates[:offset_robust_count]

    frame.to_csv(feature_path, index=False)
    ranking.to_csv(data_dir / "feature_ranking.csv", index=False)
    curve.to_csv(data_dir / "multivariate_feature_curve.csv", index=False)
    offset_robust_curve.to_csv(
        data_dir / "offset_robust_feature_curve.csv", index=False
    )
    write_catalog(data_dir / "feature_catalog.csv")
    protocol = {
        "purpose": "feature discovery only; reserved audit set remains untouched",
        "development_x0": DEVELOPMENT_X0.tolist(),
        "development_seed_base": args.seed + 100000,
        "reserved_audit_x0": RESERVED_AUDIT_X0.tolist(),
        "reserved_audit_seed_base": args.seed + 200000,
        "arguments": vars(args) | {"out_dir": str(args.out_dir)},
        "event_definition": "first local minimum x2-x1 <= -2 within 120 s after warmup",
        "excluded": "collapse at or before 40 s warmup",
        "candidate_features": candidates,
        "selected_features": selected,
        "selected_group_cv_auc": float(
            curve.loc[curve["feature_count"] == selected_count, "group_cv_auc_mean"].iloc[0]
        ),
        "selected_group_cv_auc_std": float(
            curve.loc[curve["feature_count"] == selected_count, "group_cv_auc_std"].iloc[0]
        ),
        "offset_sensitive_features": sorted(OFFSET_SENSITIVE_FEATURES),
        "primary_offset_robust_features": offset_robust_selected,
        "primary_offset_robust_group_cv_auc": float(
            offset_robust_curve.loc[
                offset_robust_curve["feature_count"] == offset_robust_count,
                "group_cv_auc_mean",
            ].iloc[0]
        ),
        "primary_offset_robust_group_cv_auc_std": float(
            offset_robust_curve.loc[
                offset_robust_curve["feature_count"] == offset_robust_count,
                "group_cv_auc_std",
            ].iloc[0]
        ),
    }
    (data_dir / "selected_features.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    plot_discovery(
        figure_dir / "paper_single_node_feature_discovery",
        frame,
        ranking,
        candidates,
        curve,
        selected_count,
    )

    print("\nTop causal warmup features")
    print(
        ranking[["feature", "category", "group_cv_auc_mean", "group_cv_auc_std"]]
        .head(15)
        .to_string(index=False)
    )
    print(f"\nNonredundant candidates: {candidates}")
    print(f"Selected feature subset: {selected}")
    print(
        f"Grouped CV AUC: {protocol['selected_group_cv_auc']:.3f} "
        f"+/- {protocol['selected_group_cv_auc_std']:.3f}"
    )
    print(f"Primary offset-robust subset: {offset_robust_selected}")
    print(
        f"Offset-robust grouped CV AUC: "
        f"{protocol['primary_offset_robust_group_cv_auc']:.3f} +/- "
        f"{protocol['primary_offset_robust_group_cv_auc_std']:.3f}"
    )
    print("Reserved audit coordinates were not simulated.")


if __name__ == "__main__":
    main()
