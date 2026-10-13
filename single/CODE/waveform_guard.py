"""Training-only detectors for sustained abnormal waveform shape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicityReference:
    threshold: float
    training_max: float
    margin: float
    window_samples: int
    step_samples: int
    first_lag: int
    last_lag: int
    sustain_windows: int


@dataclass(frozen=True)
class MedianReference:
    threshold: float
    training_min: float
    training_max: float
    margin_fraction: float
    window_samples: int
    step_samples: int
    sustain_windows: int


def rolling_periodicity(
    signal: np.ndarray,
    window_samples: int,
    step_samples: int,
    first_lag: int,
    last_lag: int,
) -> np.ndarray:
    """Maximum centered autocorrelation over a physiologic lag range."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < window_samples:
        return np.empty(0)
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_samples)[::step_samples].copy()
    windows -= np.mean(windows, axis=1, keepdims=True)
    fft_size = 2 * window_samples
    spectrum = np.fft.rfft(windows, n=fft_size, axis=1)
    autocorrelation = np.fft.irfft(np.abs(spectrum) ** 2, n=fft_size, axis=1)
    energy_cumsum = np.cumsum(windows**2, axis=1)
    lags = np.arange(first_lag, last_lag + 1)
    left_energy = energy_cumsum[:, window_samples - lags - 1]
    right_energy = energy_cumsum[:, -1, None] - energy_cumsum[:, lags - 1]
    normalized = autocorrelation[:, lags] / np.sqrt(np.maximum(left_energy * right_energy, 1e-15))
    return np.max(normalized, axis=1)


def fit_periodicity_reference(
    normal_training: list[np.ndarray],
    sample_hz: float,
    window_s: float = 10.0,
    step_s: float = 1.0,
    lag_min_s: float = 0.25,
    lag_max_s: float = 1.25,
    sustain_windows: int = 5,
    margin: float = 0.02,
) -> PeriodicityReference:
    window = int(round(window_s * sample_hz))
    step = int(round(step_s * sample_hz))
    first_lag = int(round(lag_min_s * sample_hz))
    last_lag = int(round(lag_max_s * sample_hz))
    values = np.concatenate([
        rolling_periodicity(signal, window, step, first_lag, last_lag)
        for signal in normal_training
    ])
    training_max = float(np.max(values))
    return PeriodicityReference(
        threshold=min(0.999, training_max + margin),
        training_max=training_max,
        margin=margin,
        window_samples=window,
        step_samples=step,
        first_lag=first_lag,
        last_lag=last_lag,
        sustain_windows=sustain_windows,
    )


def periodicity_event_mask(signals: np.ndarray, reference: PeriodicityReference) -> np.ndarray:
    """Flag trajectories with five consecutive 10-s windows above the normal bound."""
    output = np.zeros(signals.shape[1], dtype=bool)
    for column in range(signals.shape[1]):
        values = rolling_periodicity(
            signals[:, column],
            reference.window_samples,
            reference.step_samples,
            reference.first_lag,
            reference.last_lag,
        )
        if len(values) >= reference.sustain_windows:
            above = values > reference.threshold
            output[column] = bool(np.any(
                np.lib.stride_tricks.sliding_window_view(above, reference.sustain_windows).all(axis=1)
            ))
    return output


def fit_median_reference(
    normal_training: list[np.ndarray],
    sample_hz: float,
    window_s: float = 10.0,
    step_s: float = 1.0,
    sustain_windows: int = 5,
    margin_fraction: float = 0.20,
) -> MedianReference:
    """Set a conservative upper baseline from normal training windows only."""
    window = int(round(window_s * sample_hz))
    step = int(round(step_s * sample_hz))
    values = np.concatenate([
        np.median(
            np.lib.stride_tricks.sliding_window_view(signal, window)[::step],
            axis=1,
        )
        for signal in normal_training
    ])
    low = float(np.min(values))
    high = float(np.max(values))
    return MedianReference(
        threshold=high + margin_fraction * (high - low),
        training_min=low,
        training_max=high,
        margin_fraction=margin_fraction,
        window_samples=window,
        step_samples=step,
        sustain_windows=sustain_windows,
    )


def high_median_event_mask(signals: np.ndarray, reference: MedianReference) -> np.ndarray:
    """Flag a persistently raised baseline outside the normal waveform envelope."""
    output = np.zeros(signals.shape[1], dtype=bool)
    for column in range(signals.shape[1]):
        if len(signals) < reference.window_samples:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(
            signals[:, column], reference.window_samples
        )[::reference.step_samples]
        above = np.median(windows, axis=1) > reference.threshold
        if len(above) >= reference.sustain_windows:
            output[column] = bool(np.any(
                np.lib.stride_tricks.sliding_window_view(above, reference.sustain_windows).all(axis=1)
            ))
    return output
