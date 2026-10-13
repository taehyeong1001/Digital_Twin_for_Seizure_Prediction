#!/usr/bin/env python3
"""Three-node Proix-2017 Epileptor ground truth for the strict multi-node SCDT.

Network, coupling and noise follow MultiNode_Epileptor's three-node SCDT
(triangle connectome, G = 0.7, noise variance 0.0025 on x2, y2).  A single
hidden drift ``lam`` sets x0_i = lam + OFFSETS[i]; it is used only to generate
records and to align a validation axis.
"""
from __future__ import annotations

import numpy as np

from paper_epileptor_core import PaperEpileptorConfig, source_proxy

N_NODES = 3
RAW_HZ = 256
RC_HZ = 32
STRIDE = RAW_HZ // RC_HZ
COUPLING = 0.7
NOISE_VARIANCE = 0.0025
EVENT_THRESHOLD = -0.5
BLANK_S = 5.0
OFFSETS = np.array([0.00, -0.05, -0.10])


def triangle_connectivity() -> np.ndarray:
    return np.asarray([[0.0, 1.0, 0.45], [1.0, 0.0, 0.70], [0.45, 0.70, 0.0]])


def _rhs(states: np.ndarray, x0: np.ndarray, conn: np.ndarray, cfg: PaperEpileptorConfig) -> np.ndarray:
    x1, y1, z, x2, y2, g = np.moveaxis(states, -1, 0)
    f1 = np.where(x1 < 0.0, x1**3 - 3.0 * x1**2, (x2 - 0.6 * (z - 4.0) ** 2) * x1)
    f2 = np.where(x2 < -0.25, 0.0, 6.0 * (x2 + 0.25))
    coupling = x1 @ conn.T - conn.sum(axis=1) * x1
    out = np.empty_like(states)
    out[..., 0] = y1 - f1 - z + cfg.i1
    out[..., 1] = 1.0 - 5.0 * x1**2 - y1
    out[..., 2] = (4.0 * (x1 - x0) - z - cfg.global_coupling * coupling) / cfg.tau0
    out[..., 3] = -y2 + x2 - x2**3 + cfg.i2 + 0.002 * g - 0.3 * (z - 3.5)
    out[..., 4] = (-y2 + f2) / cfg.tau2
    out[..., 5] = x1 - cfg.gamma * g
    return out


class BatchNetwork:
    """``batch`` independent copies of the three-node network (state [batch, node, 6])."""

    def __init__(self, batch: int, seed: int):
        self.batch = batch
        self.cfg = PaperEpileptorConfig(n_nodes=N_NODES, global_coupling=COUPLING,
                                        noise_variance=NOISE_VARIANCE, seed=seed)
        self.conn = triangle_connectivity()
        self.rng = np.random.default_rng(seed)

    def initial_state(self) -> np.ndarray:
        base = np.asarray([-1.55, -11.8, 3.05, -0.82, 0.0, 0.0])
        scale = np.asarray([0.03, 0.20, 0.04, 0.02, 0.02, 0.03])
        s = np.broadcast_to(base, (self.batch, N_NODES, 6)).copy()
        return s + self.rng.normal(0.0, scale, s.shape)

    def x0(self, lam) -> np.ndarray:
        lam = np.broadcast_to(np.asarray(lam, float), (self.batch,))
        return lam[:, None] + OFFSETS[None, :]

    def step(self, s: np.ndarray, x0: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        d1 = _rhs(s, x0, self.conn, cfg)
        noise = np.zeros_like(s)
        scale = np.sqrt(cfg.noise_variance * cfg.integration_dt)
        noise[..., 3] = scale * self.rng.standard_normal(s.shape[:2])
        noise[..., 4] = scale * self.rng.standard_normal(s.shape[:2])
        p = s + cfg.integration_dt * d1 + noise
        d2 = _rhs(p, x0, self.conn, cfg)
        return s + 0.5 * cfg.integration_dt * (d1 + d2) + noise

    def run(self, s: np.ndarray, lam, seconds: float, record: bool = True):
        """Return (state, proxy[time, batch, node]) sampled at RC_HZ."""
        x0 = self.x0(lam)
        steps = int(round(seconds * RAW_HZ))
        out = np.empty((steps // STRIDE, self.batch, N_NODES)) if record else None
        k = 0
        for i in range(steps):
            s = self.step(s, x0)
            if record and (i + 1) % STRIDE == 0:
                out[k] = source_proxy(s)
                k += 1
        if not np.all(np.isfinite(s)):
            raise FloatingPointError("network became non-finite")
        return s, out


def departure(proxy: np.ndarray, blank_s: float = BLANK_S) -> np.ndarray:
    """Any node <= -0.5 after blanking. proxy: [time, batch, node] at RC_HZ."""
    start = int(round(blank_s * RC_HZ))
    return np.any(proxy[start:] <= EVENT_THRESHOLD, axis=(0, 2))


def first_departure(proxy: np.ndarray, blank_s: float = BLANK_S) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_s, node) of the first departure; NaN / -1 when absent."""
    start = int(round(blank_s * RC_HZ))
    hit = proxy[start:] <= EVENT_THRESHOLD
    times = np.full(proxy.shape[1], np.nan)
    nodes = np.full(proxy.shape[1], -1)
    anyhit = hit.any(axis=2)
    for b in np.flatnonzero(anyhit.any(axis=0)):
        t = int(np.argmax(anyhit[:, b]))
        times[b] = (t + start) / RC_HZ
        nodes[b] = int(np.argmax(hit[t, b]))
    return times, nodes
