"""Random-matrix-theory singular value shrinkage (Idea 1: "shrink, don't whiten").

Model: momentum matrix M = low-rank signal + iid noise. Estimate the noise level from the
median singular value (assumes >half the spectrum is noise bulk), detect spikes above the
Marchenko-Pastur bulk edge, and weight each singular direction by the Gavish-Donoho
Frobenius-optimal shrinker normalized to [0,1] (a confidence, not a magnitude):

    w_i = eta(y_i) / y_i,   eta(y) = sqrt((y^2 - beta - 1)^2 - 4*beta) / y   for y > 1 + sqrt(beta)

Update direction = U diag(w) V^T. Muon is the limit w == 1; zero update is the pure-noise limit.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=512)
def mp_median(beta: float) -> float:
    """Median eigenvalue of the Marchenko-Pastur distribution with ratio beta (var=1)."""
    beta = max(min(beta, 1.0), 1e-6)
    lo, hi = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
    xs = np.linspace(lo, hi, 20001)[1:-1]
    pdf = np.sqrt(np.maximum((hi - xs) * (xs - lo), 0)) / (2 * np.pi * beta * xs)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return float(xs[np.searchsorted(cdf, 0.5)])


def shrink_weights(s: np.ndarray, shape) -> np.ndarray:
    """Confidence weights in [0,1] for each singular value of a matrix with given shape."""
    m, n = min(shape), max(shape)
    beta = m / n
    med = float(np.median(s))
    if med <= 0:
        return np.zeros_like(s)
    sigma = med / np.sqrt(n * mp_median(round(beta, 4)))
    y = s / (sigma * np.sqrt(n) + 1e-12)
    edge = 1 + np.sqrt(beta)
    w = np.zeros_like(y)
    mask = y > edge + 1e-9
    t = y[mask]
    eta = np.sqrt(np.maximum((t ** 2 - beta - 1) ** 2 - 4 * beta, 0)) / t
    w[mask] = eta / t
    return np.clip(w, 0.0, 1.0)


def shrink_weights_torch(S, shape) -> "torch.Tensor":
    """Batched GPU version of shrink_weights. S: (..., m) descending singular values."""
    import math

    import torch
    m, n = min(shape), max(shape)
    beta = m / n
    mu = mp_median(round(beta, 4))
    med = S.median(dim=-1, keepdim=True).values
    sigma = med / math.sqrt(n * mu)
    y = S / (sigma * math.sqrt(n) + 1e-12)
    t2 = y * y - beta - 1
    eta = torch.sqrt((t2 * t2 - 4 * beta).clamp_min(0)) / y.clamp_min(1e-12)
    w = torch.where(y > 1 + math.sqrt(beta) + 1e-9, eta / y.clamp_min(1e-12),
                    torch.zeros_like(y))
    w = torch.where(med > 0, w, torch.zeros_like(w))
    return torch.nan_to_num(w, nan=0.0).clamp(0.0, 1.0)


def spectrum_stats(s: np.ndarray, shape) -> dict:
    """Free diagnostics from a singular spectrum (Idea 3: spectral recycling)."""
    fro2 = float((s ** 2).sum())
    top = float(s[0]) if len(s) else 0.0
    w = shrink_weights(s, shape)
    spike = w > 0
    return {
        "top_sv": top,
        "stable_rank": fro2 / (top ** 2 + 1e-12),
        "n_spikes": int(spike.sum()),
        "spike_energy": float((s[spike] ** 2).sum() / (fro2 + 1e-12)),
        "mean_w": float(w.mean()),
    }
