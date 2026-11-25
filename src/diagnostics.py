"""MCMC convergence diagnostics for a scalar summary (e.g. the tumour area per sample).

All functions take `chains` of shape (M, N): M independent chains, N draws each.
"""
from __future__ import annotations

import numpy as np


def split_rhat(chains: np.ndarray) -> float:
    """Split-R-hat (Gelman-Rubin).  Close to 1 (say < 1.05) means the chains agree.

    Each chain is cut in half so that a chain that drifts is also detected.
        W      = mean within-chain variance
        B / n  = variance of the chain means
        var+   = (n-1)/n * W + B/n
        R-hat  = sqrt(var+ / W)
    """
    chains = np.asarray(chains, dtype=float)
    M, N = chains.shape
    n = N // 2
    if n < 2:
        raise ValueError("need at least 4 draws per chain")
    halves = np.concatenate([chains[:, :n], chains[:, n:2 * n]], axis=0)   # (2M, n)
    W = halves.var(axis=1, ddof=1).mean()
    B_over_n = halves.mean(axis=1).var(ddof=1)
    if W == 0:
        return 1.0 if B_over_n == 0 else np.inf
    var_plus = (n - 1) / n * W + B_over_n
    return float(np.sqrt(var_plus / W))


def _autocorr(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of one series, lags 0..N-1, via FFT."""
    x = x - x.mean()
    N = len(x)
    f = np.fft.rfft(x, n=2 * N)
    ac = np.fft.irfft(f * np.conj(f))[:N]
    return ac / ac[0] if ac[0] > 0 else np.zeros(N)


def effective_sample_size(chains: np.ndarray) -> float:
    """Effective number of independent draws, pooled over chains.

        ESS = M N / (1 + 2 * sum_{t>=1} rho_t)

    rho_t is the autocorrelation at lag t, averaged over chains; the sum stops at
    the first pair of lags whose sum is negative (Geyer's initial positive sequence).
    """
    chains = np.asarray(chains, dtype=float)
    M, N = chains.shape
    rho = np.mean([_autocorr(c) for c in chains], axis=0)
    s = 0.0
    for t in range(1, N - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        s += pair
    return float(M * N / max(1.0 + 2.0 * s, 1e-12))
