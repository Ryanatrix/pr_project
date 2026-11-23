"""1D Potts MRF with Gaussian likelihoods: exact inference vs Gibbs sampling.

A chain of V voxels, K labels.  Model (same as the paper's Eq. 3 with w_ij = beta):

    Q(x) ∝ exp( sum_v loglik[v, x_v]  +  beta * sum_v 1[x_v == x_{v+1}] )

Because the graph is a chain, the exact posterior marginals come from
forward-backward in O(V K^2), and we can even draw *exact independent samples*
(forward filtering, backward sampling).  That gives a ground truth to measure how
far a Gibbs sampler is from the true posterior after a given burn-in, which is the
same idea as the 26-voxel experiment in Awate et al. (MedIA 2019, Sec. 5.1).

Everything is vectorised over many independent chains, so thousands of chains
run in parallel with plain numpy.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _pair_matrix(beta: float, K: int) -> np.ndarray:
    """psi[a, b] = exp(beta * 1[a == b]), the Potts pairwise factor."""
    return np.exp(beta * np.eye(K))


def _stable_exp(loglik: np.ndarray) -> np.ndarray:
    """exp(loglik) with each row shifted by its max (avoids underflow)."""
    return np.exp(loglik - loglik.max(axis=1, keepdims=True))


def _draw(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one label per row of p (n, K) by inverse-CDF sampling."""
    u = rng.random((p.shape[0], 1))
    idx = (u > np.cumsum(p, axis=1)).sum(axis=1)
    return np.minimum(idx, p.shape[1] - 1)


def empirical_marginals(x: np.ndarray, K: int) -> np.ndarray:
    """x: (n_samples, V) integer labels  ->  (V, K) frequencies."""
    return np.stack([(x == k).mean(axis=0) for k in range(K)], axis=-1)


# ----------------------------------------------------------------------------
# exact inference
# ----------------------------------------------------------------------------
def _forward(loglik: np.ndarray, beta: float) -> np.ndarray:
    """Normalised forward messages alpha[v, k] = P(x_v = k | y_0..y_v)."""
    V, K = loglik.shape
    A = _pair_matrix(beta, K)
    a = _stable_exp(loglik)
    alpha = np.empty((V, K))
    alpha[0] = a[0] / a[0].sum()
    for v in range(1, V):
        t = a[v] * (alpha[v - 1] @ A)
        alpha[v] = t / t.sum()
    return alpha


def exact_marginals(loglik: np.ndarray, beta: float) -> np.ndarray:
    """Exact posterior marginals q[v, k] = Q(x_v = k) by forward-backward."""
    V, K = loglik.shape
    A = _pair_matrix(beta, K)
    a = _stable_exp(loglik)
    alpha = _forward(loglik, beta)
    bwd = np.ones((V, K))
    for v in range(V - 2, -1, -1):
        t = A @ (a[v + 1] * bwd[v + 1])
        bwd[v] = t / t.sum()
    q = alpha * bwd
    return q / q.sum(axis=1, keepdims=True)


def exact_samples(loglik: np.ndarray, beta: float, n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """n exact, independent draws from Q (forward filtering, backward sampling).

    For a chain this is a genuine *perfect sampler*: no burn-in, no bias.
    """
    V, K = loglik.shape
    A = _pair_matrix(beta, K)
    alpha = _forward(loglik, beta)
    x = np.empty((n, V), dtype=np.int64)
    x[:, V - 1] = _draw(np.tile(alpha[V - 1], (n, 1)), rng)
    for v in range(V - 2, -1, -1):
        # P(x_v = k | x_{v+1}) ∝ alpha_v(k) * psi(k, x_{v+1})
        p = alpha[v][None, :] * A[:, x[:, v + 1]].T
        p /= p.sum(axis=1, keepdims=True)
        x[:, v] = _draw(p, rng)
    return x


# ----------------------------------------------------------------------------
# Gibbs sampling
# ----------------------------------------------------------------------------
def gibbs_sweep(x: np.ndarray, loglik: np.ndarray, beta: float,
                rng: np.random.Generator) -> np.ndarray:
    """One systematic-scan Gibbs sweep, in place, for all chains at once.

    x: (C, V) current labels of C independent chains.
    Full conditional (the formula you should be able to derive on paper):

        P(x_v = k | rest) ∝ exp( loglik[v, k] + beta * #{neighbours of v with label k} )
    """
    C, V = x.shape
    rows = np.arange(C)
    for v in range(V):
        logits = np.tile(loglik[v], (C, 1))
        if v > 0:
            logits[rows, x[:, v - 1]] += beta
        if v < V - 1:
            logits[rows, x[:, v + 1]] += beta
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        x[:, v] = _draw(p, rng)
    return x


def gibbs_marginals_by_burnin(loglik: np.ndarray, beta: float, burnins,
                              n_chains: int, rng: np.random.Generator,
                              init: str = "random") -> dict:
    """Run n_chains independent Gibbs chains and, for every burn-in b in `burnins`,
    return the empirical marginals of the chains' states after b sweeps.

    init = "random": every voxel uniform over labels (as in the paper, Sec. 5).
    init = "zeros" : every chain starts from the all-label-0 image (a deliberately
                     bad start, to make the burn-in bias visible).
    """
    V, K = loglik.shape
    if init == "random":
        x = rng.integers(K, size=(n_chains, V))
    elif init == "zeros":
        x = np.zeros((n_chains, V), dtype=np.int64)
    else:
        raise ValueError("init must be 'random' or 'zeros'")
    out, done = {}, 0
    for b in sorted(burnins):
        while done < b:
            gibbs_sweep(x, loglik, beta, rng)
            done += 1
        out[b] = empirical_marginals(x, K)
    return out


# ----------------------------------------------------------------------------
# synthetic test problem (like Fig. 1e of the paper: an edge plus noise)
# ----------------------------------------------------------------------------
def make_edge_problem(V: int, sigma: float, rng: np.random.Generator,
                      mu=(0.0, 1.0)):
    """A 1D 'image' with an intensity edge in the middle and Gaussian noise.

    Returns y (V,), loglik (V, K=2) and the true labels (V,).
    """
    mu = np.asarray(mu, dtype=float)
    labels = (np.arange(V) >= V // 2).astype(int)
    y = mu[labels] + sigma * rng.standard_normal(V)
    loglik = -((y[:, None] - mu[None, :]) ** 2) / (2.0 * sigma ** 2)
    return y, loglik, labels
