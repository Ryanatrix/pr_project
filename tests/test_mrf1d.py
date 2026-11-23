"""Brute-force checks of src/mrf1d.py on a chain small enough to enumerate.

Run from the repo root:   python tests/test_mrf1d.py     (or: python -m pytest tests)
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mrf1d import (exact_marginals, exact_samples, empirical_marginals,
                       gibbs_marginals_by_burnin)


def brute_force_marginals(loglik, beta):
    """Sum over all K^V label images: the 'impossible' sum, feasible for tiny V."""
    V, K = loglik.shape
    q = np.zeros((V, K))
    Z = 0.0
    for x in itertools.product(range(K), repeat=V):
        score = sum(loglik[v, x[v]] for v in range(V))
        score += beta * sum(x[v] == x[v + 1] for v in range(V - 1))
        w = np.exp(score)
        Z += w
        for v in range(V):
            q[v, x[v]] += w
    return q / Z


def _problem(V=8, K=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(V, K)), 0.9, rng


def test_exact_marginals_match_brute_force():
    loglik, beta, _ = _problem()
    assert np.allclose(exact_marginals(loglik, beta),
                       brute_force_marginals(loglik, beta), atol=1e-12)


def test_exact_samples_have_the_right_marginals():
    loglik, beta, rng = _problem()
    xs = exact_samples(loglik, beta, 200_000, rng)
    q_hat = empirical_marginals(xs, loglik.shape[1])
    assert np.abs(q_hat - brute_force_marginals(loglik, beta)).max() < 0.01


def test_gibbs_converges_to_the_exact_marginals():
    loglik, beta, rng = _problem()
    out = gibbs_marginals_by_burnin(loglik, beta, [200], 50_000, rng, init="zeros")
    assert np.abs(out[200] - brute_force_marginals(loglik, beta)).max() < 0.02


if __name__ == "__main__":
    test_exact_marginals_match_brute_force();       print("exact marginals   OK")
    test_exact_samples_have_the_right_marginals();  print("exact samples     OK")
    test_gibbs_converges_to_the_exact_marginals();  print("gibbs converges   OK")
