"""Checks for the 2D sampler and the diagnostics.

Run from the repo root:   python tests/test_mrf2d.py     (or: python -m pytest tests)
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.diagnostics import effective_sample_size, split_rhat
from src.metrics import dice, entropy
from src.mrf2d import run_gibbs


def brute_force_marginals_2d(ll, beta):
    """Exact marginals on a tiny grid by summing over all K^(H*W) label images."""
    K, H, W = ll.shape
    edges = [((i, j), (i, j + 1)) for i in range(H) for j in range(W - 1)] + \
            [((i, j), (i + 1, j)) for i in range(H - 1) for j in range(W)]
    q = np.zeros((K, H, W))
    Z = 0.0
    for lab in itertools.product(range(K), repeat=H * W):
        x = np.array(lab).reshape(H, W)
        score = sum(ll[x[i, j], i, j] for i in range(H) for j in range(W))
        score += beta * sum(x[a] == x[b] for a, b in edges)
        w = np.exp(score)
        Z += w
        for k in range(K):
            q[k] += w * (x == k)
    return q / Z


def test_gibbs_2d_matches_brute_force_on_a_3x3_grid():
    rng = np.random.default_rng(1)
    ll = rng.normal(size=(3, 3, 3)).astype(np.float32)      # K=3 labels on a 3x3 grid
    mask = np.ones((3, 3), bool)
    beta = 0.8
    out = run_gibbs(ll, beta, mask, rng, n_chains=8, burn=50, samples=4000)
    exact = brute_force_marginals_2d(ll.astype(np.float64), beta)
    assert np.abs(out["q"] - exact).max() < 0.02, np.abs(out["q"] - exact).max()


def test_split_rhat_flags_chains_that_disagree():
    rng = np.random.default_rng(0)
    good = rng.normal(size=(4, 400))
    bad = good + np.array([[0.0], [0.0], [3.0], [3.0]])      # two chains stuck elsewhere
    assert split_rhat(good) < 1.05
    assert split_rhat(bad) > 1.5


def test_effective_sample_size_of_an_ar1_chain():
    rho, N, M = 0.9, 4000, 4
    rng = np.random.default_rng(0)
    x = np.zeros((M, N))
    for t in range(1, N):
        x[:, t] = rho * x[:, t - 1] + rng.normal(size=M)
    theory = M * N * (1 - rho) / (1 + rho)                    # about 0.0526 * M * N
    ess = effective_sample_size(x)
    assert 0.6 * theory < ess < 1.6 * theory, (ess, theory)


def test_metrics():
    a = np.array([[1, 1], [0, 0]], bool)
    assert dice(a, a) == 1.0 and dice(a, ~a) == 0.0 and dice(~a & ~a & False, a & False) == 1.0
    uniform = np.full((4, 2, 2), 0.25)
    assert np.allclose(entropy(uniform), np.log(4))


if __name__ == "__main__":
    test_gibbs_2d_matches_brute_force_on_a_3x3_grid(); print("2D gibbs vs brute force  OK")
    test_split_rhat_flags_chains_that_disagree();      print("split R-hat              OK")
    test_effective_sample_size_of_an_ar1_chain();      print("ESS on AR(1)             OK")
    test_metrics();                                    print("metrics                  OK")
