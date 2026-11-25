"""Checks for the class-conditional intensity models in src/mrf2d.py.

Run from the repo root:   python tests/test_models.py     (or: python -m pytest tests)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal as mvn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mrf2d import fit_models, loglik_points


def _toy_voxels(rng, n=6000):
    """'Healthy' tissue = 3 well separated intensity populations; one 'tumour' class in between."""
    centres = np.array([[-2.5, 1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.5, -1.5, 0.0, 0.0]])
    healthy = np.concatenate([c + 0.35 * rng.normal(size=(n // 3, 4)) for c in centres])
    tumour = np.array([1.2, 1.2, 0.0, 0.0]) + np.array([1.2, 1.2, 0.35, 0.35]) * rng.normal(size=(n // 6, 4))  # broad, between the healthy populations
    Y = np.concatenate([healthy, tumour])
    t = np.concatenate([np.zeros(len(healthy), np.int8), np.ones(len(tumour), np.int8)])
    return Y, t


def test_single_gaussian_loglik_matches_scipy():
    rng = np.random.default_rng(0)
    Y, t = _toy_voxels(rng)
    models, pi = fit_models([(Y, t)], K=2, healthy_components=1)
    ll = loglik_points(Y[:200], models, pi)
    for k in range(2):
        w, mu, Sigma = models[k]
        ref = mvn.logpdf(Y[:200], mu[0], Sigma[0]) + np.log(pi[k])
        assert np.allclose(ll[k], ref, atol=1e-3)


def test_healthy_mixture_classifies_better_than_one_gaussian():
    rng = np.random.default_rng(1)
    Y, t = _toy_voxels(rng)
    recall = {}
    for M in (1, 3):
        models, pi = fit_models([(Y, t)], K=2, healthy_components=M)
        pred = loglik_points(Y, models, pi).argmax(axis=0)
        recall[M] = (pred[t == 1] == 1).mean()          # how many tumour voxels are found
    # one broad healthy Gaussian (times a large prior) swallows part of the tumour class
    assert recall[3] > recall[1] + 0.04, recall


if __name__ == "__main__":
    test_single_gaussian_loglik_matches_scipy();               print("loglik vs scipy               OK")
    test_healthy_mixture_classifies_better_than_one_gaussian(); print("healthy mixture beats 1 Gauss OK")
