"""Segmentation and uncertainty metrics.

q arrays are posterior marginals with the label axis FIRST: shape (K, H, W).
"""
from __future__ import annotations

import numpy as np


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice = 2|A ∩ B| / (|A| + |B|); defined as 1 when both masks are empty."""
    a, b = a.astype(bool), b.astype(bool)
    tot = a.sum() + b.sum()
    return 1.0 if tot == 0 else float(2.0 * (a & b).sum() / tot)


def entropy(q: np.ndarray) -> np.ndarray:
    """Shannon entropy of the per-pixel label distribution, in nats: -sum_k q_k log q_k."""
    return -(q * np.log(np.clip(q, 1e-12, 1.0))).sum(axis=0)


def unalikeability(q: np.ndarray) -> np.ndarray:
    """1 - sum_k q_k^2 : the probability that two independent draws disagree
    (the quantity Awate et al. take the square root of as their uncertainty)."""
    return 1.0 - (q ** 2).sum(axis=0)


def error_vs_entropy(q: np.ndarray, labels: np.ndarray, truth: np.ndarray,
                     mask: np.ndarray, n_bins: int = 5):
    """Is the model's uncertainty informative?  Bin brain pixels by entropy (equal-count
    bins) and report, per bin, the fraction of pixels whose MPM label is wrong.

    A useful uncertainty map has an error rate that rises from the first bin to the last.
    Returns a list of (mean_entropy, error_rate, n_pixels).
    """
    h = entropy(q)[mask]
    wrong = (labels[mask] != truth[mask]).astype(float)
    order = np.argsort(h)
    rows = []
    for idx in np.array_split(order, n_bins):
        rows.append((float(h[idx].mean()), float(wrong[idx].mean()), int(len(idx))))
    return rows
