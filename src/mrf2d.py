"""2D hidden Potts MRF for multimodal brain-tumour segmentation, sampled by Gibbs.

Model on one slice (pixels i inside the brain, labels x_i in {0..K-1}, observations y_i in R^4):

    Q(x) ∝ prod_i  pi_{x_i} N(y_i ; mu_{x_i}, Sigma_{x_i})  *  exp( beta * sum_{i~j} 1[x_i = x_j] )

 - N(.) are class-conditional Gaussians on the 4 z-scored modalities,
 - pi_k is a class prior (a unary term; the paper's Eq. 3 has none, we add it because
   healthy tissue is ~90% of the brain and the classes are very unbalanced),
 - beta is the Potts smoothness (kept FIXED, like the paper; study it by running several values),
 - i~j are 4-neighbours inside the brain.

Gibbs conditional for one pixel (what the sampler implements):

    P(x_i = k | rest) ∝ exp( log pi_k + log N(y_i; mu_k, Sigma_k) + beta * n_k(i) ),
    n_k(i) = number of 4-neighbours currently labelled k.

Checkerboard trick: pixels of the same colour (i+j even / odd) are never neighbours, so they
are conditionally independent given the other colour and can be updated all at once.  That is
still an exact Gibbs sweep, and it makes 2D sampling fast in plain numpy.

Arrays with a label axis use the layout (K, H, W).  Pixels outside the brain hold label -1.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from .data_io import load_patient

DEFAULT_REGIONS = {"whole": (1, 2, 3, 4), "core": (1, 3, 4), "enhancing": (4,)}


# ----------------------------------------------------------------------------
# preprocessing at volume level and parameter fitting
# ----------------------------------------------------------------------------
def normalise_volume(vol: np.ndarray):
    """z-score each modality over the whole 3D brain (voxels > 0 in any modality).

    Normalising per VOLUME (not per slice) keeps intensities comparable between slices and
    between patients, which is what makes class Gaussians fitted on other patients usable.
    Returns (vol_normalised float32, brain 3D bool); voxels outside the brain are set to 0.
    """
    brain = (vol > 0).any(axis=-1)
    out = np.zeros_like(vol, dtype=np.float32)
    for c in range(vol.shape[-1]):
        v = vol[..., c][brain]
        sd = float(v.std())
        out[..., c][brain] = (v - v.mean()) / (sd if sd > 0 else 1.0)
    return out, brain


def fit_class_gaussians(pdirs, K: int = 5, ridge: float = 1e-3, verbose: bool = True):
    """Fit a Gaussian per label (and the label frequencies) from LABELLED training patients.

    Uses running sums, so only one patient is in memory at a time.
    Returns mu (K, C), Sigma (K, C, C) and pi (K,).  Test patients must NOT be in `pdirs`.
    """
    n = s = S = None
    for pdir in pdirs:
        vol, truth, _ = load_patient(pdir)
        vol, brain = normalise_volume(vol)
        C = vol.shape[-1]
        if n is None:
            n, s, S = np.zeros(K), np.zeros((K, C)), np.zeros((K, C, C))
        for k in range(K):
            Yk = vol[brain & (truth == k)].astype(np.float64)
            n[k] += len(Yk)
            s[k] += Yk.sum(axis=0)
            S[k] += Yk.T @ Yk
        if verbose:
            print(f"  fitted from {pdir.name}")
    if np.any(n == 0):
        raise ValueError(f"labels {np.flatnonzero(n == 0).tolist()} never occur in the training patients")
    mu = s / n[:, None]
    Sigma = S / n[:, None, None] - mu[:, :, None] * mu[:, None, :] + ridge * np.eye(mu.shape[1])
    return mu, Sigma, n / n.sum()


def loglik_image(Y: np.ndarray, mask: np.ndarray, mu, Sigma, pi) -> np.ndarray:
    """Per-pixel, per-label log( pi_k * N(y; mu_k, Sigma_k) ), shape (K, H, W), float32.

    Y is (H, W, C); pixels outside `mask` get 0 (never used).
    """
    K, C = mu.shape
    H, W = mask.shape
    ll = np.zeros((K, H, W), dtype=np.float32)
    Ym = Y[mask].astype(np.float64)
    for k in range(K):
        L = np.linalg.cholesky(Sigma[k])
        z = solve_triangular(L, (Ym - mu[k]).T, lower=True)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        ll[k][mask] = (-0.5 * ((z ** 2).sum(axis=0) + logdet + C * np.log(2 * np.pi))
                       + np.log(pi[k])).astype(np.float32)
    return ll


def softmax_marginals(ll: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exact posterior marginals when beta = 0 (no spatial prior): a softmax over labels."""
    q = np.exp(ll - ll.max(axis=0, keepdims=True))
    q /= q.sum(axis=0, keepdims=True)
    q[:, ~mask] = 0.0
    return q


# ----------------------------------------------------------------------------
# Gibbs sampling
# ----------------------------------------------------------------------------
def neighbour_counts(onehot: np.ndarray) -> np.ndarray:
    """n_k(i): for each label k and pixel i, how many of its 4 neighbours carry label k."""
    n = np.zeros_like(onehot)
    n[:, 1:, :] += onehot[:, :-1, :]
    n[:, :-1, :] += onehot[:, 1:, :]
    n[:, :, 1:] += onehot[:, :, :-1]
    n[:, :, :-1] += onehot[:, :, 1:]
    return n


def _parity_masks(mask: np.ndarray):
    ii, jj = np.indices(mask.shape)
    return [((ii + jj) % 2 == p) & mask for p in (0, 1)]


def gibbs_sweep(x: np.ndarray, ll: np.ndarray, beta: float, parities, rng) -> np.ndarray:
    """One exact Gibbs sweep (both checkerboard colours), in place.

    Sampling from a softmax is done with the Gumbel-max trick:
    argmax_k (logit_k + Gumbel noise) has probability softmax(logit)_k.
    """
    K = ll.shape[0]
    for par in parities:
        onehot = np.stack([(x == k) for k in range(K)]).astype(np.float32)   # all-0 outside brain
        logits = ll + beta * neighbour_counts(onehot)
        new = np.argmax(logits + rng.gumbel(size=logits.shape).astype(np.float32), axis=0)
        x[par] = new[par]
    return x


def run_gibbs(ll: np.ndarray, beta: float, mask: np.ndarray, rng, n_chains: int = 4,
              burn: int = 100, samples: int = 200, thin: int = 1, regions=None):
    """Run several chains from over-dispersed starts and accumulate posterior summaries.

    Chain 0 starts at the per-pixel maximum-likelihood labels, the others at uniformly random
    labels.  (Different starts are what make R-hat a meaningful convergence check.)

    Returns a dict:
        q       (K, H, W)  posterior marginals from all kept samples of all chains
        areas   {region name: (n_chains, samples) pixel counts per kept sample}
        last    the final label image of the last chain (for a quick look)
    """
    regions = DEFAULT_REGIONS if regions is None else regions
    K = ll.shape[0]
    parities = _parity_masks(mask)
    counts = np.zeros(ll.shape, dtype=np.float64)
    areas = {name: np.zeros((n_chains, samples)) for name in regions}

    for c in range(n_chains):
        init = np.argmax(ll, axis=0) if c == 0 else rng.integers(K, size=mask.shape)
        x = np.where(mask, init, -1).astype(np.int16)
        for _ in range(burn):
            gibbs_sweep(x, ll, beta, parities, rng)
        for s in range(samples):
            for _ in range(thin):
                gibbs_sweep(x, ll, beta, parities, rng)
            for k in range(K):
                counts[k] += (x == k)
            for name, labs in regions.items():
                areas[name][c, s] = np.isin(x, labs).sum()
    return {"q": counts / (n_chains * samples), "areas": areas, "last": x}
