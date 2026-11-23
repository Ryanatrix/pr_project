"""Toy experiment: how wrong is Gibbs sampling after a given burn-in?

Mirrors the 26-voxel simulation of Awate et al. (MedIA 2019, Sec. 5.1, Fig. 1e-g):
a binary 1D 'image' with an intensity edge plus noise, small enough that the exact
posterior is known.  Here the exact answer comes from forward-backward (src/mrf1d.py).

For many noisy images and many burn-in lengths we measure

    error(b) = mean over voxels of | P_gibbs(x_v = 1) - P_exact(x_v = 1) |

where P_gibbs is estimated from n_chains independent chains stopped after b sweeps.
The dashed 'noise floor' is the error of n_chains *exact* independent samples: the
best any sampler of that size can do, because of Monte Carlo noise alone.

Run from the repo root:
    python experiments/toy_1d.py                 (defaults)
    python experiments/toy_1d.py --beta 1.5 --show
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mrf1d import (empirical_marginals, exact_marginals, exact_samples,
                       gibbs_marginals_by_burnin, make_edge_problem)

BURNINS = [1, 2, 3, 5, 10, 20, 50, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxels", type=int, default=26)
    ap.add_argument("--beta", type=float, default=1.0, help="Potts smoothness")
    ap.add_argument("--sigma", type=float, default=0.6, help="noise std of the 1D image")
    ap.add_argument("--images", type=int, default=70, help="number of noisy images")
    ap.add_argument("--chains", type=int, default=2000, help="independent chains per image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="also open the figures")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    inits = ("random", "zeros")
    err = {m: np.zeros((args.images, len(BURNINS))) for m in inits}
    floor = np.zeros(args.images)
    example = {}

    for i in range(args.images):
        _, loglik, _ = make_edge_problem(args.voxels, args.sigma, rng)
        q = exact_marginals(loglik, args.beta)[:, 1]

        xs = exact_samples(loglik, args.beta, args.chains, rng)
        floor[i] = np.abs(empirical_marginals(xs, 2)[:, 1] - q).mean()

        for m in inits:
            out = gibbs_marginals_by_burnin(loglik, args.beta, BURNINS,
                                            args.chains, rng, init=m)
            for j, b in enumerate(BURNINS):
                err[m][i, j] = np.abs(out[b][:, 1] - q).mean()
            if i == 0 and m == "random":
                example = {"q": q, "gibbs": {b: out[b][:, 1] for b in (1, 5, 50)}}

    # ---- table -------------------------------------------------------------
    print(f"beta={args.beta}  sigma={args.sigma}  voxels={args.voxels}  "
          f"images={args.images}  chains={args.chains}")
    print(f"{'burn-in':>8} {'random init':>12} {'all-0 init':>12} {'noise floor':>12}")
    for j, b in enumerate(BURNINS):
        print(f"{b:>8} {err['random'][:, j].mean():>12.4f} "
              f"{err['zeros'][:, j].mean():>12.4f} {floor.mean():>12.4f}")

    # ---- figure 1: error vs burn-in ---------------------------------------
    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    for m, lab in (("random", "Gibbs, random start"), ("zeros", "Gibbs, all-0 start")):
        mean = err[m].mean(axis=0)
        se = err[m].std(axis=0, ddof=1) / np.sqrt(args.images)
        ax.errorbar(BURNINS, mean, yerr=se, marker="o", capsize=2, label=lab)
    ax.axhline(floor.mean(), ls="--", color="k", label="noise floor (exact samples)")
    ax.set_xscale("log")
    ax.set_xlabel("burn-in (sweeps)")
    ax.set_ylabel("mean |P_gibbs - P_exact|")
    ax.set_title(f"Gibbs error vs burn-in (beta={args.beta})")
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / "toy_burnin_error.png"
    fig.savefig(p1, dpi=150)

    # ---- figure 2: marginals for one image --------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    v = np.arange(args.voxels)
    ax2.plot(v, example["q"], "k-", lw=2, label="exact")
    for b, qb in example["gibbs"].items():
        ax2.plot(v, qb, "o--", ms=3, label=f"Gibbs, burn-in {b}")
    ax2.set_xlabel("voxel")
    ax2.set_ylabel("P(label = 1)")
    ax2.set_title("Posterior marginals, one noisy image")
    ax2.legend()
    fig2.tight_layout()
    p2 = out_dir / "toy_marginals.png"
    fig2.savefig(p2, dpi=150)

    print(f"saved {p1}\nsaved {p2}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
