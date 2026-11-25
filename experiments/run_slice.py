"""Stage 2: Bayesian MRF segmentation of ONE slice of a held-out BRATS patient.

    python experiments/run_slice.py HG 0001
    python experiments/run_slice.py HG 0001 --beta 0.7 --n-train 10 --chains 4 --burn 100 --samples 200
    python experiments/run_slice.py HG 0001 --beta 0            # no spatial prior (baseline only)

What it does
  1. Fits the class intensity models (a Gaussian per tumour label, a mixture for healthy tissue)
     and the label frequencies on OTHER, labelled patients.
  2. Builds the per-pixel log-likelihoods for the test slice.
  3. Baseline: beta = 0, i.e. each pixel on its own (exact softmax marginals, no sampling).
  4. Gibbs sampling of the Potts-MRF posterior from several over-dispersed starts.
  5. Posterior marginals -> MPM segmentation (most probable label per pixel), entropy map.
  6. Reports Dice (whole / core / enhancing) for baseline vs MRF, convergence diagnostics
     (split R-hat, effective sample size), credible interval of the tumour area, and the
     error rate per entropy bin (is the uncertainty informative?).

Caveats printed at the end: one slice chosen for its large tumour is an optimistic test; and
choose beta on validation patients, never on the patient you report.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import patient_dir
from src.data_io import best_slice, list_patients, load_patient, region_masks
from src.diagnostics import effective_sample_size, split_rhat
from src.metrics import dice, entropy, error_vs_entropy
from src.mrf2d import (DEFAULT_REGIONS, fit_class_models, loglik_image, normalise_volume,
                       run_gibbs, softmax_marginals)

K = 5


def region_dice(pred_labels, truth, mask):
    """Dice of the nested BRATS regions between a predicted and the expert label image."""
    out = {}
    for name, labs in DEFAULT_REGIONS.items():
        out[name] = dice(np.isin(pred_labels, labs) & mask, np.isin(truth, labs) & mask)
    return out


def get_params(grade, test_pid, n_train, norm, M, refit, cache_dir):
    """Class models fitted on other patients of the same grade (cached on disk as .npz)."""
    ids = [p for p in list_patients(grade) if p != test_pid][:n_train]
    cache = cache_dir / f"models_{grade}_{norm}_M{M}_n{len(ids)}_excl{test_pid}.npz"
    if cache.exists() and not refit:
        d = np.load(cache)
        if list(d["ids"]) == ids:
            print(f"using cached class models ({cache.name})")
            models = [(d[f"w{k}"], d[f"mu{k}"], d[f"S{k}"]) for k in range(K)]
            return models, d["pi"], ids
    print(f"fitting class models ({norm} normalisation, healthy mixture M={M}) on "
          f"{len(ids)} patients: {ids}")
    t = time.time()
    models, pi = fit_class_models([patient_dir(grade, p) for p in ids], norm=norm, K=K,
                                  healthy_components=M)
    print(f"  done in {time.time() - t:.0f}s")
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"pi": pi, "ids": np.array(ids)}
    for k, (w, mu, S) in enumerate(models):
        arrays.update({f"w{k}": w, f"mu{k}": mu, f"S{k}": S})
    np.savez(cache, **arrays)
    return models, pi, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grade", choices=["HG", "LG"])
    ap.add_argument("pid", help="TEST patient id, e.g. 0001")
    ap.add_argument("--z", type=int, default=None, help="slice (default: largest tumour)")
    ap.add_argument("--beta", type=float, default=0.7, help="Potts smoothness (K=5 critical value ~1.17)")
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--norm", choices=["zscore", "robust"], default="zscore")
    ap.add_argument("--healthy-components", type=int, default=1,
                    help="Gaussians for the healthy class (see experiments/eval_classifier.py)")
    ap.add_argument("--prior-power", type=float, default=1.0,
                    help="class prior strength: 1 = Bayes-consistent, <1 weaker")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--burn", type=int, default=100)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit", action="store_true", help="ignore the cached class parameters")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(args.seed)

    # ---- 1. class parameters from other patients ---------------------------
    models, pi, train_ids = get_params(args.grade, args.pid, args.n_train, args.norm,
                                       args.healthy_components, args.refit, root / "results")
    print("label frequencies in the brain (0=healthy 1=necrosis 2=edema 3=non-enh 4=enhancing):",
          np.round(pi, 4).tolist())

    # ---- 2. test slice ------------------------------------------------------
    vol, truth, _ = load_patient(patient_dir(args.grade, args.pid))
    vol, brain3 = normalise_volume(vol, args.norm)
    z = args.z if args.z is not None else best_slice(truth > 0)
    Y, mask, truth_z = vol[z], brain3[z], truth[z]
    print(f"test patient {args.grade}/{args.pid}, slice {z}: {int(mask.sum())} brain pixels, "
          f"{int((truth_z > 0).sum())} tumour pixels")
    ll = loglik_image(Y, mask, models, pi, args.prior_power)

    # ---- 3. baseline: no spatial prior -------------------------------------
    q0 = softmax_marginals(ll, mask)
    lab0 = np.where(mask, q0.argmax(axis=0), -1)
    d0 = region_dice(lab0, truth_z, mask)

    # ---- 4. Gibbs sampling --------------------------------------------------
    if args.beta > 0:
        t = time.time()
        out = run_gibbs(ll, args.beta, mask, rng, n_chains=args.chains, burn=args.burn,
                        samples=args.samples)
        print(f"Gibbs: {args.chains} chains x ({args.burn} burn-in + {args.samples} kept sweeps) "
              f"in {time.time() - t:.0f}s")
        q = out["q"]
    else:
        out, q = None, q0
    labels = np.where(mask, q.argmax(axis=0), -1)       # MPM: most probable label per pixel
    d1 = region_dice(labels, truth_z, mask)

    # ---- 5. report ----------------------------------------------------------
    print(f"\nDice on slice {z} (beta = {args.beta})")
    print(f"{'region':>10} {'no prior (beta=0)':>18} {'MRF, MPM':>10}")
    for name in DEFAULT_REGIONS:
        print(f"{name:>10} {d0[name]:>18.3f} {d1[name]:>10.3f}")

    if out is not None:
        print(f"\nconvergence (area traces, {args.chains} chains from different starts)")
        print(f"{'region':>10} {'split R-hat':>12} {'ESS':>8} {'area mean':>10} {'95% interval':>16} {'truth':>7}")
        for name, labs in DEFAULT_REGIONS.items():
            a = out["areas"][name]
            lo, hi = np.percentile(a, [2.5, 97.5])
            true_area = int((np.isin(truth_z, labs) & mask).sum())
            print(f"{name:>10} {split_rhat(a):>12.3f} {effective_sample_size(a):>8.0f} "
                  f"{a.mean():>10.0f} {f'[{lo:.0f}, {hi:.0f}]':>16} {true_area:>7d}")
        print("  (R-hat close to 1, e.g. < 1.05, means the chains agree; ESS is the number of "
              "independent-equivalent samples out of "
              f"{args.chains * args.samples})")

    print("\nIs the uncertainty informative?  error rate of the MPM label per entropy bin")
    print(f"{'bin':>4} {'mean entropy':>13} {'error rate':>11} {'pixels':>8}")
    for i, (h, e, n) in enumerate(error_vs_entropy(q, labels, truth_z, mask), 1):
        print(f"{i:>4} {h:>13.3f} {e:>11.3f} {n:>8d}")

    # ---- 6. figure ----------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    show = lambda a, img, title, **kw: (a.imshow(img, **kw), a.set_title(title), a.axis("off"))
    flair = np.where(mask, Y[..., 0], np.nan)
    show(ax[0, 0], flair, "FLAIR + expert whole-tumour outline", cmap="gray")
    ax[0, 0].contour(truth_z > 0, levels=[0.5], colors="r", linewidths=0.8)
    show(ax[0, 1], np.where(mask, truth_z, np.nan), "expert labels", cmap="viridis", vmin=0, vmax=4)
    show(ax[0, 2], np.where(mask, labels, np.nan), f"MPM labels (beta={args.beta})",
         cmap="viridis", vmin=0, vmax=4)
    im = ax[1, 0].imshow(np.where(mask, entropy(q), np.nan), cmap="magma")
    ax[1, 0].set_title("posterior entropy (uncertainty)"); ax[1, 0].axis("off")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)
    im = ax[1, 1].imshow(np.where(mask, 1 - q[0], np.nan), cmap="hot", vmin=0, vmax=1)
    ax[1, 1].set_title("P(tumour) = 1 - P(healthy)"); ax[1, 1].axis("off")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    if out is not None:
        for c, tr in enumerate(out["areas"]["whole"]):
            ax[1, 2].plot(tr, lw=0.8, label=f"chain {c}")
        ax[1, 2].axhline((truth_z > 0)[mask].sum(), color="k", ls="--", label="expert")
        ax[1, 2].set_title("whole-tumour area per sample"); ax[1, 2].set_xlabel("kept sweep")
        ax[1, 2].legend(fontsize=7)
    else:
        ax[1, 2].axis("off")
    fig.suptitle(f"{args.grade} {args.pid}, slice {z}")
    fig.tight_layout()
    fdir = root / "results" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    path = fdir / f"slice_{args.grade}_{args.pid}_z{z}_beta{args.beta}.png"
    fig.savefig(path, dpi=130)
    print("\nsaved", path)
    print("\nCaveats: (1) this slice was picked for its large tumour, so it is an optimistic test;"
          "\n         (2) choose beta on validation patients, not on the one you report;"
          "\n         (3) with --healthy-components 1 the healthy class is one Gaussian, which is "
          "crude for a mix of grey matter, white matter and CSF.")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
