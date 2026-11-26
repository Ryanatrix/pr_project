"""Does the spatial prior help, and which smoothness beta is best?

Same patient split as experiments/eval_classifier.py (first n-train patients train the class
models, the next n-val validate, the next n-test test).  For each patient several tumour slices
are segmented, for every beta in the grid, with the region-wise rule  P(region) > 0.5  applied to
the Gibbs posterior marginals.  beta = 0 is the baseline without any spatial prior (exact softmax).

    beta is chosen on the VALIDATION patients only; the test patients are then used once,
    to compare beta = 0 against the chosen beta.

Use the likelihood settings printed by eval_classifier.py, e.g.

    python experiments/eval_beta.py HG --norm robust --healthy-components 4 --prior-power 1.0
    python experiments/eval_beta.py HG --betas 0 0.3 0.6 0.9 --slices 4 --chains 2 --burn 50 --samples 100

The K = 5 Potts model has a critical smoothness ln(1 + sqrt(5)) ~ 1.17; keep the grid below it.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import patient_dir
from src.data_io import list_patients, load_patient
from src.mrf2d import (DEFAULT_REGIONS, fit_class_models, loglik_image, normalise_volume,
                       region_predictions, run_gibbs, softmax_marginals)


def pick_slices(truth, n_slices, min_pixels=300):
    """Up to n_slices evenly spaced slices (axis 0) whose whole-tumour area is >= min_pixels (skips the tumour's thin edge slices)."""
    area = (truth > 0).reshape(truth.shape[0], -1).sum(axis=1)
    cand = np.flatnonzero(area >= min_pixels)
    if len(cand) == 0:
        return []
    idx = np.unique(np.linspace(0, len(cand) - 1, n_slices).round().astype(int))
    return cand[idx].tolist()


def slice_counts(q, truth_z, mask):
    """Per region: [overlap, predicted size, expert size] in pixels, for one slice."""
    preds = region_predictions(q, mask)
    out = []
    for n, labs in DEFAULT_REGIONS.items():
        t = np.isin(truth_z, labs) & mask
        out.append([(preds[n] & t).sum(), preds[n].sum(), t.sum()])
    return np.array(out, dtype=float)                          # (3 regions, 3 counts)


def evaluate_patient(pid, grade, models, pi, args, betas, rng):
    """Dice [whole, core, enh] for each beta, POOLED over the patient's chosen slices:
    2 * sum(overlap) / (sum(predicted) + sum(expert)).  (Averaging per-slice Dice would be
    dominated by slices at the edge of the tumour, where a region can be empty or tiny.)"""
    vol, truth, _ = load_patient(patient_dir(grade, pid))
    vol, brain = normalise_volume(vol, args.norm)
    zs = pick_slices(truth, args.slices)
    acc = np.zeros((len(betas), 3, 3))
    for z in zs:
        mask, truth_z = brain[z], truth[z]
        ll = loglik_image(vol[z], mask, models, pi, args.prior_power)
        for i, b in enumerate(betas):
            if b == 0:
                q = softmax_marginals(ll, mask)
            else:
                q = run_gibbs(ll, b, mask, rng, n_chains=args.chains, burn=args.burn,
                              samples=args.samples)["q"]
            acc[i] += slice_counts(q, truth_z, mask)
    tp, pr, tr = acc[..., 0], acc[..., 1], acc[..., 2]
    d = np.where(pr + tr == 0, 1.0, 2.0 * tp / np.maximum(pr + tr, 1.0))
    return d, len(zs)                                           # (n_betas, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grade", choices=["HG", "LG"])
    ap.add_argument("--norm", choices=["zscore", "robust"], default="robust")
    ap.add_argument("--healthy-components", type=int, default=4)
    ap.add_argument("--prior-power", type=float, default=1.0)
    ap.add_argument("--betas", type=float, nargs="+", default=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--slices", type=int, default=5, help="tumour slices per patient")
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--burn", type=int, default=50)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    betas = sorted(set(args.betas) | {0.0})
    ids = list_patients(args.grade)
    need = args.n_train + args.n_val + args.n_test
    if need > len(ids):
        sys.exit(f"need {need} patients but {args.grade} has {len(ids)}")
    train = ids[:args.n_train]
    val = ids[args.n_train:args.n_train + args.n_val]
    test = ids[args.n_train + args.n_val:need]
    print(f"train {train}\nvalidation {val}\ntest {test}  (disjoint)")
    print(f"likelihood: {args.norm} normalisation, healthy mixture M={args.healthy_components}, "
          f"prior^{args.prior_power};  betas {betas}\n")

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print("fitting class models on the training patients ...")
    models, pi = fit_class_models([patient_dir(args.grade, p) for p in train], norm=args.norm,
                                  healthy_components=args.healthy_components, verbose=False)

    print(f"validation patients ({args.slices} tumour slices each, {len(betas)} betas) ...")
    val_rows = []
    for p in val:
        r, n = evaluate_patient(p, args.grade, models, pi, args, betas, rng)
        val_rows.append(r)
        print(f"  {p}: {n} slices done ({time.time() - t0:.0f}s elapsed)")
    val_mean = np.mean(val_rows, axis=0)                          # (n_betas, 3)

    print(f"\nVALIDATION Dice, mean over {len(val)} patients (patient-level Dice pooled over the chosen slices)")
    print(f"{'beta':>6} {'whole':>7} {'core':>7} {'enh.':>7} {'mean':>7} {'vs beta=0':>10}")
    base = val_mean[betas.index(0.0)].mean()
    for b, r in zip(betas, val_mean):
        print(f"{b:>6.2f} {r[0]:>7.3f} {r[1]:>7.3f} {r[2]:>7.3f} {r.mean():>7.3f} {r.mean() - base:>+10.3f}")
    best = betas[int(np.argmax(val_mean.mean(axis=1)))]
    print(f"\nbest beta on validation: {best}")

    cmp_betas = sorted({0.0, best})
    print(f"\nTEST patients {test}: beta = 0 (no spatial prior) vs beta = {best}  (used once)")
    test_rows = []
    for p in test:
        r, n = evaluate_patient(p, args.grade, models, pi, args, cmp_betas, rng)
        test_rows.append(r)
    test_rows = np.array(test_rows)                               # (patients, betas, 3)
    print(f"{'patient':>8} " + " ".join(f"{'b=' + str(b) + ' ' + reg:>13}" for b in cmp_betas
                                        for reg in ("whole", "core", "enh")))
    for p, r in zip(test, test_rows):
        print(f"{p:>8} " + " ".join(f"{r[i, j]:>13.3f}" for i in range(len(cmp_betas)) for j in range(3)))
    m = test_rows.mean(axis=0)
    print(f"{'mean':>8} " + " ".join(f"{m[i, j]:>13.3f}" for i in range(len(cmp_betas)) for j in range(3)))
    if len(cmp_betas) == 2:
        gain = (test_rows[:, 1, :] - test_rows[:, 0, :])
        print(f"\nchange from the spatial prior, per region (mean over test patients +- std): "
              + ", ".join(f"{n} {gain[:, j].mean():+.3f} +- {gain[:, j].std(ddof=1):.3f}"
                          for j, n in enumerate(DEFAULT_REGIONS)))
        print(f"patients where the MRF helped (mean of the 3 regions): "
              f"{int((gain.mean(axis=1) > 0).sum())} of {len(test)}")
    else:
        print("\n(beta = 0 won on validation: the spatial prior did not help with this likelihood)")


if __name__ == "__main__":
    main()
