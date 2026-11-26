"""Which per-voxel intensity model is best?  (no spatial prior yet, i.e. beta = 0)

Patients are split by ID into disjoint train / validation / test groups:

    train       fit the class models
    validation  compare the model variants and pick one      <- all tuning happens here
    test        report the single chosen variant, once       <- never used for any choice

Variants differ in: normalisation (z-score vs robust median/MAD), number of Gaussians for the
healthy class (grey matter / white matter / CSF are different populations), and the strength
of the class prior (prior_power 1 = Bayes-consistent, 0.5 = weakened).

Two decision rules are reported for every variant:
    argmax  : each voxel gets its single most probable label (what the first run_slice used)
    region  : each nested region (whole / core / enhancing) is decided separately by
              P(region) > 0.5.  Variants are ranked by this rule.

Dice is computed in 3D over every `z_stride`-th slice of each patient, for the nested
BRATS regions: whole (labels 1-4), core (1,3,4) and enhancing (4).

    python experiments/eval_classifier.py HG
    python experiments/eval_classifier.py HG --n-train 10 --n-val 4 --n-test 4 --z-stride 2
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import patient_dir
from src.data_io import list_patients
from src.metrics import dice, precision_recall
from src.mrf2d import (DEFAULT_REGIONS, extract_voxels, fit_models, loglik_points,
                       region_predictions)

VARIANTS = [
    dict(norm="zscore", M=1, alpha=1.0),     # what run_slice.py did first
    dict(norm="zscore", M=1, alpha=0.5),
    dict(norm="zscore", M=4, alpha=1.0),
    dict(norm="zscore", M=4, alpha=0.5),
    dict(norm="robust", M=4, alpha=1.0),
    dict(norm="robust", M=4, alpha=0.5),
]


def name(v):
    return f"{v['norm']:>6} healthy M={v['M']} prior^{v['alpha']}"


def evaluate(models, pi, alpha, data):
    """Per patient: Dice [whole, core, enh] under the region rule, the argmax rule, and
    (precision, recall) of the whole-tumour region.  Returns an array (n_patients, 8)."""
    rows = []
    for Y, t in data:
        ll = loglik_points(Y, models, pi, alpha)
        q = softmax(ll, axis=0)
        pred_arg = ll.argmax(axis=0)
        pred_reg = region_predictions(q)
        d_reg, d_arg = [], []
        for rname, labs in DEFAULT_REGIONS.items():
            tr = np.isin(t, labs)
            d_reg.append(dice(pred_reg[rname], tr))
            d_arg.append(dice(np.isin(pred_arg, labs), tr))
        pr, rc = precision_recall(pred_reg["whole"], np.isin(t, DEFAULT_REGIONS["whole"]))
        rows.append(d_reg + d_arg + [pr, rc])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grade", choices=["HG", "LG"])
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--z-stride", type=int, default=2, help="use every k-th slice (saves memory/time)")
    args = ap.parse_args()

    ids = list_patients(args.grade)
    need = args.n_train + args.n_val + args.n_test
    if need > len(ids):
        sys.exit(f"need {need} patients but {args.grade} has {len(ids)}")
    train = ids[:args.n_train]
    val = ids[args.n_train:args.n_train + args.n_val]
    test = ids[args.n_train + args.n_val:need]
    print(f"train {train}\nvalidation {val}\ntest {test}  (disjoint)\n")

    load = lambda pids, norm: [extract_voxels(patient_dir(args.grade, p), norm, args.z_stride) for p in pids]
    fitted, results = {}, {}
    t0 = time.time()
    for norm in sorted({v["norm"] for v in VARIANTS}):
        print(f"[{norm}] reading {len(train) + len(val)} patients ...")
        tr, va = load(train, norm), load(val, norm)
        for v in (v for v in VARIANTS if v["norm"] == norm):
            key = (norm, v["M"])
            if key not in fitted:
                fitted[key] = fit_models(tr, K=5, healthy_components=v["M"])
            models, pi = fitted[key]
            results[name(v)] = evaluate(models, pi, v["alpha"], va).mean(axis=0)
        del tr, va
    print(f"  ({time.time() - t0:.0f}s)\n")

    print(f"VALIDATION Dice (mean over {len(val)} patients, no spatial prior)")
    print(f"{'':<36} {'------ region rule ------':>30}   {'argmax':>7}")
    print(f"{'variant':<36} {'whole':>7} {'core':>7} {'enh.':>7} {'mean':>7}   {'mean':>7}")
    for v in VARIANTS:
        r = results[name(v)]
        print(f"{name(v):<36} {r[0]:>7.3f} {r[1]:>7.3f} {r[2]:>7.3f} {r[:3].mean():>7.3f}   {r[3:6].mean():>7.3f}")

    best = max(VARIANTS, key=lambda v: results[name(v)][:3].mean())
    print(f"\nbest on validation (region rule): {name(best)}")

    models, pi = fitted[(best["norm"], best["M"])]
    te = evaluate(models, pi, best["alpha"], load(test, best["norm"]))
    print(f"\nTEST of that variant, patients {test} (the only time the test patients are used)")
    print(f"{'patient':>8} {'whole':>7} {'core':>7} {'enh.':>7}   {'whole precision':>16} {'whole recall':>13}")
    for p, r in zip(test, te):
        print(f"{p:>8} {r[0]:>7.3f} {r[1]:>7.3f} {r[2]:>7.3f}   {r[6]:>16.3f} {r[7]:>13.3f}")
    print(f"{'mean':>8} {te[:, 0].mean():>7.3f} {te[:, 1].mean():>7.3f} {te[:, 2].mean():>7.3f}   "
          f"{te[:, 6].mean():>16.3f} {te[:, 7].mean():>13.3f}")
    print(f"{'std':>8} {te[:, 0].std(ddof=1):>7.3f} {te[:, 1].std(ddof=1):>7.3f} {te[:, 2].std(ddof=1):>7.3f}")
    print(f"(same patients with the single-label argmax rule: whole {te[:, 3].mean():.3f}, "
          f"core {te[:, 4].mean():.3f}, enh. {te[:, 5].mean():.3f})")
    print("low recall = tumour is being missed; low precision = healthy tissue is labelled tumour")

    print("\nuse these settings in the other experiments:")
    s = f"--norm {best['norm']} --healthy-components {best['M']} --prior-power {best['alpha']}"
    print(f"  python experiments/run_slice.py {args.grade} <test id> {s}")
    print(f"  python experiments/eval_beta.py {args.grade} {s}")


if __name__ == "__main__":
    main()
