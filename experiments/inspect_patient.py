"""Look at one BRATS patient: the four modalities and the expert labels on one slice.

Run from the repo root:
    python experiments/inspect_patient.py HG 0001
    python experiments/inspect_patient.py LG 0003 --z 80 --show

This is the 'does my data load and line up?' check: all four images should show the
same brain, and the label overlay should sit on the bright tumour in FLAIR / T2.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import patient_dir
from src.data_io import MODALITIES, best_slice, load_patient, prep_slice, region_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grade", choices=["HG", "LG"])
    ap.add_argument("pid", help="patient id, e.g. 0001")
    ap.add_argument("--z", type=int, default=None, help="slice index (default: largest tumour)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol, truth, files = load_patient(patient_dir(args.grade, args.pid))
    print("files used:")
    for k, f in files.items():
        print(f"  {k:6s} {f.name}")
    print("volume", vol.shape, " labels present", np.unique(truth).tolist())
    for name, m in region_masks(truth).items():
        print(f"  {name:9s} {int(m.sum()):8d} voxels")

    z = args.z if args.z is not None else best_slice(truth > 0)
    Y, brain, truth_z = prep_slice(vol, truth, z)
    print(f"slice z={z}: brain pixels {int(brain.sum())}, tumour pixels {int((truth_z > 0).sum())}")

    fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
    for c, m in enumerate(MODALITIES):
        ax[c].imshow(Y[..., c], cmap="gray")
        ax[c].contour(truth_z > 0, levels=[0.5], colors="r", linewidths=0.8)
        ax[c].set_title(m)
    ax[4].imshow(truth_z, cmap="viridis")
    ax[4].set_title("expert labels")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{args.grade} {args.pid}, slice {z}  (red = whole-tumour outline)")
    fig.tight_layout()

    out = Path(__file__).resolve().parents[1] / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"inspect_{args.grade}_{args.pid}.png"
    fig.savefig(path, dpi=130)
    print("saved", path)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
