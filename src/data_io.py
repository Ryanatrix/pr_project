"""Loading and preprocessing of BRATS 2012/2013 .mha volumes.

(Named data_io, not io, on purpose: a module called io.py inside src/ can shadow
Python's standard-library `io` module and break imports.)

Expected layout (the Academic Torrents / BRATS2013 packaging), one folder per patient:

    data/BRATS2013/Image_Data/HG/0001/
        VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.<id>.mha
        VSD.Brain.XX.O.MR_T1/...        (a bias-corrected *_N4ITK.mha may also exist)
        VSD.Brain.XX.O.MR_T1C/...
        VSD.Brain.XX.O.MR_T2/...
        VSD.Brain_3more.XX.XX.OT/VSD.Brain_3more.XX.XX.OT.<id>.mha   (expert labels)

File names are matched by pattern, not hard-coded, so small naming differences are
tolerated.  Run   python -m src.data_io HG 0001   to see what is found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .config import DATA_ROOT, patient_dir

MODALITIES = ("Flair", "T1", "T1C", "T2")      # channel order of every array below


# ----------------------------------------------------------------------------
# finding files
# ----------------------------------------------------------------------------
def list_patients(grade: str) -> list[str]:
    """Patient ids under data/.../Image_Data/<grade>, e.g. ['0001', '0002', ...]."""
    root = DATA_ROOT / grade
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found. Check DATA_ROOT in src/config.py")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _pick(files, pattern, prefer=None, what=""):
    """First file whose *name* matches `pattern`; among those, prefer names matching `prefer`."""
    rx = re.compile(pattern, re.IGNORECASE)
    hits = [f for f in files if rx.search(f.name)]
    if not hits:
        raise FileNotFoundError(f"no .mha file matching '{pattern}' ({what}). "
                                f"Files seen: {[f.name for f in files]}")
    if prefer:
        pref = [f for f in hits if re.search(prefer, f.name, re.IGNORECASE)]
        hits = pref or hits
    return hits[0]


def find_files(pdir, prefer_n4: bool = True) -> dict:
    """Map each modality name (and 'truth') to the .mha path inside a patient folder."""
    files = sorted(Path(pdir).rglob("*.mha"))
    if not files:
        raise FileNotFoundError(f"no .mha files under {pdir}")
    found = {}
    for m in MODALITIES:
        # 'MR_T1\.' does not match 'MR_T1C.', so T1 and T1C cannot be confused
        found[m] = _pick(files, rf"MR_{m}\.", prefer="N4ITK" if prefer_n4 else None,
                         what=f"modality {m}")
    # labels: '...OT...' files (or '*truth*' in the older 2012 naming); prefer the 4-class one
    found["truth"] = _pick(files, r"\.OT\.|truth", prefer="3more", what="expert labels")
    return found


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def read_volume(path) -> np.ndarray:
    """Read one .mha file as a float32 array of shape (slices, H, W)."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def load_patient(pdir, prefer_n4: bool = True):
    """Load all four modalities and the expert labels of one patient.

    Returns
        vol    (Z, H, W, 4) float32, channels ordered as MODALITIES
        truth  (Z, H, W)    int16 expert labels
        files  dict of the paths actually used (print it once to check)
    """
    files = find_files(pdir, prefer_n4)
    vol = np.stack([read_volume(files[m]) for m in MODALITIES], axis=-1)
    truth = np.rint(read_volume(files["truth"])).astype(np.int16)
    if truth.shape != vol.shape[:3]:
        raise ValueError(f"label shape {truth.shape} != image shape {vol.shape[:3]}")
    return vol, truth, files


def region_masks(truth: np.ndarray) -> dict:
    """Nested BRATS evaluation regions from the label volume.

    ASSUMES the 4-class convention 1 = necrosis, 2 = edema, 3 = non-enhancing tumour,
    4 = enhancing tumour (check against the dataset documentation and np.unique(truth)).
        whole     = all tumour labels (1-4)
        core      = whole minus edema (1, 3, 4)
        enhancing = label 4
    With 3-level labels (older 2012 files: 1 = edema, 2 = active tumour) only 'whole'
    and 'active' are returned.
    """
    whole = truth > 0
    if truth.max() >= 3:
        return {"whole": whole,
                "core": np.isin(truth, (1, 3, 4)),
                "enhancing": truth == 4}
    return {"whole": whole, "active": truth == 2}


# ----------------------------------------------------------------------------
# slice selection and preprocessing
# ----------------------------------------------------------------------------
def best_slice(mask: np.ndarray) -> int:
    """Index (along axis 0) of the slice with the largest area of `mask`."""
    return int(np.argmax(mask.reshape(mask.shape[0], -1).sum(axis=1)))


def prep_slice(vol: np.ndarray, truth: np.ndarray, z: int):
    """One 2D slice, ready for the model.

    Returns
        Y        (H, W, 4) each channel z-scored inside the brain, 0 outside
        brain    (H, W) bool brain mask (BRATS volumes are skull-stripped, so voxels
                 that are > 0 in any modality)
        truth_z  (H, W) int16 expert labels of this slice
    """
    Y = vol[z].copy()
    brain = (Y > 0).any(axis=-1)
    for c in range(Y.shape[-1]):
        ch = Y[..., c]                       # a view, so the assignment below sticks
        v = ch[brain]
        sd = float(v.std()) if v.size else 1.0
        ch[brain] = (v - (v.mean() if v.size else 0.0)) / (sd if sd > 0 else 1.0)
    Y[~brain] = 0.0
    return Y, brain, truth[z]


def load_slice(grade: str, pid: str, z: int | None = None):
    """Convenience: load a patient and return (Y, brain, truth_z, z).

    If z is None the slice with the largest whole-tumour area is used.
    """
    vol, truth, _ = load_patient(patient_dir(grade, pid))
    if z is None:
        z = best_slice(truth > 0)
    Y, brain, truth_z = prep_slice(vol, truth, z)
    return Y, brain, truth_z, z


# ----------------------------------------------------------------------------
# quick check:  python -m src.data_io HG 0001
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    grade = sys.argv[1] if len(sys.argv) > 1 else "HG"
    pid = sys.argv[2] if len(sys.argv) > 2 else list_patients(grade)[0]
    vol, truth, files = load_patient(patient_dir(grade, pid))
    print(f"{grade}/{pid}")
    for k, f in files.items():
        print(f"  {k:6s} <- {f.relative_to(patient_dir(grade, pid))}")
    print(f"  volume shape {vol.shape}   labels present {np.unique(truth).tolist()}")
    for name, m in region_masks(truth).items():
        print(f"  {name:9s} {int(m.sum()):8d} voxels")
    z = best_slice(truth > 0)
    Y, brain, tz = prep_slice(vol, truth, z)
    print(f"  best slice z={z}: brain pixels {int(brain.sum())}, "
          f"tumour pixels {int((tz > 0).sum())}, "
          f"channel means inside brain {np.round(Y[brain].mean(0), 3).tolist()}")
