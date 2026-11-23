from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "BRATS2013" / "Image_Data"

def patient_dir(grade, pid):          # e.g. patient_dir("HG", "0001")
    return DATA_ROOT / grade / pid