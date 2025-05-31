import numpy as np
from pathlib import Path
from src.preprocessing.load import load_record
from src.preprocessing.filtering import clean_ecg_v2
from src.preprocessing.r_peaks import detect_rpeaks, fscore

DATA_DIR = Path("data/mit")

def _score(rec_id: str) -> float:
    rec = load_record(rec_id, base_dir=DATA_DIR)
    print(rec)
    clean = clean_ecg_v2(rec.signal, rec.fs, add_dwt=True)
    pred, _ = detect_rpeaks(clean, rec.fs, method="neurokit")
    print(fscore(pred, rec.r_peaks, rec.fs))
    return fscore(pred, rec.r_peaks, rec.fs)

def test_fscore_100():
    assert _score("100") >= 0.95

def test_fscore_101():
    assert _score("101") >= 0.95