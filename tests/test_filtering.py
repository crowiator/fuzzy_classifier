from src.preprocessing.filtering import clean_ecg
from src.preprocessing.load import load_record
import numpy as np

def test_clean_ecg_shapes():
    rec = load_record("100")
    out = clean_ecg(rec.signal, rec.fs, mode="both")
    assert out.shape == rec.signal.shape
    assert not np.isnan(out).any()