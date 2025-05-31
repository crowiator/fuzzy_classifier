from src.preprocessing.filtering import clean_ecg_v2
from src.preprocessing.load import load_record
import numpy as np

def test_v2_pipeline_shapes():
    rec = load_record("100")
    clean = clean_ecg_v2(rec.signal, rec.fs)
    assert clean.shape == rec.signal.shape
    assert not np.isnan(clean).any()