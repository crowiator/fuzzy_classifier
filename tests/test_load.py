# tests/test_load.py
import numpy as np
from src.preprocessing.load import load_record

def test_load_record_basic():
    rec = load_record("100")            # ← objekt LoadedRecord
    assert isinstance(rec.signal, np.ndarray)
    assert rec.signal.ndim == 1
    assert rec.signal.size > 1000       # ≈ aspoň 3 s signálu
    assert rec.fs > 0
    assert len(rec.r_peaks) > 0