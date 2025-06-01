import numpy as np
from pathlib import Path
from src.preprocessing.load import load_record
from src.feature_extraction.wavelet import extract_wavelet_features

def test_wavelet_energy_finite():
    rec = load_record("100", base_dir=Path("data/mit"))
    feats = extract_wavelet_features(rec.signal, rec.fs)
    # všetky energie aj pomer musia byť konečné čísla
    assert all(np.isfinite(list(feats.values())))