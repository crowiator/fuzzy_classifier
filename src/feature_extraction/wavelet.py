# src/feature_extraction/wavelet.py
"""
Wavelet-based energetické príznaky pre EKG
-----------------------------------------
extract_wavelet_features(raw_sig, fs)
    • prefiltrovanie clean_ecg_v2
    • db6, level=6
    • energie hladín 2-4 + pomer L2/L3
"""
from __future__ import annotations
from typing import Sequence, Dict
import numpy as np
import pywt

from src.preprocessing.filtering import clean_ecg_v2


def _detail(coeffs: list [np.ndarray], level:int)-> np.ndarray:
    # coeffs = [cA6, cD6, cD5, cD4, cD3, cD2, cD1
    return coeffs[-level] # cD_level


def extract_wavelet_features(
        raw_sig: np.ndarray,
        fs: int,
        *,
        wavelet: str = "db6",
        decomp_level: int = 6,
        levels: Sequence[int] = (2, 3, 4, 5, 6),
        clean_first: bool = True,
) -> Dict[str, float]:
    signal = clean_ecg_v2(raw_sig, fs) if clean_first else raw_sig.astype(float)
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=decomp_level)

    feats: Dict[str, float] = {}
    for L in levels:
        try:
            d = _detail(coeffs, L)
            feats[f"E_L{L}"] = float(np.sum(d ** 2))
        except IndexError:
            feats[f"E_L{L}"] = np.nan

    # pomer L2 / L3
    e2, e3 = feats.get("E_L2", np.nan), feats.get("E_L3", np.nan)
    feats["ratio_L2_L3"] = e2 / e3 if np.isfinite(e2) and np.isfinite(e3) and e3 != 0 else np.nan
    # ratio_P_T napr. E_L6 / E_L5
    e5, e6 = feats["E_L5"], feats["E_L6"]
    feats["ratio_P_T"] = e6 / e5 if np.isfinite(e5) and e5 != 0 else np.nan
    return feats
