"""
Vystrihnutie okna okolo R-vrcholu
---------------------------------
segment_beats(signal, r_idx, fs, pre=0.2, post=0.4) →
    2-D pole [n_segment × seg_len]
"""
from __future__ import annotations
import numpy as np


def segment_beats(
    signal: np.ndarray,
    r_idx: np.ndarray,
    fs: int,
    *,
    pre: float = 0.2,
    post: float = 0.4,
) -> np.ndarray:
    """Vráti výseky (-pre, +post) s pevným počtom vzoriek."""
    seg_len = int(round((pre + post) * fs))
    pre_samp = int(round(pre * fs))
    post_samp = seg_len - pre_samp
    segments = []

    for r in r_idx:
        start = r - pre_samp
        end = r + post_samp
        if start < 0 or end > signal.size:
            continue  # segment pretŕča za okraj
        segments.append(signal[start:end])

    return np.vstack(segments) if segments else np.empty((0, seg_len))