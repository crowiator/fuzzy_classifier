from __future__ import annotations
import numpy as np
import pandas as pd
import neurokit2 as nk

from src.preprocessing.filtering import clean_ecg_v2
from src.preprocessing.r_peaks import detect_rpeaks


def extract_beats(raw_sig: np.ndarray, fs: int) -> pd.DataFrame:
    """
    Každý riadok = jeden úder (QRSd_ms, r_amp, RR_s, HR_bpm, p_amp, t_amp, has_P, has_T)
    """
    # 1) prefiltrovanie
    clean = clean_ecg_v2(raw_sig, fs, add_dwt=True)

    # 2) R-vrcholy
    r_idx, _ = detect_rpeaks(clean, fs)

    # 3) delineácia (waves = dict!)
    _, waves = nk.ecg_delineate(clean, r_idx, sampling_rate=fs, method="dwt")

    # --- prevod na numpy (NaN tam, kde vlna chýba) -------------------------
    r_on = np.asarray(waves["ECG_R_Onsets"],  dtype=float)
    r_off = np.asarray(waves["ECG_R_Offsets"], dtype=float)

    p_idx = np.asarray(waves["ECG_P_Peaks"],  dtype=float)
    t_idx = np.asarray(waves["ECG_T_Peaks"],  dtype=float)

    # 4) beat-wise metriky ---------------------------------------------------
    beats = []
    for i, r in enumerate(r_idx):
        # QRS dĺžka: R_on → R_off
        if np.isfinite(r_on[i]) and np.isfinite(r_off[i]) and r_off[i] > r_on[i]:
            qrs_ms = (r_off[i] - r_on[i]) / fs * 1000
        else:
            qrs_ms = np.nan

        # amplitúdy
        r_amp = clean[int(r)]
        p_amp = clean[int(p_idx[i])] if np.isfinite(p_idx[i]) else np.nan
        t_amp = clean[int(t_idx[i])] if np.isfinite(t_idx[i]) else np.nan

        # RR a HR
        rr = (r - r_idx[i - 1]) / fs if i > 0 else np.nan
        hr = 60 / rr if np.isfinite(rr) else np.nan

        beats.append({
            "beat_idx": i,
            "R_sample": int(r),
            "QRSd_ms": qrs_ms,
            "r_amp":   r_amp,
            "RR_s":    rr,
            "HR_bpm":  hr,
            "p_amp":   p_amp,
            "t_amp":   t_amp,
            "has_P":   float(np.isfinite(p_idx[i])),
            "has_T":   float(np.isfinite(t_idx[i])),
        })

    # 5) vráť DataFrame až PO skončení cyklu
    return pd.DataFrame(beats)