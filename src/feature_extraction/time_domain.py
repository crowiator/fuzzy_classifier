from __future__ import annotations
import numpy as np
import pandas as pd
import neurokit2 as nk

from src.preprocessing.filtering import clean_ecg_v2
from src.preprocessing.r_peaks import detect_rpeaks


def extract_beats(raw_sig: np.ndarray, fs: int, *, early_theta: float = 0.7) -> pd.DataFrame:
    """Beat‑wise časové a morfologické črty.

    Nové stĺpce:
    ----------------
    * **RR1_s**  – interval k *nasledujúcemu* R‑vrcholu (sekundy)
    * **early_P** – 1.0 ak P‑vlna začína skôr než ``early_theta * RR0``.
    """

    # 1) prefiltrovanie (Butterworth+notch+DWT+NK baseline)
    clean = clean_ecg_v2(raw_sig, fs, add_dwt=True)

    # 2) R‑peaks (už na čistom signále)
    r_idx, _ = detect_rpeaks(clean, fs)

    # 3) delineácia – získať P/QRS/T indexy
    _, waves = nk.ecg_delineate(clean, r_idx, sampling_rate=fs, method="dwt")

    # numpy polia (NaN kde chýba)
    r_on  = np.asarray(waves["ECG_R_Onsets"],  dtype=float)
    r_off = np.asarray(waves["ECG_R_Offsets"], dtype=float)
    p_idx = np.asarray(waves["ECG_P_Peaks"],   dtype=float)
    t_idx = np.asarray(waves["ECG_T_Peaks"],   dtype=float)

    beats: list[dict] = []
    for i, r in enumerate(r_idx):
        # ----- základné RR‑metriky -------------------------------
        rr0 = (r - r_idx[i - 1]) / fs if i > 0 else np.nan   # predchádzajúce RR
        rr1 = (r_idx[i + 1] - r) / fs if i < len(r_idx) - 1 else np.nan  # nasledujúce RR

        hr = 60 / rr0 if np.isfinite(rr0) and rr0 > 0 else np.nan

        # ----- QRS dĺžka ----------------------------------------
        if np.isfinite(r_on[i]) and np.isfinite(r_off[i]) and r_off[i] > r_on[i]:
            qrs_ms = (r_off[i] - r_on[i]) / fs * 1000
        else:
            qrs_ms = np.nan

        # ----- amplitúdy / polarity -----------------------------
        r_amp = clean[int(r)]
        p_amp = clean[int(p_idx[i])] if np.isfinite(p_idx[i]) else np.nan
        t_amp = clean[int(t_idx[i])] if np.isfinite(t_idx[i]) else np.nan

        # polarity (up = 1, down = -1, nan = 0)
        qrs_pol = float(np.sign(r_amp)) if np.isfinite(r_amp) else 0.0
        t_pol   = float(np.sign(t_amp)) if np.isfinite(t_amp) else 0.0

        # ----- Early‑P indikátor --------------------------------
        if np.isfinite(p_idx[i]) and np.isfinite(rr0) and rr0 > 0:
            PR_sec = (r - p_idx[i]) / fs
            early_P = float(np.isfinite(PR_sec) and PR_sec < 0.12)  # 120 ms
        else:
            early_P = np.nan

        beats.append({
            "beat_idx": i,
            "R_sample": int(r),
            "QRSd_ms":  qrs_ms,
            "r_amp":   r_amp,
            "RR0_s":   rr0,
            "RR1_s":   rr1,
            "HR_bpm":  hr,
            "p_amp":   p_amp,
            "t_amp":   t_amp,
            "has_P":   float(np.isfinite(p_idx[i])),
            "has_T":   float(np.isfinite(t_idx[i])),
            "early_P": early_P,
            "qrs_pol": qrs_pol,
            "t_pol":   t_pol,
        })

    return pd.DataFrame(beats)