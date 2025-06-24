"""
Robust beat-wise feature extraction for the MIT-BIH Arrhythmia database.
-----------------------------------------------------------------------
* Zachováva *všetky* údery vrátane atypických (dôležité pre AAMI triedu Q).
* Pridáva flagy kvality namiesto tvrdého filtra – model (fuzzy/ML) sa rozhodne sám.
* Voliteľný robust-z scaling a clipping extrémov.
* Žiadne IndexError: čítanie amplitúd bezpečne ošetrené.
"""
from __future__ import annotations

from typing import Final
import numpy as np
import pandas as pd
import neurokit2 as nk

from src.preprocessing.filtering import clean_ecg_v2

# ---------------------------------------------------------------------------
#  Prahové konstanty (dospelý, MIT-BIH, fs≈360 Hz)
# ---------------------------------------------------------------------------
MIN_QRS_MS:   Final = 50
MAX_QRS_MS:   Final = 300
MIN_PR_MS:    Final = 50
MAX_PR_MS:    Final = 400
MIN_RR_S:     Final = 0.25
MAX_RR_S:     Final = 4.0
MAX_R_AMP_MV: Final = 5.0      # po prefiltri; raw amplitúda je v r_amp_raw
PAC_RATIO:    Final = 0.8      # RR1/RR0 < 0.8 => potenciálny PAC

__all__ = ["extract_beats", "add_quality_flags"]

# ---------------------------------------------------------------------------
#  Pomocné funkcie
# ---------------------------------------------------------------------------

def _safe_amp(sig: np.ndarray, idx: float | int | np.floating) -> float:
    """Bezpečne vráti amplitúdu alebo NaN, ak index leží mimo signálu."""
    # Metóda np.isfinite() z knižnice NumPy slúži na zistenie, či je daná hodnota konečné číslo
    if np.isfinite(idx):
        i = int(round(idx))
        if 0 <= i < sig.shape[0]: # sig.shape[0] - dlzka signalu
            return float(sig[i])
    return np.nan


def _nearest(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ku každému prvku *left* nájdi najbližší menší a väčší index v *right*."""
    lo, hi = [], []
    for x in left:
        a = right[right < x] # všetky prvky v right, ktoré sú menšie ako x
        b = right[right > x] # všetky prvky v right, ktoré sú väčšie ako x
        lo.append(a[-1] if a.size else np.nan)  # posledný menší (teda najbližší menší)
        hi.append(b[0]  if b.size else np.nan)  # prvý väčší (teda najbližší väčší)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

# ---------------------------------------------------------------------------
#  Hlavná funkcia – beat-wise extrakcia
# ---------------------------------------------------------------------------

def extract_beats(
    raw_sig: np.ndarray,
    fs: int,
    *,
    r_idx: np.ndarray,
    zscore_amp: bool = False,
    clip_extremes: bool = False,
) -> pd.DataFrame:
    """Vráti DataFrame s beat-wise príznakmi (R/P/T amplitúdy, RR, HR, QRS…)."""

    # 1) prefiltrovanie & voliteľný robust-z
    clean_pref = clean_ecg_v2(raw_sig, fs, add_dwt=True)
    clean = clean_pref.copy()
    if zscore_amp:
        med = np.median(clean)
        mad = np.median(np.abs(clean - med)) or 1.0
        clean = (clean - med) / mad

    if len(r_idx) < 3:
        return pd.DataFrame()

    # 2) delineácia ⇒ binárne stĺpce (rovnako dlhé ako signál)
    waves, _ = nk.ecg_delineate(clean, rpeaks=r_idx, sampling_rate=fs, method="dwt")

    r_on_idx   = np.where(waves["ECG_R_Onsets" ].to_numpy() == 1)[0]
    print(f"r_on_idx: {len(r_on_idx)}")
    r_off_idx  = np.where(waves["ECG_R_Offsets"].to_numpy() == 1)[0]
    print(f"r_off_idx: {len(r_off_idx)}")
    p_peaks_idx = np.where(waves["ECG_P_Peaks" ].to_numpy() == 1)[0]
    print(f"p_peaks_idx: {len(p_peaks_idx)}")
    t_peaks_idx = np.where(waves["ECG_T_Peaks" ].to_numpy() == 1)[0]
    print(f"t_peaks_idx: {len(t_peaks_idx)}")

    # 3) priradenie onset/offset ku každému R-peaku
    #r_on_corr = najbližší onset PRED každým R-peakom - (teda začiatok QRS pre daný úder).
    # r_off_corr = najbližší offset ZA každým R-peakom -(koniec QRS pre daný úder)
    r_on_corr, r_off_corr = _nearest(r_idx, r_on_idx)[0], _nearest(r_idx, r_off_idx)[1]
    rows: list[dict] = []
    for i, r in enumerate(r_idx):
        # RR & HR
        rr0 = (r - r_idx[i - 1]) / fs if i else np.nan
        rr1 = (r_idx[i + 1] - r) / fs if i < len(r_idx) - 1 else np.nan
        hr_bpm = 60 / rr0 if np.isfinite(rr0) and rr0 > 0 else np.nan

        # QRS duration
        qrs_ms = ((r_off_corr[i] - r_on_corr[i]) / fs * 1_000
                   if np.isfinite(r_on_corr[i]) and np.isfinite(r_off_corr[i]) else np.nan)

        # amplitúdy
        r_amp_raw = _safe_amp(clean_pref, r)
        r_amp = _safe_amp(clean, r)
        p_candidates = p_peaks_idx[p_peaks_idx < r]
        t_candidates = t_peaks_idx[t_peaks_idx > r]
        p_amp = _safe_amp(clean, p_candidates[-1]) if p_candidates.size else np.nan
        t_amp = _safe_amp(clean, t_candidates[0])  if t_candidates.size else np.nan

        # PR interval & early-P
        pr_ms = ((r - p_candidates[-1]) / fs * 1_000) if p_candidates.size else np.nan
        early_p = float(0 < pr_ms < 120) if np.isfinite(pr_ms) else np.nan

        # PAC flag
        rr_ratio = rr1 / rr0 if np.isfinite(rr0) and np.isfinite(rr1) and rr0 > 0 else np.nan
        is_pac = float(rr_ratio < PAC_RATIO) if np.isfinite(rr_ratio) else np.nan

        rows.append({
            "beat_idx": i,
            "R_sample": int(r),
            "R_amplitude": r_amp,
            "P_amplitude": p_amp,
            "T_amplitude": t_amp,
            "RR0_s": rr0,
            "RR1_s": rr1,
            "Heart_rate_bpm": hr_bpm,
            "QRSd_ms": qrs_ms,
            "PR_ms": pr_ms,
            "early_P": early_p,
            "is_PAC": is_pac,
            "r_amp_raw": r_amp_raw,
            "qrs_pol": float(np.sign(r_amp)) if np.isfinite(r_amp) else 0.0,
            "t_pol": float(np.sign(t_amp)) if np.isfinite(t_amp) else 0.0,
        })

    df = pd.DataFrame(rows)

    # 4) clipping extrémov (voliteľné)
    if clip_extremes and not df.empty:
        df["QRSd_ms"] = df["QRSd_ms"].clip(MIN_QRS_MS, MAX_QRS_MS)
        df["PR_ms" ]  = df["PR_ms" ].clip(MIN_PR_MS,  MAX_PR_MS)
        df[["R_amplitude", "P_amplitude", "T_amplitude"]] = (
            df[["R_amplitude", "P_amplitude", "T_amplitude"]].clip(-MAX_R_AMP_MV, MAX_R_AMP_MV)
        )

    # 5) typy
    if not df.empty:
        float_cols = df.select_dtypes(float).columns
        df[float_cols] = df[float_cols].astype("float32")
        df[["beat_idx", "R_sample"]] = df[["beat_idx", "R_sample"]].astype("int32")

    return df

# ---------------------------------------------------------------------------
#  Quality flags
# ---------------------------------------------------------------------------

def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()
    df["qrs_out_rng"] = ~df["QRSd_ms"].between(MIN_QRS_MS, MAX_QRS_MS)
    df["pr_out_rng" ] = ~df["PR_ms" ].between(MIN_PR_MS, MAX_PR_MS)
    rr_ok = df["RR0_s"].between(MIN_RR_S, MAX_RR_S) & df["RR1_s"].between(MIN_RR_S, MAX_RR_S)
    df["rr_out_rng"] = ~rr_ok
    df["ramp_high"]  = df["r_amp_raw"].abs() > MAX_R_AMP_MV
    df["pt_missing"] = df["P_amplitude"].isna() | df["T_amplitude"].isna()

    df["sig_bad"] = df[[
        "qrs_out_rng", "pr_out_rng", "rr_out_rng", "ramp_high", "pt_missing"
    ]].any(axis=1)

    return df

left = np.array([10, 20, 30])
right = np.array([5, 15, 25, 35])
print(_nearest(left, right))