# feature_extraction/time_domain.py
"""
Robustná extrakcia príznakov z jednotlivých srdcových úderov pre databázu MIT-BIH Arrhythmia.
-------------------------------------------------------------------------------------------
* Zachováva všetky údery vrátane atypických (napr. artefakty, šum), čo je kľúčové pre reprezentáciu triedy Q (neklasifikovateľné).
* Namiesto tvrdého filtrovania používa flagy kvality, ktoré ponechávajú rozhodovanie na klasifikačnom systéme (napr. fuzzy alebo ML).
* Umožňuje voliteľné škálovanie a orezávanie extrémnych hodnôt.
* Ošetruje prístup k signálu bezpečne (žiadne výnimky pri čítaní mimo rozsahu).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Final
from src.preprocessing.filtering import clean_ecg_v2
import matplotlib

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
#  Definícia klinicky relevantných prahových hodnôt (pre dospelých pacientov)
# ---------------------------------------------------------------------------
MIN_QRS_MS: Final = 50
MAX_QRS_MS: Final = 300
MIN_PR_MS: Final = 50
MAX_PR_MS: Final = 400
MIN_RR_S: Final = 0.25
MAX_RR_S: Final = 4.0
MAX_R_AMP_MV: Final = 5.0  # po prefiltri; raw amplitúda je v r_amp_raw
PAC_RATIO: Final = 0.8  # RR1/RR0 < 0.8 => potenciálny PAC

__all__ = ["extract_beats", "add_quality_flags"]


# ---------------------------------------------------------------------------
#  Pomocné funkcie
# ---------------------------------------------------------------------------

def _safe_amp(sig: np.ndarray, idx: float | int | np.floating) -> float:
    """Bezpečný prístup k hodnote signálu na danom indexe.
        V prípade, že index leží mimo rozsahu, funkcia vráti NaN.
    """
    if np.isfinite(idx):
        i = int(round(idx))
        if 0 <= i < sig.shape[0]:  # sig.shape[0] - dlzka signalu
            return float(sig[i])
    return np.nan


def _nearest(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pre každý prvok v poli 'left' nájde najbližší menší (lo) a väčší (hi) index z poľa 'right'.
    Používa sa napr. na určenie okolia R-peaku v rámci QRS komplexu.
    """
    lo, hi = [], []
    for x in left:
        a = right[right < x]  # všetky prvky v right, ktoré sú menšie ako x
        b = right[right > x]  # všetky prvky v right, ktoré sú väčšie ako x
        lo.append(a[-1] if a.size else np.nan)  # posledný menší (teda najbližší menší)
        hi.append(b[0] if b.size else np.nan)  # prvý väčší (teda najbližší väčší)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


# ---------------------------------------------------------------------------
#  Hlavná funkcia – beat-wise extrakcia
# ---------------------------------------------------------------------------

def extract_beats(
        raw_sig: np.ndarray,
        fs: int,
        *,
        r_idx: np.ndarray,
        clip_extremes: bool = False,
) -> pd.DataFrame:
    """Extrakcia morfologických a časových čŕt pre každý úder EKG.

    Parametre:
    - raw_sig: surový signál
    - fs: vzorkovacia frekvencia (Hz)
    - r_idx: detegované R-vrcholové indexy
    - clip_extremes: ak True, extrémne hodnoty sa orežú na definované hranice

    Výstup:
    - DataFrame, kde každý riadok reprezentuje jeden srdcový úder
    """

    # 1) Prefiltrovanie signálu (vrátane DWT komponentu)
    clean_pref = clean_ecg_v2(raw_sig, fs, add_dwt=True)
    clean = clean_pref.copy()

    if len(r_idx) < 3:
        return pd.DataFrame()

    # 2) Delineácia P/Q/R/S/T vĺn pomocou NeuroKit2
    waves, _ = nk.ecg_delineate(clean, rpeaks=r_idx, sampling_rate=fs, method="peak")
    p_peaks_idx = np.where(waves["ECG_P_Peaks"].to_numpy() == 1)[0]
    t_peaks_idx = np.where(waves["ECG_T_Peaks"].to_numpy() == 1)[0]
    # QRS začína od Q-peaku a končí na S-peaku
    q_peaks_idx = np.where(waves["ECG_Q_Peaks"].to_numpy() == 1)[0]
    s_peaks_idx = np.where(waves["ECG_S_Peaks"].to_numpy() == 1)[0]

    # 3) Nájdeme najbližšie Q a S ku každému R-peaku (začiatok a koniec QRS komplexu)
    r_on_corr, _ = _nearest(r_idx, q_peaks_idx)
    _, r_off_corr = _nearest(r_idx, s_peaks_idx)

    rows: list[dict] = []
    for i, r in enumerate(r_idx):
        # Výpočet RR intervalov (RR0 – predchádzajúci, RR1 – nasledujúci) a HR
        rr0 = (r - r_idx[i - 1]) / fs if i else np.nan
        rr1 = (r_idx[i + 1] - r) / fs if i < len(r_idx) - 1 else np.nan
        hr_bpm = 60 / rr0 if np.isfinite(rr0) and rr0 > 0 else np.nan

        # Výpočet trvania QRS komplexu (v ms)
        qrs_ms = ((r_off_corr[i] - r_on_corr[i]) / fs * 1_000
                  if np.isfinite(r_on_corr[i]) and np.isfinite(r_off_corr[i]) else np.nan)

        # Amplitúdy P, R a T vĺn (bezpečne)
        r_amp_raw = _safe_amp(clean_pref, r)
        r_amp = _safe_amp(clean, r)
        p_candidates = p_peaks_idx[p_peaks_idx < r]
        t_candidates = t_peaks_idx[t_peaks_idx > r]
        p_amp = _safe_amp(clean, p_candidates[-1]) if p_candidates.size else np.nan
        t_amp = _safe_amp(clean, t_candidates[0]) if t_candidates.size else np.nan

        # PR interval (čas medzi P a R) a identifikácia predčasných P vĺn
        pr_ms = ((r - p_candidates[-1]) / fs * 1_000) if p_candidates.size else np.nan
        early_p = float(0 < pr_ms < 120) if np.isfinite(pr_ms) else np.nan

        # Identifikácia potenciálneho PAC (predčasný predsieňový úder)
        rr_ratio = rr1 / rr0 if np.isfinite(rr0) and np.isfinite(rr1) and rr0 > 0 else np.nan
        is_pac = float(rr_ratio < PAC_RATIO) if np.isfinite(rr_ratio) else np.nan

        # Príznaky úderu – pridanie do výstupného zoznamu
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

    # 4) Voliteľné orezanie extrémnych hodnôt podľa klinických hraníc
    if clip_extremes and not df.empty:
        df["QRSd_ms"] = df["QRSd_ms"].clip(MIN_QRS_MS, MAX_QRS_MS)
        df["PR_ms"] = df["PR_ms"].clip(MIN_PR_MS, MAX_PR_MS)
        df[["R_amplitude", "P_amplitude", "T_amplitude"]] = (
            df[["R_amplitude", "P_amplitude", "T_amplitude"]].clip(-MAX_R_AMP_MV, MAX_R_AMP_MV)
        )

    # 5) Optimalizácia dátových typov pre nižšiu pamäťovú náročnosť
    if not df.empty:
        float_cols = df.select_dtypes(float).columns
        df[float_cols] = df[float_cols].astype("float32")
        df[["beat_idx", "R_sample"]] = df[["beat_idx", "R_sample"]].astype("int32")

    return df


# ---------------------------------------------------------------------------
#  Pridanie flagov kvality signálu
# ---------------------------------------------------------------------------

def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()
    df["qrs_out_rng"] = ~df["QRSd_ms"].between(MIN_QRS_MS, MAX_QRS_MS)
    df["pr_out_rng"] = ~df["PR_ms"].between(MIN_PR_MS, MAX_PR_MS)
    rr_ok = df["RR0_s"].between(MIN_RR_S, MAX_RR_S) & df["RR1_s"].between(MIN_RR_S, MAX_RR_S)
    df["rr_out_rng"] = ~rr_ok
    df["ramp_high"] = df["r_amp_raw"].abs() > MAX_R_AMP_MV
    df["pt_missing"] = df["P_amplitude"].isna() | df["T_amplitude"].isna()

    df["sig_bad"] = df[[
        "qrs_out_rng", "pr_out_rng", "rr_out_rng", "ramp_high", "pt_missing"
    ]].any(axis=1)

    return df
