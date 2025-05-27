# src/preprocessing/filtering.py
from __future__ import annotations
from typing import Literal
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt, iirnotch

# ---- základné parametre (môžeš presunúť do config.py) ----------------------
LOWCUT   = 0.5        # Hz
HIGHCUT  = 40.0       # Hz
NOTCH_F  = 50.0       # alebo 60.0 podľa siete
ORDER    = 5          # Butterworth

# ---------------------------------------------------------------------------
def butter_bandpass(signal: np.ndarray, fs: int,
                    low: float = LOWCUT, high: float = HIGHCUT,
                    order: int = ORDER) -> np.ndarray:
    """0.5-40 Hz pásmový filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, signal)

def notch(signal: np.ndarray, fs: int,
          f0: float = NOTCH_F, q: float = 30.0) -> np.ndarray:
    """50/60 Hz notch (Q≈30)."""
    b, a = iirnotch(f0, q, fs)
    return filtfilt(b, a, signal)

# ---------------------------------------------------------------------------
def clean_ecg(signal: np.ndarray,
              fs: int,
              mode: Literal["nk", "manual", "both"] = "both",
              nk_method: str = "biosppy") -> np.ndarray:
    """
    • 'nk'     – len neurokit2.ecg_clean()
    • 'manual' – bandpass + notch (bez NK)
    • 'both'   – bandpass → notch → nk.ecg_clean() (odporúčané)
    """
    if mode == "nk":
        return nk.ecg_clean(signal, fs, nk_method)
    if mode == "manual":
        sig = butter_bandpass(signal, fs)
        return notch(sig, fs)
    if mode == "both":
        sig = butter_bandpass(signal, fs)
        sig = notch(sig, fs)
        return nk.ecg_clean(sig, fs, nk_method)
    raise ValueError(f"Nepodporovaný mode='{mode}'")