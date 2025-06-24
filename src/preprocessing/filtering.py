# ── src/preprocessing/filtering.py ───────────────────────────────────────────
from __future__ import annotations
from typing import Literal, Any
import numpy as np
import neurokit2 as nk
import pywt


# ───────────── 1. kompletné spracovanie (df + info)  ─────────────
def process_full(
        signal: np.ndarray,
        fs: int,
        *,
        method: str = "neurokit",
        **kwargs: Any,
) -> tuple[nk.pandas.DataFrame, dict]:
    """
    Obal na nk.ecg_process().
    → vracia (signals_df, info_dict)
    """
    return nk.ecg_process(signal, sampling_rate=fs, method=method, **kwargs)


# ───────────── 2. len čistenie  ──────────────────────────────────
def clean_only(
        signal: np.ndarray,
        fs: int,
        *,
        method: str = "neurokit",
        powerline: int | None = 60,
        **kwargs: Any,
) -> np.ndarray:
    """
    High-pass (0.5 Hz) + power-line filter (50/60 Hz) podľa zvoleného
    „method“.  Vráti prefiltrovaný signál.
    """
    return nk.ecg_clean(
        signal,
        sampling_rate=fs,
        method=method,
        powerline=powerline,
        **kwargs,
    )


# ───────────── 3. detekcia R-vrcholov  ───────────────────────────
def detect_rpeaks(
        clean_signal: np.ndarray,
        fs: int,
        *,
        method: Literal["neurokit", "pantompkins1985",
        "elgendi2010", "ssf", "hamilton"] = "neurokit",
        **kwargs: Any,
) -> tuple[dict, dict]:
    """
    Volá nk.ecg_peaks(); očakáva už prefiltrovaný signál.
    → vracia (rpeaks_dict, info_dict)
    """
    return nk.ecg_peaks(
        clean_signal, sampling_rate=fs, method=method, **kwargs
    )


# ───────────── 4. vlastné band / notch filtre  ──────────────────
def custom_filter(
        signal: np.ndarray,
        fs: int,
        *,
        lowcut: float | None = 0.5,
        highcut: float | None = 40.0,
        powerline: int | None = 60,
        order: int = 5,
        filter_type: Literal["butterworth", "fir"] = "butterworth",
        **kwargs: Any,
) -> np.ndarray:
    """
    Univerzálna obálka na nk.signal_filter() – podporuje band-pass,
    low-/high-pass aj notch.
    """
    return nk.signal_filter(
        signal,
        sampling_rate=fs,
        lowcut=lowcut,
        highcut=highcut,
        method=filter_type,
        order=order,
        powerline=powerline,
        **kwargs,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5. Waveletové denoising (DWT)
# ─────────────────────────────────────────────────────────────────────────────
def dwt_denoise(
    signal: np.ndarray,
    wavelet: str = "db6",
    level: int | None = None,
    mode: Literal["soft", "hard"] = "soft",
) -> np.ndarray:
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

    new_coeffs = [coeffs[0]]
    new_coeffs += [pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]]

    # Zachová exaktne pôvodnú dĺžku a dátový typ
    recon = pywt.waverec(new_coeffs, wavelet)
    return recon[: len(signal)].astype(signal.dtype, copy=False)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Kombinovaná funkcia clean_ecg_v2
#    (band-pass → notch → DWT → nk.ecg_clean)
# ─────────────────────────────────────────────────────────────────────────────


def clean_ecg_v2(
        signal: np.ndarray,
        fs: int,
        *,
        add_dwt: bool = True,
        wavelet: str = "db6",
        nk_method: str = "neurokit",
) -> np.ndarray:
    """
    Odporúčaný preset:
        1) band-pass 0.5–40 Hz
        2) notch 50 Hz
        3) (voliteľne) DWT denoising
        4) NeuroKit2 ecg_clean (bazálna línia)
    """
    # krok 1 + 2 (band-pass + notch)
    sig = custom_filter(signal, fs, lowcut=0.5, highcut=40, powerline=60)

    # krok 3 (voliteľne DWT)
    if add_dwt:
        sig = dwt_denoise(sig, wavelet=wavelet)

    # krok 4 – NK baseline filter
    sig = nk.ecg_clean(sig, fs, method=nk_method, powerline=60)

    return sig


def mitbih_filter(signal, fs=360):
    sig = custom_filter(signal, fs, lowcut=0.5, highcut=40, powerline=60)
    sig = dwt_denoise(sig, wavelet="db6")
    sig = clean_only(sig, fs, powerline=60)        # NK2 final pass
    return sig