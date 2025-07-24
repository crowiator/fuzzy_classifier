"""
Viacstupňové predspracovanie EKG signálu pre analýzu a klasifikáciu
--------------------------------------------------------------------
* Obsahuje kombinovateľné kroky: band-pass filtráciu, notch filter, waveletové denoising a korekciu bazálnej línie.
* Podporuje rôzne metódy čistenia a detekcie R-vrcholov (napr. Pan-Tompkins, Neurokit).
* Waveletové denoising pomocou DWT zachováva morfológiu vĺn a redukuje šum bez deformácie QRS komplexu.
* Funkcia `clean_ecg_v2` predstavuje odporúčaný „preset“ pre robustné predspracovanie signálu z MIT-BIH databázy.
* Výstupné signály sú vhodné pre ďalšiu analýzu: extrakciu príznakov, klasifikáciu úderov, vizualizáciu a interpretáciu.
"""

from __future__ import annotations
from typing import Literal, Any
import numpy as np
import neurokit2 as nk
import pywt


# ─────────────────────────────────────────────────────────────────────────────
# 1. Kompletné spracovanie EKG signálu – výstup je DataFrame s črtami a slovník s informáciami
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Čistenie signálu pomocou zabudovaných filtrov NeuroKit2
# ─────────────────────────────────────────────────────────────────────────────

def clean_only(
        signal: np.ndarray,
        fs: int,
        *,
        method: str = "neurokit",
        powerline: int | None = 60,
        **kwargs: Any,
) -> np.ndarray:
    """
    Použije high-pass filter (0.5 Hz) a power-line notch filter (napr. 50/60 Hz),
    podľa zvoleného „method“. Výstupom je prefiltrovaný signál.
    """
    return nk.ecg_clean(
        signal,
        sampling_rate=fs,
        method=method,
        powerline=powerline,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Detekcia R-vrcholov (R-peaks) v už prefiltrovanom signále
# ─────────────────────────────────────────────────────────────────────────────
def detect_rpeaks(
        clean_signal: np.ndarray,
        fs: int,
        *,
        method: Literal["neurokit", "pantompkins1985",
        "elgendi2010", "ssf", "hamilton"] = "neurokit",
        **kwargs: Any,
) -> tuple[dict, dict]:
    """
        Volá nk.ecg_peaks() na detekciu R-vrcholov.
        Predpokladá, že vstupný signál je už prefiltrovaný.
        Výstupom je dvojica: (slovník s R-peaks, informačný slovník)
        """
    return nk.ecg_peaks(
        clean_signal, sampling_rate=fs, method=method, **kwargs
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Vlastný (customizovateľný) filter – band-pass, notch, FIR/Butterworth
# ─────────────────────────────────────────────────────────────────────────────
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
    Univerzálny obal pre nk.signal_filter().
    Umožňuje kombinovať rôzne typy filtrov – lowpass, highpass, bandpass, notch.
    Vhodné na presné doladenie filtrácie podľa typu šumu alebo diagnostickej potreby.
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
# 5. Waveletové denoising – založené na diskrétnej waveletovej transformácii (DWT)
# ─────────────────────────────────────────────────────────────────────────────
def dwt_denoise(
        signal: np.ndarray,
        wavelet: str = "db6",
        level: int | None = None,
        mode: Literal["soft", "hard"] = "soft",
) -> np.ndarray:
    """
       Aplikuje denoising pomocou prahovania waveletových koeficientov.
       - Detaily s malou amplitúdou sa potlačia (soft/hard thresholding)
       - Zachováva morfológiu vĺn, najmä QRS
       - Odporúčaný wavelet: 'db6' pre EKG
       """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

    new_coeffs = [coeffs[0]]
    new_coeffs += [pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]]

    # Zachová exaktne pôvodnú dĺžku a dátový typ
    recon = pywt.waverec(new_coeffs, wavelet)
    return recon[: len(signal)].astype(signal.dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Kombinovaná funkcia clean_ecg_v2 – odporúčaný „preset“ filtrácie signálu
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
    Odporúčané predspracovanie EKG signálu:
    1) Band-pass 0.5–40 Hz → odstráni pomalé zmeny a vysokofrekvenčný šum
    2) Notch filter 50/60 Hz → odstránenie sieťového rušenia
    3) (voliteľne) Wavelet DWT denoising → jemné potlačenie šumu
    4) NeuroKit2 ecg_clean → korekcia bazálnej línie

    Výstupom je robustne prefiltrovaný signál pripravený na analýzu.
    """
    # krok 1 + 2 (band-pass + notch)
    sig = custom_filter(signal, fs, lowcut=0.5, highcut=40, powerline=60)

    # krok 3 (voliteľne DWT)
    if add_dwt:
        sig = dwt_denoise(sig, wavelet=wavelet)

    # krok 4 – NK baseline filter
    sig = nk.ecg_clean(sig, fs, method=nk_method, powerline=60)

    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Alternatívna verzia clean_ecg pre MIT-BIH databázu
# ─────────────────────────────────────────────────────────────────────────────
def mitbih_filter(signal, fs=360):
    sig = custom_filter(signal, fs, lowcut=0.5, highcut=40, powerline=60)
    sig = dwt_denoise(sig, wavelet="db6")
    sig = clean_only(sig, fs, powerline=60)  # NK2 final pass
    return sig
