
"""
Wavelet-based energetické príznaky pre EKG
-----------------------------------------
Funkcia extract_wavelet_features poskytuje výpočet energetických príznakov z EKG signálu
pomocou diskrétnej waveletovej transformácie (DWT).

Vlastnosti:
    • využíva predspracovanie signálu pomocou clean_ecg_v2
    • wavelet: Daubechies 6 (db6), úroveň rozkladu: 6
    • vracia energie hladín 2 až 6 a pomer vybraných hladín (napr. L2/L3, L6/L5)
"""
from __future__ import annotations
from typing import Sequence, Dict
import numpy as np
import pywt
from src.preprocessing.filtering import clean_ecg_v2


# Pomocná funkcia: výber detailových koeficientov z konkrétnej hladiny (napr. cD2, cD3, ...)
def _detail(coeffs: list[np.ndarray], level: int) -> np.ndarray:
    # coeffs = [cA6, cD6, cD5, cD4, cD3, cD2, cD1
    return coeffs[-level]  # cD_level


# Hlavná funkcia na výpočet waveletových energetických čŕt
def extract_wavelet_features(
        raw_sig: np.ndarray,
        fs: int,
        *,
        wavelet: str = "db6",  # použitý wavelet (Daubechies 6)
        decomp_level: int = 6,  # počet úrovní rozkladu
        levels: Sequence[int] = (2, 3, 4, 5, 6),  # ktoré hladiny energie sa počítajú
        clean_first: bool = True,  # či sa signál predtým prefiltruje
        normalize_energy: bool = True,  # či sa energia normalizuje dĺžkou koeficientov
        use_log: bool = False  # či sa energia prevedie na logaritmus (nepoužité v základnej verzii)
) -> Dict[str, float]:
    # Voliteľné predspracovanie signálu (napr. odstránenie šumu, baseline wander)
    signal = clean_ecg_v2(raw_sig, fs) if clean_first else raw_sig.astype(float)
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=decomp_level)

    # Diskrétna waveletová transformáci
    feats: Dict[str, float] = {}
    for L in levels:
        try:
            d = _detail(coeffs, L)  # detailové koeficienty pre danú hladinu
            energy = float(np.sum(d ** 2))  # výpočet energie ako súčet štvorcov amplitúd
            if normalize_energy:
                energy /= len(d)  # normalizácia energií (nezávislá od dĺžky signálu)
            feats[f"E_L{L}"] = energy  # výsledná energia pre hladinu L
            if use_log:
                feats[f"log_E_L{L}"] = np.log(energy + 1e-8)  # logaritmická transformácia (voliteľná)
        except IndexError:
            # Ak wavelet rozklad neobsahuje danú hladinu (zriedkavé), vráti NaN
            feats[f"E_L{L}"] = np.nan
            if use_log:
                feats[f"log_E_L{L}"] = np.nan

    # Výpočet pomeru energií medzi hladinami L2 a L3 – môže indikovať zmenu spektrálnej rovnováhy
    e2, e3 = feats.get("E_L2", np.nan), feats.get("E_L3", np.nan)
    feats["ratio_L2_L3"] = e2 / e3 if e3 else np.nan

    # Pomer medzi vyššími hladinami – napr. L6 (nízke frekvencie) a L5 – môže súvisieť s patológiami
    e5, e6 = feats["E_L5"], feats["E_L6"]
    feats["ratio_P_T"] = e6 / e5 if e5 else np.nan

    return feats
