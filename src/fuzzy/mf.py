"""Automatické generovanie fuzzy príslušnostných funkcií (Gaussovských
alebo crisp) pre každý vstupný príznak.

Použitie
--------
>>> mf_dict = auto_mf(X_train, feature_names)

Výsledný slovník *mf_dict* má tvar::

    {
        "QRSd_ms": {
            "low":  (mu, sigma),
            "mid":  (mu, sigma),
            "high": (mu, sigma)
        },
        "early_P": {
            "no":  (0.0, 1e-3),   # crisp → delta funkcia
            "yes": (1.0, 1e-3)
        },
        "qrs_pol": {
            "down": (-1.0, 1e-3),
            "up":   ( 1.0, 1e-3)
        }
        ...
    }

*  Súvislé (kontinuálne) črty dostanú **tri Gauss MF** (low / mid / high)
   – pokiaľ názov nie je v `TWO_LEVEL_FEATURES`, kedy vytvoríme len "low"
   a "high".
*  Centrá Gaussoviek sa nastavia podľa 33 % / 66 % kvantilov (ak nie sú
   prebité ručným zadaním v `MANUAL_BOUNDS`).
*  Štandardná odchýlka sa volí tak, aby sa dve susedné Gaussovky
   pretli v hodnote členstva **0.5** (σ ≈ 0.43·Δμ/2).
*  Binárne / crisp príznaky kódujeme tiež ako Gaussovky s veľmi malou σ,
   aby bol výstup vždy dvojica *(μ, σ)*, ktorú očakáva Mamdani engine.
"""
from __future__ import annotations

from typing import Dict, Tuple, Set, Sequence

import numpy as np
import pandas as pd
IGNORE_FEATURES = {"beat_idx", "R_sample"}
# ----------------------------------------------------------------------------
# Kategorizácia príznakov -----------------------------------------------------
# ----------------------------------------------------------------------------
# Jedno‑bitové príznaky (0/1)
BINARY_FEATURES: Set[str] = {
    "has_P",
    "has_T",
    "early_P",
}
# Polaritné príznaky – dva crisp stavy (‑1 / +1)
POLARITY_FEATURES: Dict[str, Tuple[str, str]] = {
    "qrs_pol": ("down", "up"),   # QRS smerom dole / hore
    "t_pol":   ("down", "up"),
}
# Príznaky, ktoré spracujeme v log‑škále pred výpočtom kvantilov
LOG_FEATURE_PREFIX = ("E_L",)          # E_L2, E_L3, …
LOG_FEATURE_EXACT  = {"ratio_L2_L3"}
# Črty, ktoré majú iba dve Gauss MF
TWO_LEVEL_FEATURES: Set[str] = {"ratio_L2_L3"}

# Klinické ručné hranice (μ_low, μ_high)
MANUAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "RR_s":   (0.55, 1.00),   # krátky / dlhý RR
    "HR_bpm": (55.0, 100.0),  # brady / tachy
}

_CRISP_SIGMA = 1e-3  # prakticky delta‑funkcia (ale σ ≠ 0, aby nedošlo k /0)

# ----------------------------------------------------------------------------
# Pomocné funkcie -------------------------------------------------------------
# ----------------------------------------------------------------------------

def _sigma_from_bounds(lo: float, hi: float) -> float:
    """Výpočet σ tak, aby sa susedné Gauss MF pretli pri μ ± Δμ/2."""
    return 0.43 * (hi - lo) / 2.0


def _is_log_feature(name: str) -> bool:
    """Vráti True, ak sa príznak má najprv log10‑transformovať."""
    return name.startswith(LOG_FEATURE_PREFIX) or name in LOG_FEATURE_EXACT

# ----------------------------------------------------------------------------
# Hlavná funkcia --------------------------------------------------------------
# ----------------------------------------------------------------------------

def auto_mf(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    manual_bounds: Dict[str, Tuple[float, float]] | None = None,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Vypočíta parametre príslušnostných funkcií pre všetky črty.

    Parametre
    ---------
    X : np.ndarray  (n_samples, n_features)
        Matica beat × príznak.
    feature_names : list[str]
        Mená stĺpcov v rovnakom poradí ako sú v *X*.
    manual_bounds : dict[str, (low, high)], voliteľné
        Ručné prebitie μ_low / μ_high pre vybrané príznaky.

    Návratová hodnota
    -----------------
    dict
        Štruktúra `{feature: {label: (mu, sigma)}}`, ktorú vie použiť
        `FuzzyECGClassifier`.
    """

    mf_dict: Dict[str, Dict[str, Tuple[float, float]]] = {}
    manual_bounds = manual_bounds or {}

    # Pre jednoduchosť si spravíme DataFrame
    df = pd.DataFrame(X, columns=feature_names)

    for col in feature_names:
        if col in IGNORE_FEATURES:
            continue
        x = df[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]            # odstraň NaN / ±inf
        if x.size == 0:
            continue                    # preskoč prázdny stĺpec

        # ---------------- binárne príznaky (crisp) ---------------------
        if col in BINARY_FEATURES:
            mf_dict[col] = {
                "no":  (0.0, _CRISP_SIGMA),
                "yes": (1.0, _CRISP_SIGMA),
            }
            continue

        # ---------------- polaritné príznaky (crisp) -------------------
        if col in POLARITY_FEATURES:
            neg_lbl, pos_lbl = POLARITY_FEATURES[col]
            mf_dict[col] = {
                neg_lbl: (-1.0, _CRISP_SIGMA),
                pos_lbl: ( 1.0, _CRISP_SIGMA),
            }
            continue

        # ---------------- súvislé (kontinuálne) príznaky --------------
        # Log‑škála pre energie / pomer, ak treba
        if _is_log_feature(col):
            x = np.log10(x + 1e-6)

        # Manuálne prebitie klinických hraníc?
        if col in (manual_bounds or MANUAL_BOUNDS):
            mu_low, mu_high = (manual_bounds or MANUAL_BOUNDS)[col]
        else:
            q33, q66 = np.quantile(x, [0.33, 0.66])
            mu_low, mu_high = float(q33), float(q66)

        mu_mid = (mu_low + mu_high) / 2.0
        sigma = _sigma_from_bounds(mu_low, mu_high)

        # Dva alebo tri MF?
        if col in TWO_LEVEL_FEATURES:
            mf_dict[col] = {
                "low":  (mu_low,  sigma),
                "high": (mu_high, sigma),
            }
        else:
            mf_dict[col] = {
                "low":  (mu_low,  sigma),
                "mid":  (mu_mid,  sigma),
                "high": (mu_high, sigma),
            }

    return mf_dict
