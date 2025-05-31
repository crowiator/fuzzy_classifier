"""
Wrapper nad NeuroKit2 na detekciu R-vrcholov + metrika F-score
-------------------------------------------------------------

•  detect_rpeaks() – volá nk.ecg_peaks() a vráti indexy aj info.
•  fscore()        – porovná predikciu s referenciou v okne ±75 ms.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Literal
import numpy as np
import neurokit2 as nk
import wfdb
EXCLUDED = {"+", "~", "|", '"', "x", "!", "[", "]"}   # artefakty z MIT-BIH

def reference_rpeaks(
    record_id: str,
    *,
    base_dir: Path,
) -> np.ndarray:
    """
    Načíta R-vrcholy zo súboru <record_id>.atr v MIT-BIH databáze.

    Vracia 1-D `np.ndarray[int32]` s indexmi vo vzorkách.
    Artefakty (EXCLUDED) sa automaticky vypustia.
    """
    path = (base_dir / record_id).with_suffix("")      # bez .atr
    ann = wfdb.rdann(str(path), "atr")
    mask = ~np.isin(ann.symbol, list(EXCLUDED))
    return ann.sample[mask].astype(np.int32)
# ────────────────────────────────────────────────────────────────


def detect_rpeaks(
    signal: np.ndarray,
    fs: int,
    *,
    method: Literal[
        "neurokit", "pantompkins1985",
        "elgendi2010", "ssf", "hamilton"
    ] = "neurokit",
    **kwargs,
) -> Tuple[np.ndarray, dict]:
    """Obal na nk.ecg_peaks()."""
    print("detect kdasdjbashjdasd ")
    """Vracia (indices, info).  Indexy berieme z info dictu!"""
    _, info = nk.ecg_peaks(signal, sampling_rate=fs, method=method, **kwargs)
    indices = np.asarray(info["ECG_R_Peaks"], dtype=np.int32)
    print("detect kdasdjasdasdasdasdasdabashjdasd ")
    return indices, info


# ────────────────────────────────────────────────────────────────
def fscore(
    predicted: np.ndarray,
    reference: np.ndarray,
    fs: int,
    tol_ms: float = 75.0,
) -> float:
    """F1-score v tolerancii ±tol_ms."""
    tol = int((tol_ms / 1_000) * fs)
    ref_set = set(reference)
    tp = 0
    used: set[int] = set()
    print("fscore")
    print(len(predicted))
    print(len(reference))
    for p in predicted:
        cand = [
            r for r in ref_set
            if abs(r - p) <= tol and r not in used
        ]
        if cand:
            tp += 1
            used.add(min(cand, key=lambda x: abs(x - p)))

    fp = len(predicted) - tp
    fn = len(reference) - tp
    print(fp)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)