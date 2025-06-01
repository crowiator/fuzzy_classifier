"""
Jednoduché ukladanie a čítanie datasetu do *.npz súboru
-------------------------------------------------------
save_cache(X, y, record_ids, path)
load_cache(path) -> (X, y, record_ids)

• X   : 2-D np.ndarray   (n_samples × n_features)
• y   : 1-D np.ndarray / list
• ids : 1-D list / array (napr. "100_035", "101_247" …)

Interný formát: numpy .npz (komprimovaný zip s tromi poľami).
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────

def save_cache(
        X: np.ndarray,
        y: Sequence,
        record_ids: Sequence[str],
        path: Path | str
) -> None:
    """Uloží X/y/ids do komprimovaného .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=np.asarray(y), ids=np.asarray(record_ids))


def load_cache(path: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Načíta X, y, ids z cache a vráti tuple."""
    with np.load(Path(path)) as data:
        return data["X"], data["y"], data["ids"]