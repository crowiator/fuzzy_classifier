from __future__ import annotations
from typing import Sequence, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.wavelet      import extract_wavelet_features


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Kombinuje beat‑wise časové črty a globálne wavelet energie.

    Parametre
    ---------
    wavelet_levels : tuple[int]
        Ktoré detailové hladiny DWT extrahovať (predvolene 2‑6).
    return_array : bool
        Ak *True*, `transform()` vracia `np.ndarray`, inak `pd.DataFrame`.
    impute_wavelet_only : bool, default=False
        • *False*  – imputuje medián **energii aj morfologickým črtám**
          (`p_amp`, `t_amp`, `QRSd_ms`).  Hodí sa pre modely, ktoré nevedia
          pracovať s NaN (napr. RandomForest).
        • *True*   – imputuje **iba wavelet energie**, morfologické črty
          nechá NaN (zachová informáciu „vlna chýba“).  Vtedy sa odporúča
          v následnej pipeline použiť `SimpleImputer` + *nan_* flagy.
    """

    ENERGY_COLS = ["E_L2", "E_L3", "E_L4", "E_L5", "E_L6", "ratio_L2_L3"]
    WAVE_COLS   = ["QRSd_ms", "p_amp", "t_amp"]

    def __init__(self,
                 wavelet_levels: Sequence[int] = (2,3,4,5,6),
                 return_array: bool = False,
                 impute_wavelet_only: bool = False):
        self.wavelet_levels      = tuple(wavelet_levels)
        self.return_array        = return_array
        self.impute_wavelet_only = impute_wavelet_only

        # mediany pre imputáciu energie (a prípadne morfolog. črty)
        self.medians_: dict[str, float] = {}
        self.feature_names_: list[str]  = []

    # ------------------------------------------------------------------
    def _norm(self, item):
        """Ak *item* nie je tuple (sig, fs), doplní default 360 Hz."""
        return item if isinstance(item, tuple) else (item, 360)

    # ------------------------------------------------------------------
    def fit(self, X, y: Any = None):
        # ziskaj zoznam všetkých stĺpcov
        sig, fs = self._norm(X[0])
        td = extract_beats(sig, fs).iloc[0]
        wl = extract_wavelet_features(sig, fs, levels=self.wavelet_levels)
        base_cols = list(td.index) + list(wl.keys())

        # ktoré stĺpce budeme imputovať mediánom?
        self._impute_cols_ = self.ENERGY_COLS.copy()
        if not self.impute_wavelet_only:
            self._impute_cols_.extend(self.WAVE_COLS)

        tmp = {c: [] for c in self._impute_cols_}
        for sig, fs in map(self._norm, X):
            beats = extract_beats(sig, fs)
            for c in beats.columns.intersection(self._impute_cols_):
                tmp[c].extend(beats[c].to_numpy(float))

        self.medians_ = {
            c: float(np.nanmedian(tmp[c])) if len(tmp[c]) else 0.0
            for c in self._impute_cols_
        }

        nan_cols = [f"nan_{c}" for c in self._impute_cols_]
        self.feature_names_ = base_cols + nan_cols
        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        rows = []
        for sig, fs in map(self._norm, X):
            beats_df = extract_beats(sig, fs)
            wl_feats = extract_wavelet_features(sig, fs, levels=self.wavelet_levels)

            for _, beat in beats_df.iterrows():
                row = pd.concat([beat, pd.Series(wl_feats)])

                nan_flags = {}
                # imputácia podľa nastavenia
                for c in self._impute_cols_:
                    if pd.isna(row[c]):
                        row[c] = self.medians_[c]
                        nan_flags[f"nan_{c}"] = 1.0
                    else:
                        nan_flags[f"nan_{c}"] = 0.0
                # stĺpce, ktoré nechávame NaN, ale chceme flag
                if self.impute_wavelet_only:
                    for c in self.WAVE_COLS:
                        if c not in self._impute_cols_:
                            nan_flags[f"nan_{c}"] = float(pd.isna(row[c]))
                row = pd.concat([row, pd.Series(nan_flags)])
                rows.append(row)

        df = pd.DataFrame(rows, columns=self.feature_names_)
        return df.to_numpy(float) if self.return_array else df
