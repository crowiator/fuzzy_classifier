# feature_extraction/transfomer.py
"""
ML-friendly wrapper kombinujúci extrakciu časových čŕt z jednotlivých úderov s globálnou wavelet energiou.
Aktualizovaný tak, aby bol kompatibilný s funkciou extract_beats(raw_sig, fs, r_idx, ...).

Vstup X môže byť:
* n-tica (signál, vzorkovacia frekvencia, R-indexy)
* alebo objekt typu LoadedRecord (obsahuje .signal, .fs, .r_peaks)
"""
from __future__ import annotations
from typing import Sequence, Any, Iterable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.wavelet import extract_wavelet_features


class FeatureExtractor(BaseEstimator, TransformerMixin):
    ENERGY_COLS = [f"E_L{i}" for i in range(2, 7)] + ["ratio_L2_L3"]
    WAVE_COLS = ["QRSd_ms", "P_amplitude", "T_amplitude"]

    def __init__(
            self,
            wavelet_levels: Sequence[int] = (2, 3, 4, 5, 6),
            *,
            clip_extremes: bool = False,
            return_array: bool = False,
            impute_wavelet_only: bool = False,
    ) -> None:
        self.wavelet_levels = tuple(wavelet_levels)
        self.clip_extremes = clip_extremes
        self.return_array = return_array
        self.impute_wavelet_only = impute_wavelet_only
        # will be set in fit()
        self.medians_: dict[str, float] = {}
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------
    @staticmethod
    def _norm(item: Any) -> tuple[np.ndarray, int, np.ndarray]:
        if isinstance(item, tuple):
            if len(item) == 3:
                return item
            if len(item) == 2:
                sig, fs = item
                return sig, fs, np.array([], dtype=int)
        sig = item.signal
        fs = item.fs
        r_idx = item.r_peaks if hasattr(item, "r_peaks") else np.array([], dtype=int)
        return sig, fs, r_idx

    # ------------------------------------------------------------------
    def _collect_impute_values(self, beats_iter: Iterable[pd.DataFrame]):
        tmp: dict[str, list[float]] = {c: [] for c in self._impute_cols_}
        for beats in beats_iter:
            for c in beats.columns.intersection(self._impute_cols_):
                tmp[c].extend(beats[c].to_numpy(float))
        self.medians_ = {
            c: float(np.nanmedian(tmp[c])) if tmp[c] else 0.0
            for c in self._impute_cols_
        }

    # ------------------------------------------------------------------
    def fit(self, X, y: Any = None):
        sig, fs, r_idx = self._norm(X[0])
        if r_idx.size == 0:
            raise ValueError("FeatureExtractor.fit: r_idx missing; provide LoadedRecord or tuple with r_idx.")

        td_cols = extract_beats(sig, fs, r_idx=r_idx, clip_extremes=self.clip_extremes).columns.tolist()
        wl_cols = list(extract_wavelet_features(sig, fs, levels=self.wavelet_levels).keys())

        self._impute_cols_ = self.ENERGY_COLS.copy()
        if not self.impute_wavelet_only:
            self._impute_cols_.extend(self.WAVE_COLS)

        nan_cols = [f"nan_{col}" for col in self._impute_cols_]

        self.feature_names_ = td_cols + wl_cols + nan_cols

        beats_iter = (
            extract_beats(*self._norm(x)[:2], r_idx=self._norm(x)[2], clip_extremes=self.clip_extremes)
            for x in X
        )
        self._collect_impute_values(beats_iter)
        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        """
                Pre každý vstupný záznam:
                - extrahuje beat-wise črty (časové)
                - pridá globálne wavelet črty
                - doplní chýbajúce hodnoty mediánmi
                - pridá binárne príznaky, ktoré indikujú chýbajúce hodnoty
                """
        global beats_df, wl_feats, nan_flags
        rows = []
        for item in X:
            sig, fs, r_idx = self._norm(item)
            if r_idx.size == 0:
                raise ValueError("transform expects r_idx for each sample.")
            beats_df = extract_beats(sig, fs, r_idx=r_idx, clip_extremes=self.clip_extremes)
            wl_feats = extract_wavelet_features(sig, fs, levels=self.wavelet_levels)

            for i, (_, beat) in enumerate(beats_df.iterrows()):
                row = pd.concat([beat, pd.Series(wl_feats)])

                nan_flags = {}
                processed_cols = set()
                # Spojenie jedného úderu (beat) s globálnymi wavelet črtami
                for col in self._impute_cols_:
                    if pd.isna(row[col]):
                        row[col] = self.medians_[col]
                        nan_flags[f"nan_{col}"] = 1.0
                    else:
                        nan_flags[f"nan_{col}"] = 0.0
                    processed_cols.add(col)

                # Doplnenie NaN hodnôt mediánmi + indikátor chýbajúcej hodnoty
                if self.impute_wavelet_only:
                    for col in self.WAVE_COLS:
                        if col not in processed_cols:
                            nan_flags[f"nan_{col}"] = float(pd.isna(row[col]))
                            processed_cols.add(col)

                # Ak dopĺňame len wavelet črty, kontrolujeme zvyšné črty len pre informáciu
                row = pd.concat([row, pd.Series(nan_flags)])
                rows.append(row)
        # Výsledný DataFrame
        final_feature_names = (
                beats_df.columns.tolist() +
                list(wl_feats.keys()) +
                list(nan_flags.keys())
        )
        final_feature_names = list(dict.fromkeys(final_feature_names))
        df = pd.DataFrame(rows, columns=final_feature_names)
        return df.to_numpy(float) if self.return_array else df
