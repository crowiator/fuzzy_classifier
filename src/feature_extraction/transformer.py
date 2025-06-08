"""
Scikit-learn kompatibilný prevodník príznakov
--------------------------------------------
FeatureExtractor(levels=(2,3,4), return_array=False)

• fit()       – no-op, len si zapamätá poradie stĺpcov
• transform() – zoznam (signal, fs) → DataFrame alebo ndarray

Spojí beat-wise časové črty z `extract_beats()` a globálne
wavelet energie z `extract_wavelet_features()`.
"""
from __future__ import annotations
from typing import Sequence, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.wavelet import extract_wavelet_features


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            *,
            wavelet_levels: Sequence[int] = (2, 3, 4, 5, 6),
            return_array: bool = False,
    ) -> None:
        self.wavelet_levels = tuple(wavelet_levels)
        self.return_array = return_array
        self.feature_names_: list[str] = []


    def _norm(self, item):
        # ak je to tuple (sig, fs) → OK
        # inak vráť (sig, 360)
        return item if isinstance(item, tuple) else (item, 360)

    # fit – nič neučíme, len určíme poradie stĺpcov
    # fit = príprava (zapamätanie si, čo bude po sebe).
    def fit(self, X: List[np.ndarray], y: Any = None):     # type: ignore[override]
        sig, fs = self._norm(X[0])
        td = extract_beats(sig, fs).iloc[0]
        wl = extract_wavelet_features(sig, fs, levels=self.wavelet_levels)
        self.feature_names_ = list(td.index) + list(wl.keys())
        return self

    # transform – vráti DataFrame / ndarray
    # transform = reálna transformácia každého vzorku.
    def transform(
            self,
            X: List[Tuple[np.ndarray, int]]
    ):
        rows: list[pd.Series] = []
        for item in X:
            sig, fs = self._norm(item)
            beats_df = extract_beats(sig, fs)
            wl_feats = extract_wavelet_features(sig, fs, levels=self.wavelet_levels)
            for _, beat_row in beats_df.iterrows():
                beat_row.drop(["beat_idx", "R_sample"])
                rows.append(pd.concat([beat_row, pd.Series(wl_feats)]))

        df = pd.DataFrame(rows, columns=self.feature_names_)

        return df.to_numpy(dtype=float) if self.return_array else df
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = make_pipeline(
    FeatureExtractor(return_array=True),
    StandardScaler(),
    RandomForestClassifier()
)
pipe.fit(train_records, y_train)     # train_records = [(sig, fs), ...]
y_pred = pipe.predict(test_records)
"""