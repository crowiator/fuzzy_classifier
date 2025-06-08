import numpy as np
from pathlib import Path
from src.fuzzy.mf import auto_mf
from src.preprocessing.load import load_record
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer import FeatureExtractor

def _get_train_matrix():
    ids = ["101","106","108"]          # mini-subset DS1
    rows = [(load_record(r).signal, 360) for r in ids]

    fe = FeatureExtractor(return_array=False)
    fe.fit(rows)
    X_df = fe.transform(rows)
    return X_df.to_numpy(float), fe.feature_names_

def _gauss(mu, sigma, x):
    return np.exp(-0.5 * ((x-mu)/sigma)**2)

def test_overlap():
    X, names = _get_train_matrix()
    mf = auto_mf(X, names)
    LOG_FEATURES = {"ratio_L2_L3"} | {name for name in names if name.startswith("E_L")}
    for col, params in mf.items():
        # pre binárne/polaritné: stačí, že sigma je malinká
        if col in LOG_FEATURES:
            continue
        if len(params) <= 2:
            for mu, sigma in params.values():
                assert sigma < 1e-2
            continue
        # kontinuálne: overíme prekrytie ≥ 0.4
        idx = names.index(col)
        # po odfiltrovaní NaN
        col_vals = X[:, idx]
        col_vals = col_vals[np.isfinite(col_vals)]
        if col_vals.size == 0:
            continue

        # 20.–80. percentil z očistených dát
        p30, p70 = np.quantile(col_vals, [0.30, 0.70])
        xs_raw = np.linspace(p30, p70, 50)

        # ak sa príznak loguje v auto_mf(), prežeň xs cez log10
        is_log = col.startswith("E_L") or col == "ratio_L2_L3"
        xs = np.log10(xs_raw + 1e-6) if is_log else xs_raw

        thr = 0.05 if is_log else 0.10
        for x in xs:
            μmax = max(_gauss(mu, sigma, x) for mu, sigma in params.values())
            assert μmax >= thr, f"{col}: slabé prekrytie pri x={x}"