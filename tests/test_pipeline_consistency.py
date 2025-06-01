import numpy as np
from pathlib import Path

from src.preprocessing.load                  import load_record
from src.feature_extraction.time_domain      import extract_beats   # ← správny import
from src.feature_extraction.transformer      import FeatureExtractor
from src.utils.cache_io                      import save_cache, load_cache
from src.config                              import DATA_DIR, LEAD


def _build_dataset(rec_ids):
    all_rows, all_labels, all_ids = [], [], []

    for rec_id in rec_ids:
        rec = load_record(rec_id, lead=LEAD, base_dir=DATA_DIR)

        sig, fs = rec.signal, rec.fs
        beats_df = extract_beats(sig, fs)            # len = platné beaty

        # mapovanie referenčných tried len pre beaty v beats_df
        ref_map = dict(zip(rec.r_peaks, rec.labels_aami))
        labels_this = [ref_map.get(r, "Q") for r in beats_df["R_sample"]]
        ids_this    = [f"{rec_id}_{i}" for i in range(len(beats_df))]

        all_rows.append((sig, fs))
        all_labels.extend(labels_this)
        all_ids.extend(ids_this)

    fe = FeatureExtractor(return_array=False)
    fe.fit(all_rows)
    X_df = fe.transform(all_rows)           # počet riadkov = Σ len(beats_df)

    X = X_df.to_numpy(float)
    y = np.asarray(all_labels)
    ids = np.asarray(all_ids)
    return X, y, ids


def test_X_y_ids_consistent(tmp_path: Path):
    rec_list = ["100", "101"]
    X, y, ids = _build_dataset(rec_list)
    print(X)
    # 1) rovnaké počty
    assert X.shape[0] == len(y) == len(ids)

    # 2) žiadne ±inf  (NaN sú povolené)
    assert not np.isinf(X).any()

    # 3) round-trip cez cache
    # 3) round-trip cez cache
    cache_file = tmp_path / "demo.npz"
    save_cache(X, y, ids, cache_file)
    X2, y2, ids2 = load_cache(cache_file)

    assert X2.shape == X.shape
    assert np.allclose(X2, X, equal_nan=True)  # ← NaN == NaN
    assert np.array_equal(y2, y)
    assert np.array_equal(ids2, ids)