
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

from src.config import DATA_DIR, LEAD
from src.preprocessing.load import load_record
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer import FeatureExtractor

import re
from src.feature_extraction.transformer import FeatureExtractor
# načítané pravidlá z kroku 2
with open("furia_rules.txt", encoding="utf-8") as fh:
    rules = [line.strip() for line in fh if line.strip()]

# regex vytiahne názov premennej pred textom "in ["
vars_found = set()
for r in rules:
    vars_found.update(re.findall(r"\((\w+)\s+in\s+\[", r))

print("Nájdené premenné v pravidlách:")
for v in sorted(vars_found):
    print(" ", v)
print("\nCelkový počet premenných:", len(vars_found))


# ------------------------------------------------------------------#
# 1. Definícia parametrov a datasetov
# ------------------------------------------------------------------#
DS1 = ["100","101","102","103","109","111","112","114","115","116",
       "117","118","121","122","123","212","214","215","231","234"]

FIS_FEATURES = [
    "QRSd_ms","r_amp",
    "RR0_s","RR1_s","HR_bpm",
    "p_amp","t_amp",
    "E_L3","E_L4","ratio_L2_L3",
    "early_P","has_P","has_T",
]

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)

# ------------------------------------------------------------------#
# 2. Pomocná funkcia na načítanie dát (už z tvojho skriptu)
# ------------------------------------------------------------------#
def cached_rows(rec_ids: list[str], tag: str):
    f = CACHE / f"{tag}_rows.pkl"
    if f.exists():
        with f.open("rb") as fh:
            rows, labels = pickle.load(fh)
        return rows, np.asarray(labels)

    rows, labels = [], []
    for rid in tqdm(rec_ids, desc=f"extract {tag}"):
        rec = load_record(rid, lead=LEAD, base_dir=DATA_DIR)
        rows.append((rec.signal, rec.fs))
        ref = dict(zip(rec.r_peaks, rec.labels_aami))
        beats = extract_beats(rec.signal, rec.fs)
        labels.extend([ref.get(r, "Q") for r in beats["R_sample"]])
    with f.open("wb") as fh:
        pickle.dump((rows, labels), fh)
    return rows, np.asarray(labels)

# Načítaj tréningové dáta
rows_tr, y_tr = cached_rows(DS1, "ds1")

# ------------------------------------------------------------------#
# 3. Extrakcia príznakov (feature extraction)
# ------------------------------------------------------------------#
fe = FeatureExtractor(return_array=False, impute_wavelet_only=True)
fe.fit(rows_tr)
X = fe.transform(rows_tr).to_numpy(float)
feat_names = fe.feature_names_
df_feat = pd.DataFrame(X, columns=feat_names)

# len premenné, ktoré sa objavili v pravidlách
ranges = {}
for var in sorted(vars_found):
    col = df_feat[var].dropna()
    ranges[var] = (float(col.min()), float(col.max()))

print("Min–max podľa DS-1:")
for k, (lo, hi) in ranges.items():
    print(f"{k:10s}: {lo:.3f}  …  {hi:.3f}")