#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Príprava dát z MIT-BIH databázy na použitie vo WEKA (formát ARFF)
"""

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

X_tr = fe.transform(rows_tr).to_numpy(float)
feat_names = fe.feature_names_
idx = [feat_names.index(f) for f in FIS_FEATURES]

# Vybrané klinické príznaky
X_tr = X_tr[:, idx]

# ------------------------------------------------------------------#
# 4. Imputácia (odstránenie chýbajúcich hodnôt)
# ------------------------------------------------------------------#
imp = SimpleImputer(strategy="median")
X_tr = imp.fit_transform(X_tr)

# ------------------------------------------------------------------#
# 5. Balansovanie dát (SMOTE)
# ------------------------------------------------------------------#
smote = SMOTE(sampling_strategy={"F":1200, "S":1500, "V":1500}, k_neighbors=1, random_state=42)
X_os, y_os = smote.fit_resample(X_tr, y_tr)

print("Počet vzoriek po SMOTE:", pd.Series(y_os).value_counts())

# ------------------------------------------------------------------#
# 6. Export dát do ARFF formátu pre WEKA
# ------------------------------------------------------------------#
# 6. Export dát do ARFF formátu pre WEKA
df_weka = pd.DataFrame(X_os, columns=FIS_FEATURES)
df_weka["class"] = y_os

arff_dict = {
    'description': 'MIT-BIH EKG fuzzy classification dataset',
    'relation':    'EKG',
    'attributes':  [(col, 'REAL') for col in FIS_FEATURES] +
                   [('class', ['N', 'S', 'V', 'F', 'Q'])],
    'data':        df_weka.values.tolist()
}


import  arff


with open('train.arff', 'w') as f:
    arff.dump(arff_dict, f)

print("Dáta uložené do train.arff")