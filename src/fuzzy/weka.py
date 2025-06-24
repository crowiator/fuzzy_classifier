#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from src.feature_extraction.transformer import FeatureExtractor   # <- oprava: transform

import arff   # pip install liac-arff

DS1 = ["100","101","102","103","109","111","112","114","115","116",
       "117","118","121","122","123","212","214","215","231","234"] # zostáva


FIS_FEATURES = ['R_amplitude', 'P_amplitude', 'T_amplitude', 'RR0_s', 'RR1_s', 'Heart_rate_bpm', 'QRSd_ms', 'PR_ms', 'early_P', 'r_amp_raw', 'qrs_pol', 't_pol', 'E_L2', 'E_L3', 'E_L4', 'E_L5', 'E_L6', 'ratio_L2_L3', 'ratio_P_T', 'nan_E_L2', 'nan_E_L3', 'nan_E_L4', 'nan_E_L5', 'nan_E_L6', 'nan_ratio_L2_L3']

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)


def cached_rows(rec_ids: list[str], tag: str):
    """
    Vráti:
        rows   – list[(signal, fs, r_idx)]   (pripravené pre FeatureExtractor)
        labels – 1-D ndarray AAMI tried (rovnaká dĺžka ako počet beatov)
    Výsledok sa cache-uje do pickle v ./cache.
    """
    f = CACHE / f"{tag}_rows.pkl"
    if f.exists():
        with f.open("rb") as fh:
            rows, labels = pickle.load(fh)
        return rows, np.asarray(labels)

    rows, labels = [], []
    for rid in tqdm(rec_ids, desc=f"extract {tag}"):
        rec = load_record(rid, lead=LEAD, base_dir=DATA_DIR)

        # — 1) ak chýbajú R-peaks, záznam preskočíme —
        if rec.r_peaks.size == 0:
            print(f"Skipping {rid}: empty r_peaks")
            continue

        # — 2) pridaj trojicu do rows (signal, fs, r_idx) —
        rows.append((rec.signal, rec.fs, rec.r_peaks))

        # — 3) extrahuj beat-wise príznaky, aby si zistil R_sample každého beatu —
        beats = extract_beats(rec.signal, rec.fs,
                              r_idx=rec.r_peaks,
                              zscore_amp=True, clip_extremes=True)

        # — 4) priraď AAMI štítok každému beatu —
        r2label = dict(zip(rec.r_peaks, rec.labels_aami))
        labels.extend([r2label.get(r, "Q") for r in beats["R_sample"]])

    # — 5) ulož do cache —
    with f.open("wb") as fh:
        pickle.dump((rows, labels), fh)

    return rows, np.asarray(labels)


# 1) načítaj DS1
rows_tr, y_tr = cached_rows(DS1, "ds1")
print(y_tr)
# 2) extrakcia príznakov
fe = FeatureExtractor(return_array=False, impute_wavelet_only=True,
                      zscore_amp=True, clip_extremes=True)
print(rows_tr)
fe.fit(rows_tr)
X_df = fe.transform(rows_tr)           # pandas DF
print("Stĺpce v X_df:")
print(list(X_df.columns))
# 3) vybrané klinické príznaky
X_sel = X_df[FIS_FEATURES].to_numpy(float)

# 4) imputácia
X_imp = SimpleImputer(strategy="median").fit_transform(X_sel)

print("Pôvodné počty:", pd.Series(y_tr).value_counts())

# 5) SMOTE
smote = SMOTE(sampling_strategy={"F":1200, "S":1500, "V":1500},
              k_neighbors=1, random_state=42)
X_os, y_os = smote.fit_resample(X_imp, y_tr)
print("Po SMOTE:", pd.Series(y_os).value_counts())
# --- po extrakcii a výbere stĺpcov ----------
idx = {c: i for i, c in enumerate(FIS_FEATURES)}
for col in ["QRSd_ms","RR0_s","RR1_s","R_amplitude"]:
    print(col, X_os[:, idx[col]].min(), X_os[:, idx[col]].max())

# 6) export do ARFF
df_weka = pd.DataFrame(X_os, columns=FIS_FEATURES)
df_weka["class"] = y_os

arff_dict = {
    'description': 'MIT-BIH EKG fuzzy-classification dataset (DS1)',
    'relation':    'EKG',
    'attributes':  [(col, 'REAL') for col in FIS_FEATURES] +
                   [('class', ['N', 'S', 'V', 'F', 'Q'])],
    'data':        df_weka.values.tolist()
}

with open('train.arff', 'w') as fh:
    arff.dump(arff_dict, fh)

print("Dáta uložené do train.arff")