#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-vs-Rest RuleFit → fuzzy rules (MIT-BIH DS-1 / DS-2)

• fixed clinical features (FIS_FEATURES)
• targeted SMOTE for the rare classes
• caches raw (signal, fs) pairs with pickle – no ndarray-of-objects issues
• mines rules with imodels.RuleFitClassifier in one-vs-rest fashion
"""

from __future__ import annotations
from pathlib import Path
import time, pickle, re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from imblearn.over_sampling import SMOTE
from imodels import RuleFitClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier

from src.config import DATA_DIR, LEAD
from src.preprocessing.load import load_record
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer import FeatureExtractor

# ------------------------------------------------------------------#
# 0. IDs & constants
# ------------------------------------------------------------------#
DS1 = ["100","101","102","103","109","111","112","114","115","116",
       "117","118","121","122","123","212","214","215","231","234"]
DS2 = ["104","105","106","107","108","113","119","124",
       "200","201","202","203","205","207","208","209",
       "210","213","217","219","220","221","222","223",
       "228","230","232","233"]

FIS_FEATURES = [
    "QRSd_ms","r_amp",
    "RR0_s","RR1_s","HR_bpm",
    "p_amp","t_amp",
    "E_L3","E_L4","ratio_L2_L3",
    "early_P","has_P","has_T",
]

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)

# ------------------------------------------------------------------#
# 1. helper – load / save raw rows with pickle
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

rows_tr, y_tr = cached_rows(DS1, "ds1")
rows_te, y_te = cached_rows(DS2, "ds2")

# ------------------------------------------------------------------#
# 2. feature extraction
# ------------------------------------------------------------------#
fe = FeatureExtractor(return_array=False, impute_wavelet_only=True)
fe.fit(rows_tr)

X_tr = fe.transform(rows_tr).to_numpy(float)
X_te = fe.transform(rows_te).to_numpy(float)
feat_names = fe.feature_names_
idx = [feat_names.index(f) for f in FIS_FEATURES]

print("DS-1 shape:", X_tr.shape, "| DS-2 shape:", X_te.shape)

# ------------------------------------------------------------------#
# 3. impute + targeted SMOTE
# ------------------------------------------------------------------#
imp = SimpleImputer(strategy="median")
Xs_tr = imp.fit_transform(X_tr)[:, idx]
Xs_te = imp.transform(X_te)   [:, idx]

smote = SMOTE(sampling_strategy={"F":1200, "S":1500, "V":1500},
              k_neighbors=1, random_state=42)
Xs_os, y_os = smote.fit_resample(Xs_tr, y_tr)
print("After SMOTE:", dict(pd.Series(y_os).value_counts()))

# ------------------------------------------------------------------#
# 4. (optional) Balanced RF for quick sanity-check
# ------------------------------------------------------------------#
t0 = time.time()
brf = BalancedRandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    class_weight="balanced_subsample",
    random_state=42,
)
brf.fit(Xs_os, y_os)
print(f"Balanced RF trained in {time.time()-t0:.1f}s")
# ------------------------------------------------------------------#
# 5. RuleFit in one-vs-rest wrapper  (supports multiclass via OvR)
# ------------------------------------------------------------------#
clf = OneVsRestClassifier(
    RuleFitClassifier(max_rules=300, tree_size=4,
                      lin_standardise=True, random_state=42),
    n_jobs=-1
)

scaler = StandardScaler(with_mean=False).fit(Xs_os)
Xs_os_s = scaler.transform(Xs_os)
Xs_te_s = scaler.transform(Xs_te)

t0 = time.time()
clf.fit(Xs_os_s, y_os)
print(f"RuleFit OvR fitted in {time.time()-t0:.1f}s")

y_pred = clf.predict(Xs_te_s)
print("\n📊 OvR-RuleFit on DS-2")
print("macro-F1 =", f1_score(y_te, y_pred, average="macro"))
print(classification_report(y_te, y_pred, digits=3))

# ------------------------------------------------------------------#
# 6. collect rules from each OvR estimator
# ------------------------------------------------------------------#
all_rules = []
for est, class_label in zip(clf.estimators_, clf.classes_):
    rules_df = est.get_rules()
    rules_df = rules_df[(rules_df.coef != 0) & (rules_df.type == "rule")]
    for _, r in rules_df.iterrows():
        atoms = []
        for atom in re.findall(r"\(([^)]+)\)", r["rule"]):
            f, op, thr = re.split(r"(<=|>=|<|>|==)", atom)
            atoms.append((f.strip(), op, float(thr)))
        all_rules.append({
            "conds": atoms,
            "target": class_label,
            "support": abs(float(r["coef"]))
        })

print("Total mined rules:", len(all_rules))

# ------------------------------------------------------------------#
# 7. save for fuzzy engine
# ------------------------------------------------------------------#
out_f = CACHE / "cart_rules.npy"
np.save(out_f, all_rules, allow_pickle=True)
print(f"📦 Saved {len(all_rules)} rules → {out_f}")