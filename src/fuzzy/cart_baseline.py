#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RuleFit → export pravidiel (identický formát ako CART)
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from imodels import RuleFitClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

from src.config import DATA_DIR, LEAD
from src.preprocessing.load import load_record
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer import FeatureExtractor

# ---------- split ---------------------------------------------------
DS1 = [
    "100","101","102","103","109","111","112","114","115","116",
    "117","118","121","122","123","212","214","215","231","234"
]
DS2 = [
    "104","105","106","107","108","113","119","124",
    "200","201","202","203","205","207","208","209",
    "210","213","217","219","220","221","222","223",
    "228","230","232","233"
]

# ---------- klinické črty ------------------------------------------
FIS_FEATURES = [
    "QRSd_ms","r_amp",
    "RR0_s","RR1_s","HR_bpm",
    "p_amp","t_amp",
    "E_L3","E_L4","ratio_L2_L3",
    "early_P","has_P","has_T",
]

# ---------- helper --------------------------------------------------
def build_dataset(rec_ids, fe: FeatureExtractor | None = None):
    rows, labels = [], []
    for rec_id in rec_ids:
        rec = load_record(rec_id, lead=LEAD, base_dir=DATA_DIR)
        rows.append((rec.signal, rec.fs))
        ref   = dict(zip(rec.r_peaks, rec.labels_aami))
        beats = extract_beats(rec.signal, rec.fs)
        labels.extend([ref.get(r, "Q") for r in beats["R_sample"]])
    if fe is None:
        fe = FeatureExtractor(return_array=False, impute_wavelet_only=True)
        fe.fit(rows)
    X_df = fe.transform(rows)
    return X_df.to_numpy(float), np.asarray(labels), fe

# ---------- načítaj DS-1 / DS-2 -------------------------------------
print("🔄  DS-1 …")
X_tr, y_tr, fe = build_dataset(DS1)
print("shape:", X_tr.shape)

print("🔄  DS-2 …")
X_te, y_te, _ = build_dataset(DS2, fe=fe)
print("shape:", X_te.shape)

feat_names = fe.feature_names_
idx_fis    = [feat_names.index(f) for f in FIS_FEATURES]

# ---------- imputácia + škálovanie ---------------------------------
imp     = SimpleImputer(strategy="median")
scaler  = StandardScaler(with_mean=False)

X_tr_imp = scaler.fit_transform(imp.fit_transform(X_tr))[:, idx_fis]
X_te_imp = scaler.transform(imp.transform(X_te))[:, idx_fis]

# ---------- targeted SMOTE (F,S,V) ----------------------------------
smote = SMOTE(
    sampling_strategy={"F":1200,"S":1500,"V":1500},
    k_neighbors=1,
    random_state=42,
)
X_os, y_os = smote.fit_resample(X_tr_imp, y_tr)

# ---------- RuleFitClassifier --------------------------------------
rulefit = RuleFitClassifier(
    max_rules=300,
    tree_size=4,
    sample_fract="default",
    lin_standardise=True,
    random_state=42,
)
rulefit.fit(X_os, y_os, feature_names=FIS_FEATURES)

# ---------- vyhodnotenie -------------------------------------------
y_pred = rulefit.predict(X_te_imp)
print("\n📊  RuleFit na DS-2")
print("macro-F1 =", f1_score(y_te, y_pred, average="macro"))
print(classification_report(y_te, y_pred, digits=3))

# ---------- extrakcia pravidiel ------------------------------------
rules_df = rulefit.get_rules()
rules_df = rules_df[(rules_df.coef != 0) & (rules_df.type == "rule")]
print(f"\n📜  Extrahovaných {len(rules_df)} pravidiel")

rules = []
for _, row in rules_df.iterrows():
    conds = []
    for atom in re.findall(r"\\(([^)]+)\\)", row["rule"]):
        feat, op, thr = re.split(r"(<=|>=|<|>|==)", atom)
        conds.append((feat.strip(), op, float(thr)))
    rules.append({
        "conds":   conds,
        "target":  row["pred"],
        "support": abs(float(row["coef"]))   # váha = |β|
    })

outf = Path("cache/cart_rules.npy")
outf.parent.mkdir(parents=True, exist_ok=True)
np.save(outf, rules, allow_pickle=True)
print(f"\n📦  Uložených {len(rules)} pravidiel → {outf}")