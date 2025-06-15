#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
======= RULE MINER v1  =======
 * MIT-BIH DS-1  vs.  DS-2
 * impute → targeted SMOTE → Balanced RF (importance)
 * One-Vs-Rest CART (max_depth=3)  ➜  pravidlá pre fuzzy systém
 * výstup:  cache/cart_rules.npy   (zoznam dictov)
"""

from __future__ import annotations
from pathlib import Path
import re, time, json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from imblearn.over_sampling import SMOTE
from imblearn.ensemble      import BalancedRandomForestClassifier
from sklearn.impute         import SimpleImputer
from sklearn.tree           import DecisionTreeClassifier, _tree, export_text
from sklearn.preprocessing  import StandardScaler
from sklearn.multiclass     import OneVsRestClassifier
from sklearn.metrics        import classification_report, f1_score

# --- tvoje interné moduly ------------------------------------------
from src.config                          import DATA_DIR, LEAD
from src.preprocessing.load             import load_record          # <-- len(rec_id)
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer  import FeatureExtractor
# -------------------------------------------------------------------

# ≡≡≡  CONSTANTS  ≡≡≡-------------------------------------------------
DS1 = ["100","101","102","103","109","111","112","114","115","116",
       "117","118","121","122","123","212","214","215","231","234"]
DS2 = ["104","105","106","107","108","113","119","124","200","201",
       "202","203","205","207","208","209","210","213","217","219",
       "220","221","222","223","228","230","232","233"]

FIS_FEATURES = [
    "QRSd_ms","r_amp",
    "RR0_s","RR1_s","HR_bpm",
    "p_amp","t_amp",
    "E_L3","E_L4","ratio_L2_L3",
    "early_P","has_P","has_T",
]

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)

# ≡≡≡  1)  LOAD & FEATURE EXTRACTION  ≡≡≡----------------------------
def load_rows(rec_ids: list[str]) -> tuple[list[tuple[np.ndarray,int]], np.ndarray]:
    rows, labels = [], []
    for rid in tqdm(rec_ids, desc="load"):
        rec = load_record(rid)                      # <-- len(rec_id)
        rows.append((rec.signal, rec.fs))
        ref = dict(zip(rec.r_peaks, rec.labels_aami))
        beats = extract_beats(rec.signal, rec.fs)
        labels.extend([ref.get(r, "Q") for r in beats["R_sample"]])
    return rows, np.asarray(labels)

tr_rows, y_tr_raw = load_rows(DS1)
te_rows, y_te     = load_rows(DS2)

fe = FeatureExtractor(return_array=False, impute_wavelet_only=True)
fe.fit(tr_rows)

X_tr = fe.transform(tr_rows).to_numpy(float)
X_te = fe.transform(te_rows).to_numpy(float)

feat_names = fe.feature_names_
idx_fis    = [feat_names.index(f) for f in FIS_FEATURES]

print("DS-1:", X_tr.shape, "DS-2:", X_te.shape)

# 2) IMPUTE + TARGETED SMOTE ─────────────────────────────────────── #
imp = SimpleImputer(strategy="median")
X_tr_imp = imp.fit_transform(X_tr)[:, idx_fis]
X_te_imp = imp.transform(X_te)    [:, idx_fis]

sm = SMOTE(
    sampling_strategy={"F": 2000, "S": 2000, "V": 2000},
    k_neighbors=1, random_state=42
)
X_bal, y_bal = sm.fit_resample(X_tr_imp, y_tr_raw)
print("After SMOTE:", dict(pd.Series(y_bal).value_counts()))

# 3) BALANCED RF (iba dôležitosť) ────────────────────────────────── #
brf = BalancedRandomForestClassifier(
        n_estimators=300, n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
     ).fit(X_bal, y_bal)

importances = brf.feature_importances_
print("TOP importances:", [FIS_FEATURES[i] for i in np.argsort(importances)[-6:][::-1]])

# 4) ONE-VS-REST CART (depth ≤ 3) ─────────────────────────────────── #
cart = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42
)
ovr = OneVsRestClassifier(cart).fit(X_bal, y_bal)

y_pred = ovr.predict(X_te_imp)
macro = f1_score(y_te, y_pred, average="macro")
print(f"\n📊 OVR-CART – macro-F1 = {macro:.3f}")
print(classification_report(y_te, y_pred, digits=3))

# 5) PRAVIDLÁ → DICT ─────────────────────────────────────────────── #
def tree_to_rules(tree: DecisionTreeClassifier, cls_label: str):
    out = []
    def walk(node:int, path:list):
        if tree.tree_.feature[node] == _tree.TREE_UNDEFINED:
            out.append({
                "conds":   path.copy(),
                "target":  cls_label,
                "support": int(tree.tree_.n_node_samples[node])
            })
            return
        fname = FIS_FEATURES[tree.tree_.feature[node]]
        thr   = float(tree.tree_.threshold[node])
        walk(tree.tree_.children_left[node],  path + [(fname,"<=",thr)])
        walk(tree.tree_.children_right[node], path + [(fname,">", thr)])
    walk(0, [])
    return out

all_rules = []
for est, cls in zip(ovr.estimators_, ovr.classes_):
    all_rules.extend(tree_to_rules(est, cls))

out_f = CACHE / "cart_rules.npy"
np.save(out_f, all_rules, allow_pickle=True)
print(f"\n✅ Uložené {len(all_rules)} pravidiel  →  {out_f}")

# 6) (voliteľné) pokrytie na DS-2 ─────────────────────────────────── #
def rule_applies(conds, x_vec):
    """Jednoduchá evaluácia & (AND) nad vektrom už po imputácii."""
    feat_idx = {f:i for i,f in enumerate(FIS_FEATURES)}
    for feat, op, thr in conds:
        v = x_vec[feat_idx[feat]]
        if   op == "<=" and not (v <= thr): return False
        elif op == ">"  and not (v >  thr): return False
    return True

covered = defaultdict(int); total = defaultdict(int)
for xi, yi in zip(X_te_imp, y_te):
    total[yi] += 1
    if any(rule_applies(r["conds"], xi) for r in all_rules):
        covered[yi] += 1

print("\n🧮 Pokrytie pravidlami:")
for cls in sorted(total):
    pct = covered[cls]/total[cls]*100
    print(f"  {cls:>2}: {covered[cls]:5}/{total[cls]:5}  ({pct:4.1f} %)")

# 7) Ukáž jeden strom (lekársky prehľad) ──────────────────────────── #
print("\n──── Ukážkový CART (trieda", ovr.classes_[0], ") ────")
print(export_text(ovr.estimators_[0], feature_names=FIS_FEATURES))