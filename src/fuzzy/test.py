#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRF → RuleFit → export pravidiel pre fuzzy klasifikátor
-------------------------------------------------------
🎯  Cieľ pre ťaženie pravidiel: ~0.50 macro-F1 (DS-2) AND
    rozumná podpora minoritných tried F / S / V.

Skript:
  • extrahuje / kešuje črty
  • SMOTE minoritných tried (F,S,V)
  • ťaží pravidlá cez RuleFitClassifier (one-vs-rest)
  • ukladá ich do cache/cart_rules.npy
  • vypisuje klasickú metriky + krížovú tabuľku predikcií
"""

# ---------- importy ------------------------------------------------
from __future__ import annotations
from pathlib import Path
import re, time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from imblearn.over_sampling import SMOTE
from imodels import RuleFitClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import DATA_DIR, LEAD
from src.preprocessing.load import load_record
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer import FeatureExtractor
# -------------------------------------------------------------------

# ---------- 0. split + konštanty -----------------------------------
DS1 = ["100","101","102","103","109","111","112","114","115","116",
       "117","118","121","122","123","212","214","215","231","234"]
DS2 = ["104","105","106","107","108","113","119","124",
       "200","201","202","203","205","207","208","209",
       "210","213","217","219","220","221","222","223",
       "228","230","232","233"]

FIS_FEATURES = [
    "QRSd_ms","r_amp","RR0_s","RR1_s","HR_bpm",
    "p_amp","t_amp","E_L3","E_L4","ratio_L2_L3",
    "early_P","has_P","has_T",
]

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)

# ---------- 1. helper na (de)kódovanie ------------------------------
def _save_rows(path: Path, rows: list[tuple]) -> None:
    np.save(path, np.array(rows, dtype=object), allow_pickle=True)

def _load_rows(path: Path) -> list[tuple]:
    arr = np.load(path, allow_pickle=True)
    # z (n,2) objektového ndarray urob späť list[(sig,fs)]
    return [tuple(pair) for pair in arr.tolist()]


def extract_dataset(rec_ids: list[str], tag: str):
    feats_f, labels_f = CACHE/f"{tag}_feats.npy", CACHE/f"{tag}_labels.npy"
    if feats_f.exists() and labels_f.exists():
        return _load_rows(feats_f), np.load(labels_f)

    rows, labels = [], []
    for rid in tqdm(rec_ids, desc=f"extract {tag}"):
        rec   = load_record(rid, lead=LEAD, base_dir=DATA_DIR)
        rows.append((rec.signal, rec.fs))
        beats = extract_beats(rec.signal, rec.fs)
        ref   = dict(zip(rec.r_peaks, rec.labels_aami))
        labels.extend([ref.get(r, "Q") for r in beats["R_sample"]])

    _save_rows(feats_f, rows); np.save(labels_f, labels)
    return rows, np.asarray(labels)
# --------------------------------------------------------------------

# ---------- 2. Načítaj & featury ------------------------------------
tr_rows, y_tr = extract_dataset(DS1, "ds1")
te_rows, y_te = extract_dataset(DS2, "ds2")

fe = FeatureExtractor(return_array=False, impute_wavelet_only=True)
fe.fit(tr_rows)

X_tr = fe.transform(tr_rows).to_numpy(float)
X_te = fe.transform(te_rows ).to_numpy(float)
feat_names = fe.feature_names_
idx_fis    = [feat_names.index(f) for f in FIS_FEATURES]

# ---------- 3. Imputácia + SMOTE  -----------------------------------
imp   = SimpleImputer(strategy="median")
X_tr_ = imp.fit_transform(X_tr)[:, idx_fis]
X_te_ = imp.transform(X_te)    [:, idx_fis]

smote = SMOTE({"F":1200,"S":1500,"V":1500}, k_neighbors=1, random_state=42)
X_os, y_os = smote.fit_resample(X_tr_, y_tr)
print("After SMOTE counts:", dict(pd.Series(y_os).value_counts()))

# ---------- 4. (vol.) Balanced RF len na info -----------------------
brf = BalancedRandomForestClassifier(
          n_estimators=300, n_jobs=-1,
          class_weight="balanced_subsample", random_state=42
      ).fit(X_os, y_os)

# ---------- 5. RuleFit – one-vs-rest --------------------------------
scaler   = StandardScaler(with_mean=False).fit(X_os)
X_os_z   = scaler.transform(X_os)
X_te_z   = scaler.transform(X_te_)

rulefit  = RuleFitClassifier(max_rules=300, tree_size=4,
                             lin_standardise=True, random_state=42,
                             ovr=True)             # ⬅️  dôležité!
t0 = time.time()
rulefit.fit(X_os_z, y_os, feature_names=FIS_FEATURES)
print(f"RuleFit OvR fitted in {time.time()-t0:.1f}s")

# ---------- 6. Vyhodnotenie + crosstab ------------------------------
y_pred = rulefit.predict(X_te_z)
print("\n📊 OvR-RuleFit on DS-2")
print("macro-F1 =", f1_score(y_te, y_pred, average="macro").round(3))
print(classification_report(y_te, y_pred, digits=3))

print("\n🔍 Krížová tabuľka (y_true × y_pred):")
tbl = pd.crosstab(y_te, y_pred, dropna=False)
print(tbl)                          # ⬅️  teraz ju vidíš v konzole

# ---------- 7. Extrahuj a ulož pravidlá -----------------------------
rules_df = rulefit._get_rules()               # imodels >=0.3
rules_df = rules_df[(rules_df.coef != 0) & (rules_df.type == "rule")]

rules = []
for _, row in rules_df.iterrows():
    atoms = re.findall(r"\(([^)]+)\)", row["rule"])
    conds = [ (re.split(r"(<=|>=|<|>|==)", a)[0].strip(),
               re.split(r"(<=|>=|<|>|==)", a)[1],
               float(re.split(r"(<=|>=|<|>|==)", a)[2]) ) for a in atoms ]
    rules.append({"conds": conds,
                  "target": row["pred"],
                  "support": abs(float(row["coef"])) })

out_f = CACHE/"cart_rules.npy"
np.save(out_f, rules, allow_pickle=True)
print(f"\n📦  Uložených {len(rules)} pravidiel  → {out_f}")