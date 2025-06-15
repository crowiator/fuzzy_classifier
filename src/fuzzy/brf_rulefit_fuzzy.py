#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced RF  ➜  RuleFit  ➜  export pravidiel
------------------------------------------------
* FIXNÁ klinická sada čŕt (FIS_FEATURES)
* Cielený SMOTE len pre minoritné triedy F, S, V
* RuleFitClassifier dá ~50 pravidiel + lineárnu časť
* Pravidlá a ich koeficienty sa uložia do YAML, aby sa dali mapovať na fuzzy
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, f1_score
from imodels import RuleFitClassifier as RuleFit

from src.config import DATA_DIR, LEAD
from src.preprocessing.load import load_record
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer import FeatureExtractor

# ---------------- dátový split -----------------
DS1 = [
    "100", "101", "102", "103", "109", "111", "112", "114", "115", "116",
    "117", "118", "121", "122", "123", "212", "214", "215", "231", "234",
]
DS2 = [
    "104", "105", "106", "107", "108", "113", "119", "124",
    "200", "201", "202", "203", "205", "207", "208", "209",
    "210", "213", "217", "219", "220", "221", "222", "223",
    "228", "230", "232", "233",
]

# -------- klinické črty pre fuzzy / RuleFit -----
FIS_FEATURES = [
    "QRSd_ms", "r_amp",
    "RR0_s", "RR1_s", "HR_bpm",
    "p_amp", "t_amp",
    "E_L3", "E_L4", "ratio_L2_L3",
    "early_P", "has_P", "has_T",
]

# ----------------------------------------------------------------------

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

# ---------------- načítanie DS1 / DS2 ----------------
print("🔄  DS-1 …")
X_tr, y_tr, fe = build_dataset(DS1)
print("shape:", X_tr.shape)

print("🔄  DS-2 …")
X_te, y_te, _ = build_dataset(DS2)
print("shape:", X_te.shape)

feat_names = fe.feature_names_
idx_fis    = [feat_names.index(f) for f in FIS_FEATURES]

# ---------------- imputácia mediánom -----------------
imp = SimpleImputer(strategy="median")
X_tr_imp = imp.fit_transform(X_tr)
X_te_imp = imp.transform(X_te)

# -------- targeted SMOTE len pre F,S,V ---------------
smote = SMOTE(
    sampling_strategy={"F": 1200, "S": 1500, "V": 1500},
    k_neighbors=1,
    random_state=42,
)
X_os, y_os = smote.fit_resample(X_tr_imp[:, idx_fis], y_tr)

# ---------------- RuleFit pipeline -------------------
pipe = make_pipeline(
    StandardScaler(with_mean=False),
    RuleFit(
        max_rules=300,
        tree_size=4,
        rfmode="classify",
        lin_standardise=True,
        sample_fract="default",
        random_state=42,
    ),
)
pipe.fit(X_os, y_os)

rulefit: RuleFit = pipe.named_steps["rulefit"]

# ---------------- vyhodnotenie -----------------------
y_pred = pipe.predict(X_te_imp[:, idx_fis])
print("\n📊  RuleFit na DS-2")
print("macro-F1 =", f1_score(y_te, y_pred, average="macro"))
print(classification_report(y_te, y_pred, digits=3))

# ---------------- extrakcia pravidiel ----------------
rules_df = rulefit.get_rules()
rules_df = rules_df[(rules_df.coef != 0) & (rules_df.type == "rule")]
print(f"\n📜  Extrahovaných {len(rules_df)} pravidiel (coef≠0)")

# uložíme do YAML
import yaml, re

fuzzy_ready = []
for _, row in rules_df.iterrows():
    conds = []
    # rozparsuj podmienky typu "(RR0_s<=0.55) & (QRSd_ms>120.0)"
    for atom in re.findall(r"\(([^\)]+)\)", row["rule"]):
        feat, op, thr = re.split(r"(<=|>=|<|>|==)", atom)
        conds.append({"feat": feat.strip(), "op": op, "thr": float(thr)})
    fuzzy_ready.append({
        "conds":  conds,
        "target": row["pred"] ,
        "coef":   float(row["coef"])
    })

out_yaml = Path("cache/rulefit_rules.yaml")
out_yaml.parent.mkdir(parents=True, exist_ok=True)
with out_yaml.open("w", encoding="utf-8") as f:
    yaml.safe_dump(fuzzy_ready, f, sort_keys=False, allow_unicode=True)
print(f"\n📦  YAML s pravidlami uložený → {out_yaml}")