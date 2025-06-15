#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-vs-Rest RuleFit  →  fuzzy-friendly pravidlá
------------------------------------------------
• každá menšinová trieda (F,S,V,Q) sa učí proti zvyšku
• po tréningu sa pravidlá preriešia podľa podpory/dĺžky
• export = list[dict] {"conds", "target", "support"}
"""
from __future__ import annotations
from pathlib import Path
import re, time, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from imodels import RuleFitClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from src.config import DATA_DIR, LEAD
from src.preprocessing.load import load_record
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer import FeatureExtractor

warnings.filterwarnings("ignore", category=FutureWarning)  # čistejší výstup

# ------------------------------ nastavenia ---------------------------------
DS1 = ["100","101","102","103","109","111","112","114","115","116",
       "117","118","121","122","123","212","214","215","231","234"]
DS2 = ["104","105","106","107","108","113","119","124","200","201","202",
       "203","205","207","208","209","210","213","217","219","220","221",
       "222","223","228","230","232","233"]

FIS_FEATURES = ["QRSd_ms","r_amp","RR0_s","RR1_s","HR_bpm",
                "p_amp","t_amp","E_L3","E_L4","ratio_L2_L3",
                "early_P","has_P","has_T"]

MINORITY_CLASSES = ["F", "S", "V", "Q"]                 # binárne RuleFity
MAX_ATOMS        = 4                                    # max dĺžka pravidla
MIN_SUPPORT      = 25                                   # min beatov v DS-1
N_RULES_PER_CLS  = 60                                   # max export / trieda
CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(exist_ok=True, parents=True)
# ---------------------------------------------------------------------------

def load_rows(rec_ids: list[str]) -> tuple[list[tuple[np.ndarray,int]], np.ndarray]:
    rows, y = [], []
    for rid in tqdm(rec_ids, desc="load"):
        rec = load_record(rid, LEAD, DATA_DIR)
        rows.append((rec.signal, rec.fs))
        m = dict(zip(rec.r_peaks, rec.labels_aami))
        beats = extract_beats(rec.signal, rec.fs)
        y.extend([m.get(r,"Q") for r in beats["R_sample"]])
    return rows, np.asarray(y)

# 1) načítaj & featurizuj ----------------------------------------------------
tr_rows, y_tr = load_rows(DS1)
te_rows, y_te = load_rows(DS2)

fe = FeatureExtractor(return_array=False, impute_wavelet_only=True).fit(tr_rows)
X_tr = fe.transform(tr_rows)[FIS_FEATURES].to_numpy(float)
X_te = fe.transform(te_rows )[FIS_FEATURES].to_numpy(float)

imp     = SimpleImputer(strategy="median").fit(X_tr)
scaler  = StandardScaler(with_mean=False).fit(imp.transform(X_tr))

X_tr = scaler.transform(imp.transform(X_tr))
X_te = scaler.transform(imp.transform(X_te))

# 2) One-vs-Rest RuleFit + SMOTE pre každú minoritnú triedu -----------------
all_rules, y_pred = [], np.full_like(y_te, fill_value="N", dtype=object)

for tgt in MINORITY_CLASSES:
    print(f"\n=== Training RuleFit for {tgt} vs rest ===")
    y_bin = (y_tr == tgt).astype(int)

    # vyváž len ak je minoritná trieda < 5 % všetkých
    pos = (y_bin == 1).sum()
    if pos < 0.05*len(y_bin):
        sm = SMOTE(sampling_strategy={1: min(max(800,pos*4), 2000)},
                   k_neighbors=1, random_state=42)
        X_bal, y_bal = sm.fit_resample(X_tr, y_bin)
    else:
        X_bal, y_bal = X_tr, y_bin

    rf = RuleFitClassifier(max_rules=400, tree_size=4,
                           lin_standardise=False, random_state=42, n_jobs=-1)
    rf.fit(X_bal, y_bal, feature_names=FIS_FEATURES)

    # --------- zbieraj pravidlá -----------------
    rules_df = rf.get_rule_dataframe()
    rules_df = rules_df[(rules_df.coef != 0) & (rules_df.type == "rule")]

    good = []
    for _, row in rules_df.iterrows():
        atoms = re.findall(r"\(([^)]+)\)", row["rule"])
        if not (1 <= len(atoms) <= MAX_ATOMS):
            continue
        conds = []
        for atom in atoms:
            f, op, th = re.split(r"(<=|>=|<|>|==)", atom)
            conds.append((f.strip(), op, float(th)))
        # podpora = koeficent * n_beats (absolútna hodnota)
        supp = abs(float(row["coef"])) * len(y_bin)
        good.append({"conds": conds, "target": tgt, "support": supp})

    # triedenie: podpora/len(conds) →  početné, ale krátke
    good.sort(key=lambda d: -(d["support"] / len(d["conds"])))
    good = [r for r in good if r["support"] >= MIN_SUPPORT][:N_RULES_PER_CLS]
    all_rules.extend(good)

    # --------- predikcia na DS-2 ----------------
    if good:
        satisfied = np.zeros(len(y_te), dtype=bool)
        for rule in good:
            mask = np.ones(len(y_te), dtype=bool)
            for feat, op, thr in rule["conds"]:
                col = FIS_FEATURES.index(feat)
                if   op == "<=": mask &= X_te[:,col] <= thr
                elif op == ">=": mask &= X_te[:,col] >= thr
                elif op == "<":  mask &= X_te[:,col] <  thr
                elif op == ">":  mask &= X_te[:,col] >  thr
                elif op == "==": mask &= X_te[:,col] == thr
            satisfied |= mask
        y_pred[satisfied] = tgt    # ak sa splní aspoň 1 rule

# 3) hodnotenie -- len informatívne -----------------------------------------
print("\n📊  One-vs-Rest rules on DS-2")
print("macro-F1 =", f1_score(y_te, y_pred, average="macro").round(3))
print(classification_report(y_te, y_pred, digits=3))

# 4) ulož pravidlá pre fuzzy pipeline ---------------------------------------
out_path = CACHE_DIR / "cart_rules.npy"
np.save(out_path, all_rules, allow_pickle=True)
print(f"📦  Uložených {len(all_rules)} pravidiel  ➜  {out_path}")