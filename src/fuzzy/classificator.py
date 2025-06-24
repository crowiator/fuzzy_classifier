#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuzzy trapézový klasifikátor pre MIT-BIH (DS-2)
----------------------------------------------
* pravidlá z furia_rules.txt  → trapézové MF
* pravidlo = AND všetkých MF, váha = CF
* výsledok: classification_report, macro-F1
"""
from __future__ import annotations

# ── štandard ─────────────────────────────────────────────────────────────
from pathlib import Path
from functools import reduce
from multiprocessing import Pool, cpu_count, set_start_method
import re, pickle, warnings, os

# ── 3rd-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, f1_score

# ── tvoje moduly ─────────────────────────────────────────────────────────
from src.config                      import DATA_DIR, LEAD
from src.preprocessing.load          import load_record
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer  import FeatureExtractor

# ════════════════════════════════════════════════════════════════════════
# 1  Nastavenia a mená stĺpcov
# ════════════════════════════════════════════════════════════════════════
FIS_FEATURES = [
    'R_amplitude', 'P_amplitude', 'T_amplitude', 'RR0_s', 'RR1_s', 'Heart_rate_bpm',
    'QRSd_ms', 'PR_ms', 'early_P', 'r_amp_raw', 'qrs_pol', 't_pol',
    'E_L2', 'E_L3', 'E_L4', 'E_L5', 'E_L6',
    'ratio_L2_L3', 'ratio_P_T',
    'nan_E_L2', 'nan_E_L3', 'nan_E_L4', 'nan_E_L5', 'nan_E_L6',
    'nan_ratio_L2_L3',
    'nan_QRSd_ms', 'nan_P_amplitude', 'nan_T_amplitude'  # pridané nové príznaky
]

DS2 = [                     # testovacie záznamy
    "104","105","106","107","108","113","119","124",
    "200","201","202","203","205","207","208","209",
    "210","213","217","219","220","221","222","223",
    "228","230","232","233"
]

# ════════════════════════════════════════════════════════════════════════
# 2  Načítanie DS-2 a extrakcia príznakov
# ════════════════════════════════════════════════════════════════════════
CACHE = Path("cache");  CACHE.mkdir(exist_ok=True)

def build_ds(rec_ids, tag):
    CACHE = Path("cache")
    CACHE.mkdir(exist_ok=True)
    pkl_path = CACHE / f"{tag}.pkl"

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            X, y, feat_names = pickle.load(f)
        if len(feat_names) != len(FIS_FEATURES) or set(feat_names) != set(FIS_FEATURES):
            print(f"Nesúlad v cache ({len(feat_names)} vs {len(FIS_FEATURES)}), regenerujem dataset.")
            pkl_path.unlink()  # Vymaže nekorektný súbor
        else:
            return X, y, feat_names

    rows, y = [], []
    for rid in tqdm(rec_ids, desc=f"load {tag}"):
        rec = load_record(rid, lead=LEAD, base_dir=DATA_DIR)
        rows.append((rec.signal, rec.fs, rec.r_peaks))
        ref = dict(zip(rec.r_peaks, rec.labels_aami))
        beats = extract_beats(rec.signal, rec.fs,
                              r_idx=rec.r_peaks,
                              zscore_amp=True, clip_extremes=True)
        y.extend([ref.get(r, "Q") for r in beats["R_sample"]])

    fe = FeatureExtractor(return_array=False, impute_wavelet_only=True)
    fe.fit(rows)
    X_df = fe.transform(rows)

    # Odstrániť nepotrebné stĺpce
    X_df = X_df.drop(columns=["is_PAC", "R_sample", "beat_idx"], errors='ignore')

    # Získaj aktualizované názvy príznakov priamo z DF
    feat_names = X_df.columns.tolist()

    # Diagnostické výpisy pre kontrolu
    print("Extracted features:", feat_names)
    print("Expected features (FIS_FEATURES):", FIS_FEATURES)
    print("Missing in FIS_FEATURES:", set(feat_names) - set(FIS_FEATURES))
    print("Extra in FIS_FEATURES:", set(FIS_FEATURES) - set(feat_names))

    if len(feat_names) != len(FIS_FEATURES):
        raise ValueError(
            f"Nesúlad medzi extrahovanými a očakávanými príznakmi: {len(feat_names)} vs {len(FIS_FEATURES)}"
        )

    # Finálne uloženie dát
    X = X_df.to_numpy(float)
    with open(pkl_path, "wb") as f:
        pickle.dump((X, np.asarray(y), feat_names), f)
    return X, np.asarray(y), feat_names

X_te, y_te, feat_names = build_ds(DS2, "ds2")
print("Počet stĺpcov v X_te:", X_te.shape[1])
print("Počet feat_names:", len(feat_names))
print("feat_names:", feat_names)
df_te   = pd.DataFrame(X_te, columns=feat_names)

# ════════════════════════════════════════════════════════════════════════
# 3  Ranges dynamicky z dát
# ════════════════════════════════════════════════════════════════════════
ranges: dict[str, tuple[float,float]] = {}        # ← čistý dict
for col in FIS_FEATURES:
    if col not in df_te.columns:
        continue
    lo, hi = df_te[col].min(skipna=True), df_te[col].max(skipna=True)
    if np.isfinite(lo) and np.isfinite(hi):
        pad = 0.05 * (hi - lo) or 1e-3
        ranges[col] = (float(lo - pad), float(hi + pad))
    else:
        ranges[col] = (0.0, 1.0)

col_idx  = {c:i for i,c in enumerate(feat_names)}
columns  = [c for c in feat_names if c in ranges]   ### FIX – až po ranges
# indexy len pre použiteľné stĺpce
_col_idx = {c: col_idx[c] for c in columns}

# ════════════════════════════════════════════════════════════════════════
# 4  Fuzzy systém podľa pravidiel FURIA
# ════════════════════════════════════════════════════════════════════════
sys = ctrl.ControlSystem()
FS  = ctrl.ControlSystemSimulation(sys)

def add_trapmf(var:str, lo2:float, hi1:float):
    lo, hi = ranges[var]
    if not hasattr(FS, var):
        u = np.linspace(lo, hi, 1000)
        setattr(FS, var, ctrl.Antecedent(u, var))
    ante = getattr(FS, var)
    name = f"{var}_{len(ante.terms)}"
    ante[name] = fuzz.trapmf(ante.universe, [lo2,lo2,hi1,hi1])
    return ante[name]

# výstup Class
u = np.arange(1,6,1)
Class = ctrl.Consequent(u, "Class")
for i,lbl in enumerate(["N","S","V","F","Q"],1):
    Class[lbl] = fuzz.trimf(u,[i,i,i])
setattr(FS,"Class",Class)

def clamp(v:str|float, var:str)->float:
    if v=="inf":   return ranges[var][1]
    if v=="-inf":  return ranges[var][0]
    return float(v)

rules_txt, skip_rule, skip_atom = [], 0, 0
with open("furia_rules.txt") as fh:
    for line in fh:
        if "=>" not in line: continue
        conds, rhs = line.strip().split("=>",1)
        tgt = re.search(r"class=(\w)", rhs).group(1)
        cf  = float(re.search(r"CF\s*=\s*([\d.]+)", rhs).group(1))

        atoms=[]
        for var, nums in re.findall(r"\((\w+)\s+in\s+\[([^\]]+)\]\)", conds):
            if var not in ranges or var not in _col_idx:
                skip_atom += 1;  continue
            lo2, hi1 = map(lambda x: clamp(x,var), nums.split(",")[1:3])
            atoms.append(add_trapmf(var, lo2, hi1))

        if not atoms:
            skip_rule += 1;  continue

        rule = ctrl.Rule(reduce(lambda a,b:a & b, atoms), FS.Class[tgt])
        rule.weight = cf
        sys.addrule(rule)
        rules_txt.append(line.strip())

USED_FEATURES = set(v.label for v in FS.ctrl.antecedents)
print(f"Pravidlá: {len(rules_txt)},   preskočené podmienky: {skip_atom}")

RULE_OBJS = list(sys.rules)

# ════════════════════════════════════════════════════════════════════════
# 5  Helpery
# ════════════════════════════════════════════════════════════════════════
def get_firing_value(firing_obj):
    if hasattr(firing_obj, "_sim_data"):
        vals = list(firing_obj._sim_data.values())
        for v in vals:
            if isinstance(v, (int, float, np.floating)):
                return float(v)
    if isinstance(firing_obj, (int, float, np.floating)):
        return float(firing_obj)
    raise TypeError(f"Neviem získať firing hodnotu z typu: {type(firing_obj)}")


def _strength(rule):
    """Kompatibilný firing pre všetky verzie skfuzzy."""
    try:
        return get_firing_value(rule.aggregate_firing)
    except AttributeError:
        return get_firing_value(rule.firing_strength)

def predict_row(sim:ctrl.ControlSystemSimulation, row:np.ndarray)->str:
    for v in columns:
        sim.input[v] = float(row[_col_idx[v]])
    sim.compute()
    best, cls = -1.0, "Q"
    for r in RULE_OBJS:
        sc = _strength(r)*r.weight
        if sc>best:
            best, cls = sc, r.consequent[0].term.label
    sim.reset()
    return cls

# ════════════════════════════════════════════════════════════════════════
# 6  Multiprocessing worker
# ════════════════════════════════════════════════════════════════════════
_FS_w   = None
_rule_w = None

def init_worker():
    global _FS_w, _rule_w
    sys_w  = ctrl.ControlSystem(RULE_OBJS)
    _FS_w  = ctrl.ControlSystemSimulation(sys_w)
    _rule_w = list(sys_w.rules)

def block_predict(block:np.ndarray):
    preds=[]
    for idx,row in enumerate(block,1):
        if idx%50==0:
            print(f"[PID {os.getpid()}] beat {idx}/{len(block)}")
        for v in columns:
            if v in USED_FEATURES:  # <-- kontrola
                _FS_w.input[v] = float(row[_col_idx[v]])
        _FS_w.compute()
        best, cls = -1.0, "Q"
        for r in _rule_w:
            sc = _strength(r)*r.weight
            if sc>best:
                best, cls = sc, r.consequent[0].term.label
        _FS_w.reset()
        preds.append(cls)
    return preds

# ════════════════════════════════════════════════════════════════════════
# 7  Spustenie
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_start_method("spawn", force=True)

    n_proc = max(1, cpu_count()-1)
    blocks = np.array_split(X_te, n_proc*4)

    print(f"▶  Spúšťam {n_proc} procesov …")
    with Pool(n_proc, initializer=init_worker) as pool:
        y_blocks = list(tqdm(pool.imap(block_predict, blocks),
                             total=len(blocks), desc="Klasifikujem", leave=False))

    y_pred = sum(y_blocks, [])
    print(classification_report(y_te, y_pred, digits=3))
    print("macro-F1 =", f1_score(y_te, y_pred, average="macro"))
