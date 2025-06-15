#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fuzzy trapézový klasifikátor (presné intervaly z FURIA)
-------------------------------------------------------
• načíta 55 pravidiel z furia_rules.txt
• každý interval → vlastná trapézová MF (čisté jadro)
• pravidlá = AND všetkých atómov, váha = CF
• predikcia DS-2 → classification_report, macro-F1
"""

from __future__ import annotations
import re, numpy as np, skfuzzy as fuzz, skfuzzy.control as ctrl
from functools import reduce
from pathlib import Path
import pickle, warnings
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, f1_score
from multiprocessing import Pool, cpu_count, set_start_method
import os
# ------------- MIT-BIH helpery (používaš už vo svojom projekte) ---
from src.config                    import DATA_DIR, LEAD
from src.preprocessing.load        import load_record
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer  import FeatureExtractor

# ---------- 1. rozsahy (min–max z DS-1) ---------------------------
ranges = {"QRSd_ms":(55.6,500.0),"RR0_s":(0.306,5.956),"RR1_s":(0.306,5.956),
          "HR_bpm":(10.075,196.364),"p_amp":(-1.573,1.958),"t_amp":(-0.783,2.224),
          "E_L3":(42.416,3996.714),"E_L4":(1244.19,28631.18),"ratio_L2_L3":(0.006,0.017),
          "r_amp":(-0.85,3.817),"early_P":(0,1),"has_P":(0,1),"has_T":(0,1)}

# ---------- 2. načítaj DS-2 ---------------------------------------
DS2 = ["104","105","106","107","108","113","119","124","200","201","202","203","205",
       "207","208","209","210","213","217","219","220","221","222","223","228","230","232","233"]

def build_ds(rec_ids, tag):
    cache = Path("cache"); cache.mkdir(exist_ok=True)
    f = cache / f"{tag}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))

    rows, y = [], []
    for rid in tqdm(rec_ids, desc=f"load {tag}"):
        rec = load_record(rid, LEAD, DATA_DIR)
        rows.append((rec.signal, rec.fs))
        ref = dict(zip(rec.r_peaks, rec.labels_aami))
        beats = extract_beats(rec.signal, rec.fs)
        y.extend([ref.get(r, "Q") for r in beats["R_sample"]])
    fe = FeatureExtractor(return_array=False, impute_wavelet_only=True)
    fe.fit(rows)
    X = fe.transform(rows).to_numpy(float)
    pickle.dump((X, np.asarray(y), fe.feature_names_), open(f, "wb"))
    return X, np.asarray(y), fe.feature_names_

X_te, y_te, feat_names = build_ds(DS2, "ds2")
col_idx = {c:i for i,c in enumerate(feat_names)}
columns = [c for c in feat_names if c in ranges]

# ---------- 3. fuzzy systém ---------------------------------------
sys = ctrl.ControlSystem([])
FS  = ctrl.ControlSystemSimulation(sys)



def add_trapmf(var: str, lo2: float, hi1: float):
    """Pridá (ak treba) Antecedent a novú trapézovú MF, vráti MF objekt."""
    lo, hi = ranges[var]
    if not hasattr(FS, var):
        universe = np.linspace(lo, hi, 1000)
        ante = ctrl.Antecedent(universe, var)
        setattr(FS, var, ante)
    ante = getattr(FS, var)
    name = f"{var}_{len(ante.terms)}"
    ante[name] = fuzz.trapmf(ante.universe, [lo2, lo2, hi1, hi1])
    return ante[name]

# výstup Class (singletons 1–5)
u = np.arange(1, 6, 1)
Class = ctrl.Consequent(u, "Class")
for i, lbl in enumerate(["N", "S", "V", "F", "Q"], 1):
    Class[lbl] = fuzz.trimf(u, [i, i, i])
setattr(FS, "Class", Class)

# ---------- 4. načítaj a pridaj pravidlá --------------------------
def clamp(val: float | str, var: str) -> float:
    if val == "inf":   return ranges[var][1]
    if val == "-inf":  return ranges[var][0]
    return float(val)

rules, skipped_rules, skipped_atoms = [], 0, 0
with open("furia_rules.txt") as fh:
    for ln_no, line in enumerate(fh, 1):
        line = line.strip()
        if not line or "=>" not in line:
            skipped_rules += 1
            continue
        conds_txt, rhs = line.split("=>", 1)
        try:
            tgt = re.search(r"class=(\w)", rhs).group(1)
            cf  = float(re.search(r"CF\s*=\s*([0-9.]+)", rhs).group(1))
        except Exception:
            skipped_rules += 1
            continue

        atoms = []
        for var, numstr in re.findall(r"\((\w+)\s+in\s+\[([^\]]+)\]\)", conds_txt):
            if var not in ranges or var not in col_idx:
                skipped_atoms += 1; continue
            nums = [clamp(v.strip(), var) for v in numstr.split(",")]
            if len(nums) != 4: skipped_atoms += 1; continue
            lo2, hi1 = nums[1], nums[2]
            atoms.append(add_trapmf(var, lo2, hi1))

        if not atoms:
            skipped_rules += 1
            continue

        ante = reduce(lambda a, b: a & b, atoms)

        rule = ctrl.Rule(ante, FS.Class[tgt])
        rule.weight = cf  # priradenie váhy (0–1)
        sys.addrule(rule)
        rules.append(line)
rule_lines = rules.copy()
print(f"Načítaných pravidiel: {len(rules)}")
print(f"Preskočené pravidlá:  {skipped_rules}")
print(f"Preskočené podmienky: {skipped_atoms}")
rule_objs = list(sys.rules)

# ---------- 5. predikcia pre každý úder ---------------------------
columns = [c for c in feat_names if c in ranges]
map_cls = {1: "N", 2: "S", 3: "V", 4: "F", 5: "Q"}

def get_firing_value(firing_obj):
    if hasattr(firing_obj, "_sim_data"):
        vals = list(firing_obj._sim_data.values())
        for v in vals:
            if isinstance(v, (int, float, np.floating)):
                return float(v)
    if isinstance(firing_obj, (int, float, np.floating)):
        return float(firing_obj)
    raise TypeError(f"Neviem získať firing hodnotu z typu: {type(firing_obj)}")


def predict_row(row: np.ndarray) -> str:
    for v in columns:
        FS.input[v] = float(row[col_idx[v]])

    FS.compute()

    best_val, best_cls = -1.0, "Q"
    for r in rule_objs:
        try:
            act = get_firing_value(r.aggregate_firing)
        except Exception:
            act = get_firing_value(r.firing_strength)
        score = act * r.weight
        if score > best_val:
            best_val = score
            best_cls = r.consequent[0].term.label

    FS.reset()
    return best_cls

# ---------------------------------------------------------
# Multiprocessing helpers – MUSIA byť na module-level
# ---------------------------------------------------------
_FS        = None     # kópia ControlSystemSimulation v každom procese
_rule_objs = None     # list pravidiel v tom procese
_columns   = None     # zoznam vstupných mien
_col_idx   = None     # map name → index v X
_map_cls   = {1:'N', 2:'S', 3:'V', 4:'F', 5:'Q'}

def _strength(rule):
    """Kompatibilný firing pre všetky verzie skfuzzy."""
    try:
        return get_firing_value(rule.aggregate_firing)
    except AttributeError:
        return get_firing_value(rule.firing_strength)

def init_worker(rule_lines, columns, col_idx):
    """
    Spustí sa raz pri štarte každého procesu.
    Postaví si *vlastný* fuzzy systém podľa rule_lines.
    """
    global _FS, _rule_objs, _columns, _col_idx

    _columns = columns
    _col_idx = col_idx

    sys_w = ctrl.ControlSystem([])
    _FS   = ctrl.ControlSystemSimulation(sys_w)

    # --- Antecedenty -------------------------------------
    cache_mf = {}
    def get_mf(var, lo2, hi1):
        key = (var, lo2, hi1)
        if key in cache_mf:
            return cache_mf[key]
        lo, hi = ranges[var]
        if not hasattr(_FS, var):
            univ = np.linspace(lo, hi, 1000)
            setattr(_FS, var, ctrl.Antecedent(univ, var))
        ante = getattr(_FS, var)
        name = f"{var}_{len(ante.terms)}"
        ante[name] = fuzz.trapmf(ante.universe, [lo2, lo2, hi1, hi1])
        cache_mf[key] = ante[name]
        return cache_mf[key]

    # --- Consequent --------------------------------------
    u = np.arange(1,6,1)
    Class = ctrl.Consequent(u, 'Class')
    for i,lbl in enumerate(['N','S','V','F','Q'],1):
        Class[lbl] = fuzz.trimf(u,[i,i,i])
    setattr(_FS,'Class',Class)

    # --- Pravidlá ----------------------------------------
    for ln in rule_lines:
        lhs, rhs = ln.split('=>',1)
        tgt = re.search(r'class=(\w)', rhs).group(1)
        cf  = float(re.search(r'CF\s*=\s*([0-9.]+)', rhs).group(1))
        atoms=[]
        for var,num in re.findall(r'\((\w+)\s+in\s+\[([^\]]+)\]\)', lhs):
            if var not in ranges: continue
            a0,lo2,hi1,d0 = num.split(',')
            atoms.append(get_mf(var,float(lo2),float(hi1)))
        if not atoms: continue
        ante = reduce(lambda a,b: a & b, atoms)
        r = ctrl.Rule(ante, _FS.Class[tgt]); r.weight=cf
        sys_w.addrule(r)

    _rule_objs = list(sys_w.rules)


def block_predict(block):
    """
    Spracuje blok X (2D ndarray), vráti list predikovaných tried.
    """
    preds=[]
    total = len(block)
    for row in block:
        if i % 50 == 0:  # aby to nevypisovalo každý úder, ale napr. každý 50-ty
            print(f"[PID {os.getpid()}] úder {i + 1}/{total}")
        for v in _columns:
            _FS.input[v] = float(row[_col_idx[v]])
        _FS.compute()
        best=-1.0; best_cls='Q'
        for r in _rule_objs:
            val=_strength(r)*r.weight
            if val>best:
                best=val; best_cls=r.consequent[0].term.label
        _FS.reset()
        preds.append(best_cls)
    return preds


warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------
# Multiprocessing launch
# ---------------------------------------------------------
if __name__ == "__main__":
    # Windows vyžaduje 'spawn'
    set_start_method("spawn", force=True)

    n_proc  = max(1, cpu_count() - 1)        # nech 1 jadro ostane voľné
    blocks  = np.array_split(X_te, n_proc*4) # ~4 bloky na proces

    print(f"▶  Spúšťam {n_proc} procesov …")

    with Pool(processes=n_proc,
              initializer=init_worker,
              initargs=(rule_lines, columns, col_idx)) as pool:
        y_blocks = list(tqdm(pool.imap(block_predict, blocks),
                             total=len(blocks),
                             desc="Klasifikujem (MP, verbose)", leave=False))
    y_pred = sum(y_blocks, [])   # zreťaziť bloky

    # ---------- 6. vyhodnotenie ---------------------------
    print(classification_report(y_te, y_pred, digits=3))
    print("macro-F1 =", f1_score(y_te, y_pred, average="macro"))