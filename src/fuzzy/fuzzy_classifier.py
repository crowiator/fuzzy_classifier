# fuzzy_classifier.py
"""
Tento skript implementuje kompletný pipeline na klasifikáciu EKG signálov pomocou fuzzy logiky.
Kód zahŕňa načítanie a prípravu dát, tvorbu fuzzy množín, generovanie pravidiel,
vyvažovanie datasetu, klasifikáciu, a vyhodnotenie modelu pomocou rôznych metrík.
"""
import sys
from sklearn.preprocessing import label_binarize
import os
import matplotlib
matplotlib.use('Agg')
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE
import skfuzzy as fuzz
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, roc_auc_score
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from pathlib import Path
from src.config import LEAD, DATA_DIR, FIS_FEATURES, DS2, DS1
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer import FeatureExtractor
from src.preprocessing.load import load_record
import seaborn as sns
from sklearn.metrics import confusion_matrix


sys.modules['tkinter'] = None
sys.modules['_tkinter'] = None
os.environ["MPLBACKEND"] = "Agg"
CACHE = Path("cache")
CACHE.mkdir(exist_ok=True)
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
LABELS = ["N", "S", "V", "F"]  # "Q"
IDX2LBL = {i + 1: lbl for i, lbl in enumerate(LABELS)}


def build_ds(rec_ids, tag):
    """
        Načíta a spracuje záznamy EKG podľa zoznamu identifikátorov (rec_ids),
        extrahuje príznaky pre fuzzy klasifikáciu a vygeneruje príslušné labely.

        Ak už existuje cacheovaný .pkl súbor pre daný tag, načíta sa z disku.

        Args:
            rec_ids (list[str]): Zoznam ID záznamov (napr. ['100', '101', ...])
            tag (str): Označenie sady (napr. 'ds1' alebo 'ds2') pre cacheovanie

        Returns:
            X (np.ndarray): Pole tvaru (n_samples, n_features) – príznaky
            y (np.ndarray): Pole tvaru (n_samples,) – triedy (napr. 'N', 'V', ...)
            features (list[str]): Zoznam použitých príznakov
        """
    # Skontroluj, či cacheovaný súbor existuje
    pkl_path = CACHE / f"{tag}_traditional.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            X, y, features = pickle.load(f)
        return X, y, features

    rows, y = [], []

    # Iterácia cez všetky záznamy (napr. pacientov)
    for rid in tqdm(rec_ids, desc=f"Loading {tag}"):
        # Načítaj záznam (signál, sampling rate, R-vlny)
        rec = load_record(rid, lead=LEAD, base_dir=DATA_DIR)
        rows.append((rec.signal, rec.fs, rec.r_peaks))
        # Vytvor mapovanie R-vĺn na AAMI labely
        ref = dict(zip(rec.r_peaks, rec.labels_aami))
        # Extrahuj jednotlivé údery zo signálu (okolo R-vĺn)
        beats = extract_beats(rec.signal, rec.fs, r_idx=rec.r_peaks, clip_extremes=True)
        # Získaj labely pre každý beat podľa R_sample, fallback = 'Q'
        y.extend([ref.get(r, "Q") for r in beats["R_sample"]])

    # Inicializuj extraktor čŕt (FeatureExtractor)
    fe = FeatureExtractor(return_array=False, impute_wavelet_only=True)
    fe.fit(rows)
    # Extrahuj črty a ponechaj len tie, ktoré sú relevantné pre fuzzy klasifikáciu
    X_df = fe.transform(rows)[FIS_FEATURES]
    X = X_df.to_numpy(float)

    # Ulož výsledky do cache (na budúce rýchle načítanie)
    with open(pkl_path, "wb") as f:
        pickle.dump((X, np.asarray(y), FIS_FEATURES), f)
    return X, np.asarray(y), FIS_FEATURES


def load_ds1_ds2():
    """
        Načíta a spracuje trénovaciu (DS1) aj testovaciu (DS2) množinu z MIT-BIH databázy.
        Volá funkciu build_ds pre obe množiny a zároveň vracia aj Pandas DataFrame verzie dát.

        Returns:
            X_tr (np.ndarray): Trénovacie príznaky
            y_tr (np.ndarray): Trénovacie triedy
            df_tr (pd.DataFrame): Trénovacie dáta ako DataFrame (na vizualizáciu, MF generáciu, atď.)
            X_te (np.ndarray): Testovacie príznaky
            y_te (np.ndarray): Testovacie triedy
            df_te (pd.DataFrame): Testovacie dáta ako DataFrame
        """
    # Načítanie trénovacej množiny (DS1)
    X_tr, y_tr, feat_names = build_ds(DS1, "ds1")
    df_tr = pd.DataFrame(X_tr, columns=feat_names)

    # Načítanie testovacej množiny (DS2)
    X_te, y_te, _ = build_ds(DS2, "ds2")
    df_te = pd.DataFrame(X_te, columns=feat_names)

    # Vráti oba datasety vo formáte ndarray aj ako DataFrame
    return X_tr, y_tr, df_tr, X_te, y_te, df_te


def create_fuzzy_mfs(feature, data, num_mfs):
    """
    Vytvorí fuzzy členitostné funkcie pre zvolenú črtu (feature) pomocou c-means clusteringu.
    Fuzzy množiny sú pomenované podľa domény črty (napr. 'veľmi nízka', 'nízka', ...).

    Args:
        feature (str): Názov črty (napr. 'Heart_rate_bpm')
        data (np.ndarray): Hodnoty črty
        num_mfs (int): Počet fuzzy množín (zvyčajne 5)

    Returns:
        universe (np.ndarray): Diskretizovaný interval hodnôt
        mfs (dict): Slovník fuzzy množín {meno: členitostná funkcia}
    """

    # Typy čŕt, ktoré budú mať pomenované fuzzy množiny typu "nízky-vysoký"
    VALUE_FEATURES = ["P_amplitude", "T_amplitude", "Heart_rate_bpm", "R_amplitude"]
    INTERVAL_FEATURES = ["RR0_s", "RR1_s"]

    # Odstráň NaN hodnoty
    data = data[~np.isnan(data)]

    # Automatický clustering (c-means) pre určenie stredov MF
    cntr, *_ = fuzz.cluster.cmeans(data.reshape(1, -1), c=num_mfs, m=2, error=0.005, maxiter=1000)
    cntr = np.sort(cntr.flatten())

    # Vytvor univerzum hodnôt (na výpočet a vykreslenie MF)
    universe = np.linspace(np.min(data), np.max(data), 256)

    # Pomenovania množín podľa typu črty
    if feature in VALUE_FEATURES:
        mf_names = ["veľmi nízka", "nízka", "stredná", "vysoká", "veľmi vysoká"]
    elif feature in INTERVAL_FEATURES:
        mf_names = ["veľmi krátky", "krátky", "stredný", "dlhý", "veľmi dlhý"]
    else:
        mf_names = [f"mf{i+1}" for i in range(num_mfs)]

    mfs = {}

    # Generovanie jednotlivých MF
    for i, center in enumerate(cntr):
        if i == 0:
            a, b, c = np.min(data), center, cntr[i + 1]
        elif i == len(cntr) - 1:
            a, b, c = cntr[i - 1], center, np.max(data)
        else:
            a, b, c = cntr[i - 1], center, cntr[i + 1]

        if feature in VALUE_FEATURES:
            sigma = (c - a) / 4
            mfs[mf_names[i]] = fuzz.gaussmf(universe, b, sigma)
        else:
            mfs[mf_names[i]] = fuzz.trimf(universe, [a, b, c])

    return universe, mfs


def generate_all_mfs(df, normalize=False, num_mfs=5):
    """
    Vygeneruje všetky fuzzy množiny (členitostné funkcie) pre každú vstupnú črtu v DataFrame.
    Výsledky sa ukladajú do cache súboru (.pkl), aby sa nemuseli opakovane počítať.

    Args:
        df (pd.DataFrame): DataFrame s extrahovanými príznakmi (napr. z DS1)
        normalize (bool): Ak je True, normalizuje črty do rozsahu [0, 1]
        num_mfs (int): Počet fuzzy množín na jednu črtu (typicky 5)

    Returns:
        fuzzy_mfs (dict): Slovník fuzzy množín pre každú črtu.
                          Každý prvok obsahuje 'universe' a 'mfs'.
    """

    # Zvoľ názov cache súboru podľa toho, či sa použije normalizácia
    fname = "fuzzy_mfs_norm.pkl" if normalize else "fuzzy_mfs_orig.pkl"
    path = CACHE / fname

    # Ak súbor už existuje, načítaj uložené fuzzy množiny
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    fuzzy_mfs = {}

    # Iteruj cez všetky črty, pre ktoré chceme generovať fuzzy množiny
    for feature in FIS_FEATURES:
        data = df[feature].values

        # Voliteľná normalizácia črty do rozsahu [0, 1]
        if normalize:
            data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)

        # Vytvor universe a fuzzy množiny pre danú črtu
        universe, mfs = create_fuzzy_mfs(feature, data, num_mfs=num_mfs)

        # Ulož do slovníka
        fuzzy_mfs[feature] = {"universe": universe, "mfs": mfs}

    # Ulož výsledné fuzzy množiny do cache súboru
    with open(path, "wb") as f:
        pickle.dump(fuzzy_mfs, f)

    return fuzzy_mfs


def generate_rule(row, fuzzy_mfs):
    """
    Vytvorí jedno fuzzy pravidlo pre daný úder (row) na základe najviac aktivovaných fuzzy množín.
    Pre každú črtu sa zvolí fuzzy množina (MF), ktorá má najvyšší stupeň príslušnosti.
    Pravidlo zahŕňa len tie MF, ktoré majú aktiváciu aspoň 0.5.

    Args:
        row (dict): Jedna vzorka (napr. EKG úder) vo forme slovníka s hodnotami čŕt a triedou ("label")
        fuzzy_mfs (dict): Slovník fuzzy množín pre každú črtu, obsahuje aj universe pre výpočty

    Returns:
        tuple: (pravidlo vo forme zoznamu podmienok [(črta, názov MF), ...], cieľová trieda)
    """

    # Zoznam podmienok pravidla (napr. [('RR0_s', 'stredný'), ('Heart_rate_bpm', 'vysoká')])
    rule_conditions = []

    # Pre každú črtu (feature) z definovaného zoznamu vstupných príznakov
    for feature in FIS_FEATURES:
        # Získaj univerzum a fuzzy členitostné funkcie (MFs) pre danú črtu
        universe = fuzzy_mfs[feature]['universe']
        mfs = fuzzy_mfs[feature]['mfs']

        # Vypočítaj stupeň príslušnosti (μ) hodnoty z danej vzorky ku každej fuzzy množine
        memberships = {
            mf_name: fuzz.interp_membership(universe, mf_curve, row[feature])
            for mf_name, mf_curve in mfs.items()
        }

        # Vyber MF s najvyššou mierou príslušnosti
        best_mf, best_membership = max(memberships.items(), key=lambda x: x[1])

        # Ak je aktivácia dostatočne vysoká (μ ≥ 0.5), pridaj do pravidla
        if best_membership >= 0.5:
            rule_conditions.append((feature, best_mf))

    # Výstupom je tuple: pravidlo (podmienky) a trieda vzorky (napr. 'N', 'V', 'S', 'F')
    return tuple(rule_conditions), row["label"]


def build_rules(df, fuzzy_mfs):
    """
    Generuje fuzzy pravidlá z datasetu. Každý riadok (úder EKG) je spracovaný na fuzzy pravidlo
    pomocou najviac aktivovaných fuzzy množín. Následne sa spočítajú frekvencie výskytov,
    vypočíta sa podpora a spoľahlivosť (support a confidence), a filtrujú sa len silné pravidlá.

    Pravidlá sa ukladajú do súborov pre ďalšie použitie.

    Args:
        df (pd.DataFrame): Vyvážený dataset s črtami a cieľovou triedou ("label")
        fuzzy_mfs (dict): Vygenerované fuzzy členitostné funkcie pre všetky črty

    Returns:
        rules (list): Zoznam pravidiel vo formáte (conds, label, confidence)
    """

    print("Prebieha generovanie fuzzy pravidiel...")

    # Konverzia dataframe na zoznam slovníkov (každý riadok ako dict)
    rows = df.to_dict(orient="records")

    # Paralelné generovanie pravidiel z jednotlivých úderov
    results = Parallel(n_jobs=mp.cpu_count(), backend="loky")(
        delayed(generate_rule)(row, fuzzy_mfs) for row in rows
    )

    # Spočítanie frekvencií rovnakých pravidiel (vrátane triedy)
    rule_counts = Counter(results)

    # Celkový počet výskytov pravidla bez ohľadu na triedu (pre confidence)
    rule_totals = Counter([r[0] for r in results])

    rules = []

    # Iteruj cez všetky kombinácie (podmienky, trieda)
    for (conds, label), count in rule_counts.items():
        support = count  # počet prípadov, ktoré zodpovedajú pravidlu
        total = rule_totals[conds]  # všetky prípady s rovnakými podmienkami
        confidence = support / total  # spoľahlivosť pravidla

        # Nastavenie minimálnych požiadaviek na kvalitu pravidla podľa triedy
        if label in ['V', 'S']:
            min_confidence = 0.8
            min_support = 6
        elif label == 'F':
            min_confidence = 0.9
            min_support = 12
        else:  # trieda 'N'
            min_confidence = 0.9
            min_support = 10

        # Ak pravidlo splní podmienky, pridaj ho
        if support >= min_support and confidence >= min_confidence:
            rules.append((conds, label, confidence))

    # Výpis pravidiel v textovej forme (na konzolu aj do súboru)
    rule_text = [
        " AND ".join(f"{f}:{m}" for f, m in conds) + f" => {label} (conf: {conf:.2f})"
        for conds, label, conf in rules
    ]

    # Uloženie pravidiel do súboru .txt (ľahko čitateľný formát)
    with open("fuzzy_rules.txt", "w", encoding="utf-8") as f:
        f.write("Fuzzy pravidlá (podmienky => trieda):\n\n")
        for line in rule_text:
            f.write(line + "\n")

    # Uloženie pravidiel do súboru .pkl (na neskoré načítanie modelu)
    with open("fuzzy_rules.pkl", "wb") as f:
        pickle.dump(rules, f)

    print(f"✅ Bolo extrahovaných {len(rules)} fuzzy pravidiel.")
    print(f"📁 Pravidlá uložené do: 'fuzzy_rules.txt' a 'fuzzy_rules.pkl'")

    return rules

def limit_rules_by_class(rules, max_per_class=300):
    """
    Obmedzí maximálny počet fuzzy pravidiel pre každú výstupnú triedu (N, S, V, F).
    Pravidlá sú najprv zoradené podľa spoľahlivosti (confidence) a následne sa vyberie
    len najspoľahlivejších N pravidiel na triedu.

    Args:
        rules (list): Zoznam fuzzy pravidiel vo formáte (podmienky, trieda, confidence)
        max_per_class (int): Predvolený limit pre triedy, ak nie je definovaný v slovníku

    Returns:
        pruned (list): Zredukovaný zoznam pravidiel s aplikovanými limitmi
    """

    # Vlastné maximálne počty pravidiel pre jednotlivé triedy
    max_per_class_dict = {
        "N": 300,   # Normálne údery – najviac pravidiel
        "S": 300,   # Supraventrikulárne údery
        "V": 200,   # Ventrikulárne údery
        "F": 100    # Fusion beats – zriedkavé
    }

    # Zoskupenie pravidiel podľa cieľovej triedy
    rules_by_class = defaultdict(list)

    # Zoradenie pravidiel podľa confidence zostupne
    for conds, cls, conf in sorted(rules, key=lambda x: -x[2]):
        # Pridaj pravidlo len ak ešte nedosiahneme limit pre triedu
        if len(rules_by_class[cls]) < max_per_class_dict.get(cls, max_per_class):
            rules_by_class[cls].append((conds, cls, conf))

    # Spojenie všetkých pravidiel do jedného zoznamu
    pruned = [rule for group in rules_by_class.values() for rule in group]

    print(f" Po aplikovaní limitu na počet pravidiel pre triedu zostáva {len(pruned)} pravidiel.")
    return pruned


from itertools import product


def mamdani_defuzzification(rule_scores, label_universe, mf_width=0.7):
    """
    Vykoná defuzzifikáciu pomocou Mamdaniho metódy s trojuholníkovými MF a metódou 'SOM'
    (Smallest of Maximum), ktorá je vhodná pre klasifikačné úlohy so zreteľným výberom triedy.

    Args:
        rule_scores (dict): Slovník s aktiváciami výstupných tried, napr. {'N': 0.7, 'S': 0.5}.
        label_universe (np.ndarray): Diskretizovaný interval hodnôt (napr. np.linspace(1, 4, 100)).
        mf_width (float): Šírka výstupných fuzzy množín. Typická hodnota je 0.7.

    Returns:
        str: Predikovaná výstupná trieda ako reťazec ('N', 'S', 'V', alebo 'F').
    """

    # Inicializácia výstupného fuzzy vektora – reprezentuje agregovaný výstup
    aggregated = np.zeros_like(label_universe)

    # Pre každú výstupnú triedu (N, S, V, F) s nenulovou aktiváciou
    for label, membership_value in rule_scores.items():
        # Urč stred výstupnej MF na číselnej osi (1 pre 'N', 2 pre 'S', atď.)
        center = LABELS.index(label) + 1

        # Vytvor trojuholníkovú fuzzy množinu pre túto triedu
        label_mf = fuzz.trimf(
            label_universe,
            [center - mf_width, center, center + mf_width]
        )

        # Vypočítaj výstup pravidla (AND medzi aktiváciou a MF), následne agreguj (MAX)
        np.maximum(aggregated, np.fmin(membership_value, label_mf), out=aggregated)

    # Defuzzifikuj výsledný fuzzy výstup pomocou metódy SOM (Smallest of Maximum)
    result = fuzz.defuzz(label_universe, aggregated, 'som')

    # Zaokrúhli výsledok na index a zabezpeč, aby bol v rozsahu tried
    label_idx = int(np.clip(np.round(result) - 1, 0, len(LABELS) - 1))

    # Vráť názov triedy podľa indexu (napr. 'N', 'S', ...)
    return LABELS[label_idx]

def fast_fuzzy_predict(row, fuzzy_mfs, rules, mf_width=0.7):
    """
    Vykoná inferenciu (predikciu) pre jednu vzorku pomocou fuzzy pravidiel a Mamdaniho defuzzifikácie.

    Args:
        row (list or np.ndarray): Jedna vzorka s hodnotami čŕt (napr. EKG úder)
        fuzzy_mfs (dict): Slovník fuzzy množín pre každú črtu
        rules (list): Zoznam fuzzy pravidiel vo formáte (conds, label, confidence)
        mf_width (float): Šírka výstupných výstupných MF pri defuzzifikácii

    Returns:
        predicted_label (str): Predikovaná trieda ('N', 'S', 'V', 'F')
        applied_rule_str (str): Reprezentácia najlepšie aktivovaného pravidla pre danú triedu
    """

    # Uchováva skóre pre každú výstupnú triedu (agregácia aktivovaných pravidiel)
    rule_scores = defaultdict(float)

    # Ukladá najlepšie aktivované pravidlo pre každú triedu
    best_rules = {}

    # Prechádzaj všetky pravidlá z pravidlovej bázy
    for conds, label, weight in rules:
        # Získaj mieru príslušnosti pre každú podmienku pravidla
        memberships = [
            fuzz.interp_membership(
                fuzzy_mfs[feature]['universe'],        # univerzum črty
                fuzzy_mfs[feature]['mfs'][mf],         # členitostná funkcia MF
                row[FIS_FEATURES.index(feature)]       # hodnota črty zo vzorky
            )
            for feature, mf in conds
        ]

        # Vypočítaj skóre pravidla: confidence * min(μ1, μ2, ..., μn)
        score = weight * min(memberships)

        # Ak skóre je vyššie ako doterajšie pre daný label, aktualizuj
        if score > rule_scores[label]:
            rule_scores[label] = score
            best_rules[label] = (conds, label, score)

    # Defuzzifikácia – výber výslednej triedy podľa agregovaných skóre
    label_universe = np.linspace(1, len(LABELS), 100)
    predicted_label = mamdani_defuzzification(rule_scores, label_universe, mf_width=mf_width)

    # Získaj najlepšie pravidlo, ktoré prispelo k predikcii
    main_rule = best_rules.get(predicted_label, None)

    # Ak existuje, vytvor jeho textovú reprezentáciu
    if main_rule:
        conds, label, score = main_rule
        applied_rule_str = " AND ".join([f"{f}:{m}" for f, m in conds]) + f" => {label} (skóre: {score:.3f})"
    else:
        applied_rule_str = "N/A"

    return predicted_label, applied_rule_str



def predict_fast_batch(X_test, fuzzy_mfs, rules):
    """
    Paralelne vykoná fuzzy inferenciu pre všetky vzorky v testovacom datasete
    pomocou optimalizovanej funkcie fast_fuzzy_predict.

    Args:
        X_test (np.ndarray): Testovacia množina (každý riadok = jedna vzorka)
        fuzzy_mfs (dict): Slovník fuzzy množín pre všetky črty
        rules (list): Zoznam fuzzy pravidiel vo formáte (podmienky, label, confidence)

    Returns:
        predictions (tuple): N-tica predikovaných tried (napr. 'N', 'V', ...)
        applied_rules (tuple): N-tica textových reprezentácií najlepších aplikovaných pravidiel
    """

    # ⚡ Paralelná inferencia pre každú vzorku pomocou joblib.Parallel
    results = Parallel(n_jobs=mp.cpu_count(), backend='threading')(
        delayed(fast_fuzzy_predict)(row, fuzzy_mfs, rules)
        for row in tqdm(X_test, desc="Fast fuzzy inference")
    )

    # Rozdelenie výsledkov: [(label1, rule1), (label2, rule2), ...] → dve oddelené n-tice
    predictions, applied_rules = zip(*results)

    return predictions, applied_rules




def fuzzy_pipeline(X_train, y_train, X_test, y_test):
    """
    Kompletný fuzzy klasifikačný pipeline:
    - Vyváži trénovacie dáta
    - Vygeneruje fuzzy množiny a pravidlá
    - Vykoná inferenciu na testovacej množine
    - Vyhodnotí model metrikami a uloží výsledky

    Args:
        X_train (np.ndarray): Trénovacie črty (napr. z DS1)
        y_train (np.ndarray): Triedy pre trénovacie dáta
        X_test (np.ndarray): Testovacie črty (napr. z DS2)
        y_test (np.ndarray): Triedy pre testovacie dáta
    """

    # 1. Vyváženie datasetu pomocou under- a oversamplingu
    df_bal = balance_datav2(X_train, y_train)
    df_bal.to_csv("balanced_DS1_for_both.csv", index=False)
    print("Vyvážená dátová množina bola uložená do súboru 'balanced_DS1_for_both.csv'")

    # 2. Generovanie fuzzy množín (členitostných funkcií) pre všetky črty
    fuzzy_mfs = generate_all_mfs(df_bal, normalize=False)
    plot_and_save_fuzzy_mfs(fuzzy_mfs, FIS_FEATURES)

    # 3. Vizualizácia aktivácie fuzzy množín pre jeden konkrétny úder
    beat_features = X_test[0]
    plot_single_beat_memberships(beat_features, fuzzy_mfs, "Heart_rate_bpm")

    # 4. Generovanie fuzzy pravidiel + orezanie na max. počet pravidiel na triedu
    rules = build_rules(df_bal, fuzzy_mfs)
    rules = limit_rules_by_class(rules, max_per_class=300)

    # 5. Odstráň pravidlá pre triedy, ktoré nie sú v LABELS
    rules = [r for r in rules if r[1] in LABELS]

    # 6. Vizualizácia výstupných fuzzy množín pre výstupnú premennú
    plot_output_mfs(LABELS, mf_type='trimf', width=0.7)

    # 7. Výpis počtu pravidiel pre každú triedu
    rule_dist = defaultdict(int)
    for _, label, _ in rules:
        rule_dist[label] += 1
    print("Rozdelenie pravidiel podľa tried:", dict(rule_dist))

    # 8. Paralelná fuzzy inferencia nad testovacou množinou
    y_pred, applied_rules = predict_fast_batch(X_test, fuzzy_mfs, rules)

    # 9. Vytvor dataframe s detailnými predikciami a pravidlami
    detailed_results = pd.DataFrame(X_test, columns=FIS_FEATURES)
    detailed_results["True_Label"] = y_test
    detailed_results["Predicted_Label"] = y_pred
    detailed_results["Rule"] = applied_rules
    detailed_results.to_csv("detailed_predictions.csv", index=False)

    # 10. Výpis klasifikačnej správy a výpočty metrík
    print("\nVýstup klasifikačnej správy:")
    report = classification_report(y_test, y_pred, digits=3, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    # Výpočet ROC-AUC (pre multiklasy)
    y_test_bin = label_binarize(y_test, classes=LABELS)
    y_pred_bin = label_binarize(y_pred, classes=LABELS)
    roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average="macro", multi_class="ovr")

    #  Výpis do konzoly
    print(report)
    print("Macro-F1:", macro_f1)
    print("MCC:", mcc)
    print("ROC-AUC:", roc_auc)

    # 11. Uloženie metrík do textového súboru
    with open("fuzzy_report_parallel.txt", "w") as f:
        f.write(report)
        f.write(f"\nMacro-F1: {macro_f1:.4f}")
        f.write(f"\nMCC: {mcc:.4f}")
        f.write(f"\nROC-AUC: {roc_auc:.4f}")

    # 12. Konfúzna matica
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel('Predikovaná trieda')
    plt.ylabel('Skutočná trieda')
    plt.title('Konfúzna matica')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    print("Confusion matrix uložená ako 'confusion_matrix.png'")





def balance_datav2(X, y):
    """
    Vyváži dátovú množinu v dvoch krokoch:
    1. Odstráni triedu 'Q', ktorá nie je zahrnutá vo fuzzy klasifikácii.
    2. Použije RandomUnderSampler na zníženie počtu dominantnej triedy 'N'.
    3. Použije BorderlineSMOTE na syntetické doplnenie menšinových tried.

    Args:
        X (np.ndarray): Matica vstupných čŕt (napr. EKG príznaky)
        y (np.ndarray): Zodpovedajúce triedy (napr. 'N', 'V', 'S', ...)

    Returns:
        df_bal (pd.DataFrame): Vyvážená dátová množina vo forme DataFrame (vrátane stĺpca 'label')
    """

    print("Vyvažovanie dátovej množiny...")

    # 1. Imputácia chýbajúcich hodnôt (medianom)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # 2. Odstránenie vzoriek s triedou 'Q', ktoré sa nezapočítavajú do klasifikácie
    mask = y != "Q"
    X_filtered, y_filtered = X_imp[mask], y[mask]

    # 3. Undersampling: obmedz počet vzoriek triedy 'N' na max. 4000
    rus = RandomUnderSampler(sampling_strategy={'N': 4000}, random_state=42)
    X_under, y_under = rus.fit_resample(X_filtered, y_filtered)

    # 4. Oversampling: syntetické generovanie vzoriek pre ostatné triedy pomocou BorderlineSMOTE
    smote = BorderlineSMOTE(sampling_strategy='not majority', random_state=42)
    X_bal, y_bal = smote.fit_resample(X_under, y_under)

    # 5. Vytvorenie DataFrame a pripojenie cieľového stĺpca 'label'
    df_bal = pd.DataFrame(X_bal, columns=FIS_FEATURES)
    df_bal["label"] = y_bal

    # Výpis výsledného rozdelenia tried po vyvážení
    print("Rozdelenie tried po vyvážení:", Counter(df_bal["label"]))
    print("Po balansovaní:")
    return df_bal

def plot_and_save_fuzzy_mfs(fuzzy_mfs, features):
    """
    Pre každú črtu v zozname vygeneruje graf fuzzy členitostných funkcií (MF)
    a uloží ho ako PNG súbor do zvoleného výstupného adresára.

    Args:
        fuzzy_mfs (dict): Slovník fuzzy množín pre každú črtu.
                          Obsahuje 'universe' a 'mfs' (názov MF → funkcia).
        features (list): Zoznam názvov čŕt, pre ktoré sa majú MF vizualizovať.
    """

    for feature in features:
        # Získanie diskretizovaného rozsahu hodnôt (x-ová os)
        universe = fuzzy_mfs[feature]['universe']

        # Získanie fuzzy množín pre danú črtu
        mfs = fuzzy_mfs[feature]['mfs']

        # Vytvorenie grafu
        plt.figure(figsize=(8, 4))

        # Pre každú fuzzy množinu vykresli príslušnostnú funkciu
        for mf_name, mf_curve in mfs.items():
            plt.plot(universe, mf_curve, label=mf_name)

        # Popisy a štýl grafu
        plt.title(f'Fuzzy Membership Functions - {feature}', fontsize=14)
        plt.xlabel(f'Hodnoty {feature}')
        plt.ylabel('Príslušnosť (μ)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()

        # Vytvor výstupnú cestu a ulož graf
        plot_path = PLOTS_DIR / f"{feature}_mfs.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"MF pre '{feature}' uložené do: {plot_path}")


def plot_output_mfs(labels=['N', 'S', 'F', 'V'], mf_type='trimf', width=0.5):
    """
    Vytvorí graf fuzzy množín (členitostných funkcií) pre výstupnú premennú "Class".
    Triedy sú umiestnené na numerickej osi 1, 2, 3, 4 a každá má svoju fuzzy množinu.

    Args:
        labels (list): Zoznam tried, pre ktoré sa budú tvoriť fuzzy množiny.
                       Každá trieda bude reprezentovaná na svojej číselnej pozícii.
        mf_type (str): Typ členitostnej funkcie ('trimf', 'trapmf', 'gaussmf').
        width (float): Parametrická šírka fuzzy množiny (ovplyvňuje prekrytie).
    """

    # Vytvor univerzum hodnôt pre výstupnú premennú (napr. od 1.0 po 4.0)
    label_universe = np.linspace(1, len(labels), 500)

    # Inicializuj graf
    plt.figure(figsize=(8, 4))

    # Pre každú triedu vytvor príslušnú fuzzy množinu
    for idx, label in enumerate(labels):
        center = idx + 1  # napr. 'N' → 1, 'S' → 2, atď.

        # Vytvor členitostnú funkciu podľa požadovaného typu
        if mf_type == 'trimf':
            mf_curve = fuzz.trimf(label_universe, [center - width, center, center + width])
        elif mf_type == 'trapmf':
            half_width = width / 2
            mf_curve = fuzz.trapmf(label_universe,
                                   [center - width, center - half_width, center + half_width, center + width])
        elif mf_type == 'gaussmf':
            mf_curve = fuzz.gaussmf(label_universe, center, width)
        else:
            raise ValueError("Nepodporovaný typ MF.")

        # Vykresli fuzzy množinu do grafu
        plt.plot(label_universe, mf_curve, label=label)

    # Nastavenie popisov grafu
    plt.title('Fuzzy výstupné množiny - Výstupná premenná "Class"', fontsize=14)
    plt.xlabel('Výstupná premenná Class (numericky)')
    plt.ylabel('Príslušnosť (μ)')
    plt.xticks(range(1, len(labels) + 1), range(1, len(labels) + 1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Uloženie grafu do súboru
    plot_path = PLOTS_DIR / f"output_variable_mfs.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Výstupné MF uložené do: {plot_path}")


def plot_single_beat_memberships(beat_features, fuzzy_mfs, feature_name="Heart_rate_bpm"):
    """
    Vizualizuje fuzzy členitostné funkcie pre jednu črtu a označí,
    ako silno je hodnota konkrétneho úderu (beat) priradená ku každej množine.

    Args:
        beat_features (list or np.ndarray): Vektor čŕt pre jeden úder
        fuzzy_mfs (dict): Slovník fuzzy množín s 'universe' a 'mfs' pre každú črtu
        feature_name (str): Názov črty, ktorú chceme analyzovať (napr. 'Heart_rate_bpm')
    """

    # Získaj hodnotu danej črty z beat_features
    beat_value = beat_features[FIS_FEATURES.index(feature_name)]

    # Získaj univerzum a členitostné funkcie pre danú črtu
    universe = fuzzy_mfs[feature_name]['universe']
    mfs = fuzzy_mfs[feature_name]['mfs']

    # Priprav graf
    plt.figure(figsize=(10, 6))
    memberships = {}

    # Definované farby pre jednotlivé fuzzy množiny
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Pre každú fuzzy množinu vykresli krivku a vypočítaj aktiváciu pre konkrétny beat_value
    for (mf_name, mf_curve), color in zip(mfs.items(), colors):
        plt.plot(universe, mf_curve, label=mf_name, color=color)

        # Výpočet stupňa príslušnosti
        membership = fuzz.interp_membership(universe, mf_curve, beat_value)
        memberships[mf_name] = membership

        # Vizualizácia bodu aktivácie na krivke MF
        plt.plot(beat_value, membership, 'o', markersize=8, color=color)

    # Zvislá čiara znázorňujúca hodnotu beat_value na osi x
    plt.axvline(x=beat_value, color='k', linestyle='--', label=f'Hodnota beat: {beat_value:.2f}')

    # Popis grafu
    plt.title(f'Aktivácia fuzzy množín pre {feature_name}')
    plt.xlabel(f'Hodnoty črty: {feature_name}')
    plt.ylabel('Stupeň príslušnosti (μ)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Uloženie grafu do súboru
    plot_path = PLOTS_DIR / f"single_beat_{feature_name}_activation.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Výpis informácií o príslušnosti do konzoly
    print(f"Graf aktivácie uložený ako '{plot_path}'")
    print(f"\nStupne príslušnosti hodnoty {beat_value:.2f} k fuzzy množinám pre '{feature_name}':")
    for mf_name, membership in memberships.items():
        print(f" - {mf_name}: {membership:.3f}")

def main():
    """
    Hlavná vstupná funkcia skriptu. Spustí načítanie dát, predspracovanie
    a následne celý fuzzy klasifikačný pipeline.
    """

    # Potlačenie varovaní typu RuntimeWarning (napr. pri výpočtoch s NaN)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1. Načítanie trénovacej a testovacej množiny z MIT-BIH databázy
    X_tr, y_tr, _, X_te, y_te, _ = load_ds1_ds2()

    # 2. Imputácia chýbajúcich hodnôt (napr. NaN) pomocou mediánu
    imputer = SimpleImputer(strategy='median')
    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)

    # 3. Spustenie kompletného fuzzy pipeline (generovanie MF, pravidiel, inferencia, vyhodnotenie)
    fuzzy_pipeline(X_tr, y_tr, X_te, y_te)

# Spustenie skriptu, ak je spúšťaný priamo
if __name__ == "__main__":
    main()

