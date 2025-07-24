"""
Tento skript obsahuje nástroje na načítanie, predspracovanie a vyváženie dát z MIT-BIH Arrhythmia databázy.

Obsahuje nasledujúce funkcie:
- build_ds: Načítanie a predspracovanie dátových záznamov pre fuzzy klasifikáciu.
- load_ds1_ds2: Načítanie trénovacích (DS1) a testovacích (DS2) dátových množín.
- balance_datav2: Vyváženie dátovej množiny pomocou kombinácie undersamplingu a oversamplingu.

Funkcie sú optimalizované na opakované použitie s cachovaním výsledkov.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from pathlib import Path
from src.config import LEAD, DATA_DIR, FIS_FEATURES, DS1, DS2
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer import FeatureExtractor
from src.preprocessing.load import load_record
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]  # o jednu úroveň vyššie ako predtým
CACHE = ROOT / "src" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)


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
    fe.fit(rows)  # nauč sa z dát (napr. pre normalizáciu alebo výplň)

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