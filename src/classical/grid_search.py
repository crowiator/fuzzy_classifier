"""
Grid Search a vyhodnotenie tradičných klasifikátorov na EKG dátach

Tento skript vykonáva tréning a optimalizáciu hyperparametrov tradičných klasifikačných algoritmov
(Random Forest, SVM, KNN, Decision Tree) pomocou GridSearchCV nad extrahovanými príznakmi EKG signálu
z MIT-BIH databázy. Používa dva datasety (DS1 – tréning, DS2 – test), vykonáva balansovanie tried
a vyhodnocuje výkonnosť modelov pomocou klasifikačných metrík. Výsledky a modely sú uložené do súborov.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE
from tqdm import tqdm
import pickle
from pathlib import Path
from src.feature_extraction.time_domain import extract_beats
from src.feature_extraction.transformer import FeatureExtractor
from src.preprocessing.load import load_record
import matplotlib
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

matplotlib.use('Agg')  # Neinteraktívny backend
from custom_classifier import CustomClassifier
from src.config import LEAD, DATA_DIR, FIS_FEATURES, DS1, DS2

CACHE = Path("../fuzzy/cache")
CACHE.mkdir(exist_ok=True)
RESULTS = Path("../results")
RESULTS.mkdir(exist_ok=True, parents=True)
SCRIPT_DIR = Path(__file__).parent

classifiers_with_params = [
    ("RandomForest", "RandomForest", {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 5]
    }),
    ("KNN", "KNN", {
        'n_neighbors': [3, 5, 7],
        'metric': ['euclidean', 'manhattan'],
        'weights': ['uniform', 'distance']
    }),
    ("DecisionTree", "DecisionTree", {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 5]
    })
]


def find_best_params(model_name, model_type, param_grid, X_train, y_train):
    model = CustomClassifier(model_type=model_type)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"✅ Najlepšie parametre pre {model_name}: {grid_search.best_params_}")

    # Uloženie výsledkov
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(RESULTS / f"{model_name}_best_params.csv", index=False)

    # Uloženie najlepšieho modelu
    joblib.dump(grid_search.best_estimator_, RESULTS / f"{model_name}_best_model.joblib")

    return grid_search.best_estimator_




def build_ds(rec_ids, tag):
    pkl_path = CACHE / f"{tag}_traditional.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            X, y, features = pickle.load(f)
        return X, y, features

    rows, y = [], []
    for rid in tqdm(rec_ids, desc=f"Načitávanie {tag}"):
        rec = load_record(rid, lead=LEAD, base_dir=DATA_DIR)
        rows.append((rec.signal, rec.fs, rec.r_peaks))
        ref = dict(zip(rec.r_peaks, rec.labels_aami))
        beats = extract_beats(rec.signal, rec.fs, r_idx=rec.r_peaks, clip_extremes=True)
        y.extend([ref.get(r, "Q") for r in beats["R_sample"]])

    fe = FeatureExtractor(return_array=False, impute_wavelet_only=True)
    fe.fit(rows)
    X_df = fe.transform(rows)[FIS_FEATURES]
    X = X_df.to_numpy(float)

    with open(pkl_path, "wb") as f:
        pickle.dump((X, np.asarray(y), FIS_FEATURES), f)

    return X, np.asarray(y), FIS_FEATURES


def load_ds1_ds2():
    X_tr, y_tr, feat_names = build_ds(DS1, "ds1")
    df_tr = pd.DataFrame(X_tr, columns=feat_names)
    X_te, y_te, _ = build_ds(DS2, "ds2")
    df_te = pd.DataFrame(X_te, columns=feat_names)
    return X_tr, y_tr, df_tr, X_te, y_te, df_te


def balance_datav2(X, y):
    print("⚖️Vyvažovanie dátovej množiny...")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # Vyfiltruj všetky vzorky triedy "Q"
    mask = y != "Q"
    X_filtered, y_filtered = X_imp[mask], y[mask]

    rus = RandomUnderSampler(sampling_strategy={'N': 4000}, random_state=42)
    X_under, y_under = rus.fit_resample(X_filtered, y_filtered)

    smote = BorderlineSMOTE(sampling_strategy='not majority', random_state=42)
    X_bal, y_bal = smote.fit_resample(X_under, y_under)

    df_bal = pd.DataFrame(X_bal, columns=FIS_FEATURES)
    df_bal["label"] = y_bal
    print("✅ Rozdelenie tried po vyvážení:", Counter(df_bal["label"]))
    print("Po balansovaní:")
    return df_bal
def main():
    X_tr, y_tr, _, X_te, y_te, _ = load_ds1_ds2()
    imputer = SimpleImputer(strategy='median')
    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)

    df_bal = balance_datav2(X_tr, y_tr)
    X_bal = df_bal.drop(columns=["label"]).values
    y_bal = df_bal["label"].values

    RESULTS.mkdir(exist_ok=True, parents=True)

    Parallel(n_jobs=-1)(
        delayed(find_best_params)(name, model_type, params, X_bal, y_bal)
        for name, model_type, params in classifiers_with_params
    )

    # Vyhodnotenie na DS2
    for name, _, _ in classifiers_with_params:
        best_model = joblib.load(RESULTS / f"{name}_best_model.joblib")
        y_pred = best_model.predict(X_te)
        report = classification_report(y_te, y_pred, digits=3, zero_division=0)
        print(f"\nKlasifikačná správa pre {name}:\n{report}")

        with open(RESULTS / f"{name}_final_report.txt", 'w') as f:
            f.write(report)


if __name__ == "__main__":
    main()
