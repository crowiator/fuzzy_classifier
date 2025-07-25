"""
Tento skript slúži na optimalizáciu hyperparametrov klasických klasifikačných modelov (RandomForest, KNN, DecisionTree)
pomocou GridSearchCV na dátach z MIT-BIH Arrhythmia databázy. Využíva paralelizáciu na zrýchlenie procesu optimalizácie.

Postup skriptu zahŕňa:
- Načítanie a predspracovanie dát (imputácia a vyvažovanie dátovej množiny).
- Paralelné vykonanie Grid Search pre každý klasifikačný model na vyvážených dátach.
- Uloženie optimálnych parametrov, výsledkov cross-validácie a najlepších modelov.
- Vyhodnotenie optimalizovaných modelov na testovacej množine DS2 a uloženie klasifikačných správ.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
from custom_classifier import CustomClassifier
from src.dataset_loader import load_ds1_ds2, balance_datav2, CACHE


RESULTS = Path("../results")
RESULTS.mkdir(exist_ok=True, parents=True)

SCRIPT_DIR = Path(__file__).parent

#  pre Grid Search Definícia klasifikačných modelov a ich hyperparametrov
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
        'max_depth': [10, 20, 30],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 5]
    })
]


def find_best_params(model_name, model_type, param_grid, X_train, y_train):
    """
        Vykoná optimalizáciu hyperparametrov pomocou GridSearchCV pre daný model.

        Args:
            model_name (str): Názov modelu pre ukladanie výsledkov.
            model_type (str): Typ klasifikačného modelu (RandomForest, KNN, DecisionTree).
            param_grid (dict): Grid parametrov pre vyhľadávanie.
            X_train (np.ndarray): Trénovacie príznaky.
            y_train (np.ndarray): Trénovacie labely.

        Returns:
            GridSearchCV.best_estimator_: Najlepší nájdený klasifikačný model.
        """

    model = CustomClassifier(model_type=model_type)
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='f1_macro', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f" Optimalizované parametre pre model {model_name}: {grid_search.best_params_}")

    # Uloženie výsledkov
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(RESULTS / f"{model_name}_best_params.csv", index=False)

    # Uloženie najlepšieho modelu
    joblib.dump(grid_search.best_estimator_, RESULTS / f"{model_name}_best_model.joblib")

    return grid_search.best_estimator_


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
