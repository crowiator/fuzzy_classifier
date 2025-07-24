"""
Tento súbor predstavuje kompletný pipeline na trénovanie a vyhodnocovanie klasických modelov strojového učenia
(Random Forest, Decision Tree, K-Nearest Neighbors) určených na klasifikáciu elektrokardiografických (EKG) úderov.
Skript vykonáva nasledovné kroky:
- Načítanie predspracovaných trénovacích (DS1) a testovacích (DS2) dát zo súborov.
- Imputácia chýbajúcich hodnôt pomocou mediánu.
- Vyváženie trénovacieho datasetu pomocou kombinácie undersamplingu a BorderlineSMOTE oversamplingu.
- Trénovanie troch klasifikačných modelov s optimalizovanými hyperparametrami.
- Vyhodnotenie výkonu jednotlivých modelov pomocou metrík ako Accuracy, Macro-F1, MCC a ROC-AUC.
- Ukladanie podrobných výsledkov a vizualizácií (predikcie, klasifikačné správy, konfúzne matice) do priečinka "./results".
"""
from pathlib import Path
from sklearn.impute import SimpleImputer
from src.dataset_loader import load_ds1_ds2, balance_datav2, CACHE
from src.classical.traditional_model import (
    train_random_forest,
    train_knn,
    train_decision_tree,
    evaluate_model
)


RESULTS = Path("./results")
RESULTS.mkdir(exist_ok=True, parents=True)
SCRIPT_DIR = Path(__file__).parent


def main():
    """
        Hlavná funkcia skriptu, ktorá načíta dáta, vykoná ich vyváženie,
        natrénuje klasické modely (Decision Tree, Random Forest, KNN) a vyhodnotí ich na testovacích dátach.
        """

    # Načítanie datasetu pomocou existujúcej funkcie (rovnako ako fuzzy)
    X_tr, y_tr, _, X_te, y_te, _ = load_ds1_ds2()

    # Imputácia chýbajúcich hodnôt mediánom
    imputer = SimpleImputer(strategy='median')
    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)

    # Vybalansovanie tréningových dát
    df_bal = balance_datav2(X_tr, y_tr)

    # Oddelenie vybalansovaných features a labels
    X_bal = df_bal.drop(columns=["label"]).values
    y_bal = df_bal["label"].values

    classifiers = [
        ("DecisionTree", train_decision_tree,  {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2}),
        ("RandomForest", train_random_forest, {'criterion': 'gini', 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}),
        ("KNN", train_knn, {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}),

    ]

    RESULTS.mkdir(exist_ok=True, parents=True)

    # Trénovanie modelov na vybalansovaných dátach a vyhodnotenie na testovacích dátach
    for name, train_func, args in classifiers:
        print(f"\nTrénujem {name} na vybalansovaných dátach DS1 a testujem na DS2...")
        model = train_func(X_bal, y_bal, **args)
        evaluate_model(
            model, X_te, y_te,
            save_path=RESULTS / f"{name}_DS1_DS2_final",
            model_name=name
        )


if __name__ == "__main__":
    main()
