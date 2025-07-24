"""
Súbor traditional_models.py obsahuje funkcie,
ktoré umožňujú trénovanie a vyhodnotenie klasických metód strojového učenia
(K-Nearest Neighbors, Decision Tree, Random Forest)
na klasifikačné úlohy.
Každá funkcia pripraví a vráti natrénovaný klasifikačný model,
pričom sú parametre trénovania flexibilné a možno ich meniť podľa potreby.
Súčasťou súboru je aj funkcia na vyhodnotenie výsledkov klasifikácie.
"""

import os
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from src.config import FIS_FEATURES
import seaborn as sns

def train_knn(X_train, y_train, metric, n_neighbors, weights):
    # Inicializácia modelu KNN s definovanými parametrami (metrika vzdialenosti, počet susedov, váhovanie)
    # model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model = KNeighborsClassifier(metric=metric, n_neighbors=n_neighbors, weights=weights)
    # Trénovanie modelu na trénovacích dátach
    model.fit(X_train, y_train)
    # Vrátenie natrénovaného modelu
    return model


# Funkcia na trénovanie klasifikátora typu Decision Tree (rozhodovací strom)
def train_decision_tree(X_train, y_train, criterion, max_depth, max_features, min_samples_leaf, min_samples_split):
    # Inicializácia modelu Decision Tree s parametrami (kritérium delenia, maximálna hĺbka stromu, atď.)
    # model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features,
                                   min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                   random_state=42)
    # Trénovanie modelu na trénovacích dátach
    model.fit(X_train, y_train)
    # Vrátenie natrénovaného modelu
    return model


# Funkcia na trénovanie klasifikátora typu Random Forest
def train_random_forest(X_train, y_train, criterion, max_depth, max_features, min_samples_leaf, min_samples_split,
                        n_estimators):
    # Inicializácia modelu Random Forest s definovanými parametrami
    # (počet stromov, maximálna hĺbka, kritérium delenia, atď.)
    # model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model = RandomForestClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features,
                                   min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                   n_estimators=n_estimators, random_state=42)
    # Trénovanie modelu na trénovacích dátach
    model.fit(X_train, y_train)

    return model


# Funkcia na vyhodnotenie natrénovaného modelu na testovacích dátach
def evaluate_model(model, X_test, y_test, save_path=None, model_name="model", cv_scores=None):
    class_names = ["N", "S", "V", "F"]

    # Predikcie
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    print("Matica chybovosti:")
    print(cm)

    # Classification Report
    report = classification_report(y_test, y_pred, labels=class_names, zero_division=0)
    print("\nSpráva o klasifikácii:")
    print(report)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # ROC-AUC (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=class_names)
    y_pred_bin = label_binarize(y_pred, classes=class_names)

    try:
        roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro')
    except ValueError:
        roc_auc = float('nan')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1 Score: {macro_f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    if save_path:
        base_path = os.path.splitext(save_path)[0]

        # Uloženie predikcií
        df_results = pd.DataFrame(X_test, columns=[f"{FIS_FEATURES[i]}" for i in range(X_test.shape[1])])
        df_results["True_Label"] = y_test
        df_results["Predicted_Label"] = y_pred
        csv_path = f"{base_path}_{model_name}.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"Predikcie uložené do: {csv_path}")


        # Uloženie confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predikovaná trieda')
        plt.ylabel('Skutočná trieda')
        plt.title('Matica chybovosti')
        plt.tight_layout()
        cm_path = f"{base_path}_{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Matica chybovosti uložená do: {cm_path}")


        # Uloženie classification reportu
        report_path = f"{base_path}_{model_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Classification Report – {model_name}\n")
            f.write("=" * 40 + "\n\n")
            f.write(report)
            f.write(f"\nAccuracy: {accuracy:.4f}\n")
            f.write(f"Macro-F1 Score: {macro_f1:.4f}\n")
            f.write(f"MCC: {mcc:.4f}\n")
            f.write(f"ROC-AUC: {roc_auc:.4f}\n")

        print(f"Správa o klasifikácii uložená do: {report_path}")

