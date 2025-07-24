# Fuzzy klasifikátor EKG signálov

Tento projekt implementuje fuzzy klasifikačný systém (Mamdaniho inferenčný systém) a tradičné klasifikačné modely (Decision Tree, Random Forest, k-NN) pre klasifikáciu elektrokardiografických (EKG) signálov. Projekt využíva databázu MIT-BIH Arrhythmia Database a pracuje s morfologickými aj intervalovými príznakmi EKG úderov.

## Ciele projektu

- Klasifikácia EKG úderov do tried: normálne (N), supraventrikulárne (S), ventrikulárne (V), fúzne (F).
- Porovnanie výkonnosti fuzzy klasifikátora s tradičnými modelmi.
- Využitie interpretovateľnosti fuzzy pravidiel v medicínskych aplikáciách.

## Použité klasifikačné modely

- Fuzzy klasifikátor (Mamdani)
- Rozhodovací strom (Decision Tree)
- Náhodný les (Random Forest)
- Metóda k najbližších susedov (k-NN)

## Vstupné príznaky (features)

- Srdcový tep (`Heart_rate_bpm`)
- Amplitúda R vlny (`R_amplitude`)
- Amplitúda P vlny (`P_amplitude`)
- Amplitúda T vlny (`T_amplitude`)
- PR interval (`PR_ms`)
- RR0 interval (`RR0_s`)
- RR1 interval (`RR1_s`)

## Výsledky klasifikácie (Decision Tree)

Príklad výsledkov pre model Decision Tree:

| Metrika              | Výsledok |
|----------------------|----------|
| Accuracy             | 79.51 %  |
| Macro-F1 skóre       | 0.4012   |
| Matthewsov koeficient (MCC) | 0.3530   |
| ROC-AUC              | 0.7313   |

## Štruktúra projektu
```
fuzzy_classifier/
├── .pytest_cache/
├── cache/
├── data/
├── fuzzy_classifier.egg-info/
├── results/
├── src/
│   ├── cache/
│   │   ├── ds1_traditional.pkl
│   │   ├── ds2_traditional.pkl
│   │   └── fuzzy_mfs_orig.pkl
│   ├── classical/                  # Implementácia tradičných klasifikátorov
│   │   ├── cache/
│   │   ├── results/
│   │   ├── init.py
│   │   ├── classical_main.py
│   │   ├── confusion_matrix.png
│   │   ├── custom_classifier.py
│   │   ├── grid_search.py
│   │   └── traditional_model.py
│   ├── feature_extraction/         # Extrakcia príznakov EKG signálu
│   │   ├── time_domain.py
│   │   ├── transformer.py
│   │   └── wavelet.py
│   ├── fuzzy/                      # Implementácia fuzzy klasifikátora
│   │   ├── cache/
│   │   ├── plots/
│   │   ├── results/
│   │   │   ├── init.py
│   │   │   └── balanced_DS1_for_both.csv
│   │   ├── init.py
│   │   └── fuzzy_classifier.py
│   ├── preprocessing/              # Spracovanie a načítanie signálu
│   │   ├── init.py
│   │   ├── annotation_mapping.py
│   │   ├── ekg_vysek_porovnanie.png
│   │   ├── filtering.py
│   │   ├── load.py
│   │   └── show_filtering.py
│   ├── init.py
│   ├── config.py                   # Konfigurácia parametrov a ciest
│   └── dataset_loader.py
├── venv/
│   ├── bin/
│   ├── include/
│   ├── lib/
│   ├── py/
│   └── share/
├── pyvenv.cfg
├── pyproject.toml
├── requirements.txt
└── README.md
```



## Výstupy projektu
	•	Matice chybovosti (PNG)
	•	Klasifikačné reporty (TXT)
	•	Vizualizácie fuzzy množín (PNG)
	•	Uložené pravidlá fuzzy systému (TXT)


### Decision Tree
```
Classification Report – DecisionTree
========================================

              precision    recall  f1-score   support

           N       0.98      0.82      0.89     20301
           S       0.05      0.18      0.08       287
           V       0.31      0.62      0.41      1243
           F       0.14      0.61      0.22       385

    accuracy                           0.80     22216
   macro avg       0.37      0.56      0.40     22216
weighted avg       0.92      0.80      0.84     22216

Accuracy: 0.7951
Macro-F1 Score: 0.4012
MCC: 0.3530
ROC-AUC: 0.7313

```
### k-Nearest Neighbors (k-NN)

```
              precision    recall  f1-score   support

           N       0.98      0.75      0.85     20301
           S       0.04      0.24      0.07       287
           V       0.32      0.54      0.40      1243
           F       0.07      0.55      0.13       385

    accuracy                           0.73     22216
   macro avg       0.35      0.52      0.36     22216
weighted avg       0.91      0.73      0.80     22216

Accuracy: 0.7252  
Macro-F1 Score: 0.3619  
MCC: 0.2800  
ROC-AUC: 0.7021
```

### Random Forest

```
              precision    recall  f1-score   support

           N       0.99      0.90      0.95     20301
           S       0.23      0.20      0.21       287
           V       0.80      0.65      0.71      1243
           F       0.14      0.89      0.25       385

    accuracy                           0.88     22216
   macro avg       0.54      0.66      0.53     22216
weighted avg       0.95      0.88      0.91     22216

Accuracy: 0.8810  
Macro-F1 Score: 0.5301  
MCC: 0.5234  
ROC-AUC: 0.8035
```

### Fuzzy klasifikátor

```
              precision    recall  f1-score   support

           F      0.218     0.665     0.329       385
           N      0.986     0.946     0.966     20301
           S      0.145     0.279     0.191       287
           V      0.701     0.582     0.636      1243

    accuracy                          0.912     22216
   macro avg      0.513     0.618     0.530     22216
weighted avg      0.946     0.912     0.926     22216

Macro-F1: 0.5303  
MCC: 0.5616  
ROC-AUC: 0.7820

