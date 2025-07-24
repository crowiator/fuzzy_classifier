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



## Výstupy projektu
	•	Matice chybovosti (PNG)
	•	Klasifikačné reporty (TXT)
	•	Vizualizácie fuzzy množín (PNG)
	•	Uložené pravidlá fuzzy systému (TXT)


