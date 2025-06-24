# --- 1. collect beats + labels from DS1 -------------------------------
from pathlib import Path
from src.preprocessing.load      import load_record
from src.feature_extraction.time_domain import extract_beats, add_quality_flags
from src.feature_extraction.wavelet      import extract_wavelet_features
import pandas as pd
from src.feature_extraction.transformer import FeatureExtractor
print(clf.rules_)         # zoznam fuzzy pravidiel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
DS1 = ["100","101","102","103","109","111","112","114","115","116",
       "117","118","121","122","123","212","214","215","231","234"]

def create_features_csv():
    rows = []
    for rid in DS1:
        rec   = load_record(rid, lead="MLII", base_dir=Path("data/mit"))
        beats = extract_beats(rec.signal, fs=rec.fs, r_idx=rec.r_peaks,
                              zscore_amp=True, clip_extremes=True)
        beats = add_quality_flags(beats)

        # globálna wavelet energia (rovnaká pre všetky údery v zázname)
        wav   = extract_wavelet_features(rec.signal, rec.fs)
        for k, v in wav.items():
            beats[k] = v

        beats["record"]     = rid
        beats["label_aami"] = rec.labels_aami      # zarovnané na R_sample v load_record
        rows.append(beats)

    dataset = pd.concat(rows, ignore_index=True)
    dataset.to_csv("ds1_features.csv")

def main():
    dataset = create_features_csv()
    fe = FeatureExtractor(return_array=False, zscore_amp=True,
                          wavelet_levels=(2, 3, 4, 5, 6),
                          impute_wavelet_only=True)  # zachová info „vlna chýba”

    X_df = fe.fit_transform([(dataset, 360, dataset.R_sample.values)])  # quick hack
    y = dataset["label_aami"].values
    clf = FURIA(max_rules=15)  # obmedz max. počet pravidiel
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(clf, X_df.values, y, cv=cv, scoring="accuracy")
    print("CV accuracy 5×:", scores, "→ mean:", scores.mean())

    clf.fit(X_df.values, y)
    for r, rule in enumerate(clf.rules_, 1):
        print(f"Rule {r}:  IF", rule['antecedent'], "THEN", rule['class'])
