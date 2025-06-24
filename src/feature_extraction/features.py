from pathlib import Path

import pandas as pd

from src.config import LEAD, DATA_DIR
from src.feature_extraction.time_domain import extract_beats, add_quality_flags
from src.preprocessing.load import load_record

records = ["100", "101", "102"]

all_dfs = []
for rid in records:
    rec = load_record(rid, lead=LEAD, base_dir=DATA_DIR)
    beats = extract_beats(rec.signal, fs=rec.fs, r_idx=rec.r_peaks, zscore_amp=True)
    beats = add_quality_flags(beats)

    labels_df = pd.DataFrame({
        "record": rid,
        "R_sample": rec.r_peaks.astype(int),
        "label_aami": rec.labels_aami,
    })
    all_dfs.append(beats.merge(labels_df, on="R_sample", how="inner"))

dataset = pd.concat(all_dfs, ignore_index=True)

# Ukáž prvých 10 riadkov
print(dataset.head(10))