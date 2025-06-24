import numpy as np
import neurokit2 as nk
import pandas as pd
from src.preprocessing.load import load_record
from src.preprocessing.filtering import clean_ecg_v2
from src.config import DATA_DIR, LEAD


def find_nearest_onsets_offsets(r_peaks, onsets, offsets):
    r_on_corrected, r_off_corrected = [], []

    for r in r_peaks:
        onset_candidates = onsets[onsets < r]
        offset_candidates = offsets[offsets > r]

        r_on = onset_candidates[-1] if len(onset_candidates) > 0 else np.nan
        r_off = offset_candidates[0] if len(offset_candidates) > 0 else np.nan

        r_on_corrected.append(r_on)
        r_off_corrected.append(r_off)

    return np.array(r_on_corrected), np.array(r_off_corrected)


def extract_beats_features(signal, fs, r_idx):
    clean_signal = clean_ecg_v2(signal, fs, add_dwt=True)

    waves, _ = nk.ecg_delineate(clean_signal, r_idx, sampling_rate=fs, method="dwt")

    r_onsets_idx = np.where(waves["ECG_R_Onsets"].to_numpy() == 1)[0]
    r_offsets_idx = np.where(waves["ECG_R_Offsets"].to_numpy() == 1)[0]
    p_peaks_idx = np.where(waves["ECG_P_Peaks"].to_numpy() == 1)[0]
    t_peaks_idx = np.where(waves["ECG_T_Peaks"].to_numpy() == 1)[0]

    r_on_corrected, r_off_corrected = find_nearest_onsets_offsets(r_idx, r_onsets_idx, r_offsets_idx)

    beats = []

    for i, r_peak in enumerate(r_idx):
        r_amp = clean_signal[r_peak]
        rr_interval = (r_peak - r_idx[i - 1]) / fs if i > 0 else np.nan
        hr_bpm = 60 / rr_interval if not np.isnan(rr_interval) else np.nan

        # nájdenie najbližších P a T vrcholov
        p_candidates = p_peaks_idx[p_peaks_idx < r_peak]
        t_candidates = t_peaks_idx[t_peaks_idx > r_peak]

        p_amp = clean_signal[p_candidates[-1]] if len(p_candidates) > 0 else np.nan
        t_amp = clean_signal[t_candidates[0]] if len(t_candidates) > 0 else np.nan

        qrs_duration_ms = ((r_off_corrected[i] - r_on_corrected[i]) / fs * 1000
                           if np.isfinite(r_on_corrected[i]) and np.isfinite(r_off_corrected[i])
                           else np.nan)

        beats.append({
            "beat_idx": i,
            "R_sample": r_peak,
            "R_amplitude": r_amp,
            "RR_interval_s": rr_interval,
            "Heart_rate_bpm": hr_bpm,
            "P_amplitude": p_amp,
            "T_amplitude": t_amp,
            "QRS_duration_ms": qrs_duration_ms,
        })

    return pd.DataFrame(beats)


# Načítaj signál
rec = load_record("100", lead=LEAD, base_dir=DATA_DIR)

# Extrahuj beatové príznaky
beats_df = extract_beats_features(rec.signal, rec.fs, rec.r_peaks)

print(beats_df.head(10))
