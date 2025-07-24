"""
Vizualizácia EKG signálu pred a po prefiltrovaní

Tento skript slúži na načítanie EKG záznamu z MIT-BIH databázy, jeho prefiltrovanie pomocou funkcie `clean_ecg_v2`
a následné porovnanie úseku nefiltrovaného a filtrovaného signálu. Výstupom je obrázok uložený vo formáte PNG,
ktorý zobrazuje porovnanie signálu na zvolenom intervale (napr. prvých 10 sekúnd).
"""
import matplotlib.pyplot as plt
from src.preprocessing.filtering import clean_ecg_v2
from src.preprocessing.load import load_record
import numpy as np
from src.config import LEAD, DATA_DIR
import matplotlib
matplotlib.use('Agg')


# 1. Načítanie EKG záznamu
rid = "100"  # alebo iný MIT-BIH záznam
rec = load_record(rid, lead=LEAD, base_dir=DATA_DIR)
raw_sig = rec.signal
fs = rec.fs

# 2. Filtrovanie signálu
clean_pref = clean_ecg_v2(raw_sig, fs, add_dwt=True)

# 3. Vyber úseku na zobrazenie (napr. prvých 10 sekúnd)
start_sec = 0
end_sec = 10
start_idx = start_sec * fs
end_idx = end_sec * fs

t = np.arange(start_idx, end_idx) / fs  # časová os v sekundách

plt.figure(figsize=(12, 6))

# Nefiltrovaný signál (úsek)
plt.subplot(2, 1, 1)
plt.plot(t, raw_sig[start_idx:end_idx], linewidth=0.8)
plt.title(f"Nefiltrovaný EKG signál ({rec.lead}) – {start_sec}-{end_sec} s")
plt.ylabel("Amplitúda (mV)")
plt.grid(True)

# Prefiltrovaný signál (úsek)
plt.subplot(2, 1, 2)
plt.plot(t, clean_pref[start_idx:end_idx], color="orange", linewidth=0.8)
plt.title(f"Prefiltrovaný EKG signál (clean_ecg_v2) – {start_sec}-{end_sec} s")
plt.xlabel("Čas (s)")
plt.ylabel("Amplitúda (mV)")
plt.grid(True)

plt.tight_layout()
plt.savefig("ekg_vysek_porovnanie.png", dpi=300)
print("Obrázok uložený ako ekg_vysek_porovnanie.png")