"""
Tento súbor slúži ako centrálne miesto na definovanie konfigurácie projektu,
najmä dátových ciest a zoznamov používaných pri spracovaní dát z MIT-BIH Arrhythmia databázy.

Obsahuje:
- Definíciu koreňového adresára projektu.
- Dátové cesty k MIT-BIH datasetu.
- Zoznam použitých príznakov (features) pre fuzzy klasifikáciu.
- Rozdelenie datasetu na trénovaciu (DS1) a testovaciu (DS2) množinu.
- Definíciu použitého EKG kanála (lead).
"""

from pathlib import Path

# ────────── Koreň projektu (úroveň nad src) ──────────
ROOT = Path(__file__).resolve().parents[1]

# ────────── Dátové cesty ──────────
DATA_DIR = ROOT / "data" / "mit"          # WFDB súbory

# Po prvej definícii môžeš ponechať alias pre spätnú kompatibilitu
DATA_PATH = DATA_DIR   # alias na rovnakú Path

# ────────── Zoznam záznamov ──────────
FIS_FEATURES = [
    'R_amplitude', 'P_amplitude', 'T_amplitude', 'RR0_s', 'RR1_s', 'Heart_rate_bpm', 'PR_ms'
]
DS1 = ['115', '103', '112', '219', '207', '122', '215', '230', '212', '116', '214', '201', '205', '232', '100', '203',
       '202', '111', '118', '208', '101', '228', '105', '102', '210', '209', '107', '221', '117', '106', '231', '233',
       '104', '114', '222', '123', '200', '217']

DS2 = ['124', '220', '119', '223', '108', '213', '121', '234', '109', '113']
LEAD = "MLII"



