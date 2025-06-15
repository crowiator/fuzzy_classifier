# config.py
from pathlib import Path

# ────────── Koreň projektu (úroveň nad src) ──────────
ROOT = Path(__file__).resolve().parents[1]

# ────────── Dátové cesty ──────────
DATA_DIR = ROOT / "data" / "mit"          # WFDB súbory
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
CACHE_DIR = RESULTS_DIR / "data"

# Po prvej definícii môžeš ponechať alias pre spätnú kompatibilitu
DATA_PATH = DATA_DIR   # alias na rovnakú Path
MIT_LOCAL_DIR = DATA_DIR  # zrušiteľné ak nepoužívaš

# ────────── Zoznam záznamov ──────────
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230"
]

DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234"
]
# pacemaker / artefakty
MISSING_RECORDS = ["102", "104", "107", "217"]
ALL_RECORDS = DS1_RECORDS + DS2_RECORDS
TRAIN_IDS = DS1_RECORDS
TEST_IDS = DS2_RECORDS
assert set(DS1_RECORDS).isdisjoint(DS2_RECORDS)
DEBUG_RECORDS = ["100"]   # ak potrebuješ

# ────────── Signálové parametre ──────────
LOWPASS_CUTOFF = 40
DWT_THRESHOLD = 0.15
QRS_THRESHOLD = 0.35
MOVING_WINDOW_SIZE = 10

# Segmentácia R-vlny
SEGMENT_PRE_R = 0.2
SEGMENT_POST_R = 0.4
MATCHING_TOLERANCE = 50  # ms

# Dataset split & random seed
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ────────── Výstupné priečinky / súbory ──────────
OUT_DIR = RESULTS_DIR / "mitdb_fuzzy_results"
FEAT_DIR = RESULTS_DIR / "exported_features"
LEAD = "MLII"

# --- zaisti existenciu adresárov ---
for _dir in (RESULTS_DIR, REPORTS_DIR, CACHE_DIR, OUT_DIR, FEAT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
