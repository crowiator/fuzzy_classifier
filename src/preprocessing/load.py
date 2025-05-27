from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from collections import Counter
from typing import List

import numpy as np
import wfdb

from src.config import DATA_DIR, LEAD

from src.preprocessing.annotation_mapping import map_mitbih_annotation

# ────────── Konštanty ──────────
# Zoznam anotácií, ktoré nechceme zahrnúť do analýzy (napr. artefakty, chybné hodnoty)
EXCLUDED_ANNOTATIONS: set[str] = {"+", "~", "|", '"', "x", "!", "[", "]"}
PREFERRED_LEADS: list[str] = ["MLII", "II", "V1", "V2", "V5", "V6"]

# Základné logovanie (ak už projekt neinicializuje logging centrálne)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


@dataclass(frozen=True, slots=True)
class LoadedRecord:
    signal: np.ndarray  # 1‑D signál zvoleného zvodu (mV)
    fs: int  # vzorkovacia frekvencia (Hz)
    r_peaks: np.ndarray  # indexy R‑vrcholov (vo vzorkách)
    labels_raw: List[str]  # pôvodné MIT‑BIH symboly (bez artefaktov)
    labels_aami: List[str]  # zmapované na 5 AAMI tried
    counts: Counter  # početnosť AAMI tried v tomto zázname
    lead: str  # názov použitého zvodu


def choose_lead(rec: wfdb.Record, preferred: list[str] = PREFERRED_LEADS) -> str:
    """Vráti prvý preferovaný zvod, ktorý existuje v zázname; inak prvý."""
    for ld in preferred:
        if ld in rec.sig_name:
            return ld
    return rec.sig_name[0]




def load_record(record_id: str, *, lead: str | None = LEAD, base_dir: Path = DATA_DIR) -> LoadedRecord:
    """
        Načíta MIT-BIH záznam z lokálneho priečinka data/mit
        a vráti (raw_signal, fs, ann, used_lead).
        Ak `lead` je None alebo v zázname chýba, vyberie sa automaticky.

                Raises
            ------
            FileNotFoundError
                Ak .dat súbor neexistuje v `base_dir`.
            wfdb.exceptions.WFDBReaderError
                Ak WFDB súbory sú poškodené alebo neúplné.
        """
    rec_path = base_dir / record_id
    dat_path = rec_path.with_suffix(".dat")
    if not dat_path.exists():
        raise FileNotFoundError(f"Record {record_id} not found in {base_dir}")

    # Načítanie signálu + anotácií
    rec = wfdb.rdrecord(str(rec_path))
    ann = wfdb.rdann(str(rec_path), "atr")

    # Výber zvodu
    if lead is None or lead not in rec.sig_name:
        lead = choose_lead(rec)
        logging.warning(f"Lead MLII nie je k dispozícii, používam {lead}")
    idx = rec.sig_name.index(lead)

    signal = rec.p_signal[:, idx].astype(np.float32)

    # Prefiltrovanie anotácií
    r_peaks: list[int] = []
    labels_raw: list[str] = []
    labels_aami: list[str] = []

    for pos, sym in zip(ann.sample, ann.symbol):
        if sym in EXCLUDED_ANNOTATIONS:
            continue
        r_peaks.append(pos)
        labels_raw.append(sym)
        labels_aami.append(map_mitbih_annotation(sym))

    r_peaks_arr = np.asarray(r_peaks, dtype=np.int32)
    counts = Counter(labels_aami)

    return LoadedRecord(
        signal=signal,
        fs=int(rec.fs),
        r_peaks=r_peaks_arr,
        labels_raw=labels_raw,
        labels_aami=labels_aami,
        counts=counts,
        lead=lead,
    )

