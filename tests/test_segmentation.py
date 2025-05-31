from pathlib import Path
from src.preprocessing.load import load_record
from src.preprocessing.segmentation import segment_beats

def test_segment_length():
    rec = load_record("100", base_dir=Path("data/mit"))
    segs = segment_beats(rec.signal, rec.r_peaks, rec.fs)
    if segs.size > 0:  # aspoň jeden segment
        assert segs.shape[1] == int(round(0.6 * rec.fs))