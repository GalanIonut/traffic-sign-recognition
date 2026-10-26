"""Indexul setului GTSRB: citeste Train.csv / Test.csv in DataFrame-uri."""
from pathlib import Path

import pandas as pd

COLUMNS = ["path", "class_id", "track_id", "width", "height",
           "roi_x1", "roi_y1", "roi_x2", "roi_y2"]
_RENAME = {"Width": "width", "Height": "height", "Roi.X1": "roi_x1",
           "Roi.Y1": "roi_y1", "Roi.X2": "roi_x2", "Roi.Y2": "roi_y2",
           "ClassId": "class_id", "Path": "path"}
_INT_COLS = ["class_id", "width", "height", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]


def parse_track_id(filename: str) -> str | None:
    """'00020_00005_00012.png' -> '00020_00005'; fara track (test) -> None."""
    parts = Path(filename).stem.split("_")
    if len(parts) != 3:
        return None
    return f"{parts[0]}_{parts[1]}"


def _load_index(data_dir: Path, csv_name: str, with_tracks: bool) -> pd.DataFrame:
    """Citeste CSV-ul; caile devin absolute, relativ la folderul CSV-ului."""
    data_dir = Path(data_dir).resolve()
    df = pd.read_csv(data_dir / csv_name).rename(columns=_RENAME)
    df[_INT_COLS] = df[_INT_COLS].astype(int)
    if with_tracks:
        df["track_id"] = df["path"].map(lambda p: parse_track_id(Path(p).name))
    else:
        df["track_id"] = pd.Series([None] * len(df), dtype=object, index=df.index)
    df["path"] = [str((data_dir / p).resolve()) for p in df["path"]]
    return df[COLUMNS]


def load_train_index(data_dir: Path) -> pd.DataFrame:
    """Indexul setului de antrenare (cu track_id din numele fisierului)."""
    return _load_index(data_dir, "Train.csv", with_tracks=True)


def load_test_index(data_dir: Path) -> pd.DataFrame:
    """Indexul setului de test (fara track-uri: track_id = None)."""
    return _load_index(data_dir, "Test.csv", with_tracks=False)
