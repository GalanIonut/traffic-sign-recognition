"""Impartirea train/validare pe track-uri, stratificata pe clase."""
import os
from pathlib import Path

import numpy as np
import pandas as pd


def split_by_track(df: pd.DataFrame, val_fraction: float = 0.15,
                   seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imparte pe track-uri: niciun track nu apare si in train, si in val."""
    if df["track_id"].isna().any():
        raise ValueError("Exista randuri fara track_id; impartirea pe track-uri nu e posibila")
    rng = np.random.default_rng(seed)
    val_tracks: set[str] = set()
    for class_id in sorted(df["class_id"].unique()):
        tracks = sorted(df.loc[df["class_id"] == class_id, "track_id"].unique())
        if len(tracks) < 2:
            raise ValueError(f"Clasa {class_id} are doar {len(tracks)} track-uri; minim 2")
        shuffled = rng.permutation(tracks)
        n_val = max(1, round(val_fraction * len(tracks)))
        val_tracks.update(shuffled[:n_val])
    is_val = df["track_id"].isin(val_tracks)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def save_split(train: pd.DataFrame, val: pd.DataFrame, out_path: Path,
               data_dir: Path) -> None:
    """Salveaza CSV cu coloana `split`; `path` e relativ la data_dir."""
    data_dir = Path(data_dir).resolve()
    out = pd.concat([train.assign(split="train"), val.assign(split="val")],
                    ignore_index=True)
    out["path"] = [Path(os.path.relpath(p, data_dir)).as_posix() for p in out["path"]]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def load_split(out_path: Path, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Citeste CSV-ul si reface caile absolute; intoarce (train, val)."""
    data_dir = Path(data_dir).resolve()
    df = pd.read_csv(out_path, dtype={"track_id": str})
    df["path"] = [str((data_dir / p).resolve()) for p in df["path"]]
    parts = []
    for name in ("train", "val"):
        part = df[df["split"] == name].drop(columns="split")
        parts.append(part.reset_index(drop=True))
    return parts[0], parts[1]
