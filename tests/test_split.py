from pathlib import Path

import pandas as pd
import pytest

from traffic_signs.dataset import load_train_index
from traffic_signs.split import load_split, save_split, split_by_track


def _synthetic(n_tracks=10, classes=(0, 1, 2), frames=2):
    """DataFrame sintetic: n_tracks track-uri/clasa, `frames` cadre/track."""
    rows = []
    for c in classes:
        for t in range(n_tracks):
            for f in range(frames):
                rows.append({"path": f"/x/{c:05d}_{t:05d}_{f:05d}.png",
                             "class_id": c, "track_id": f"{c:05d}_{t:05d}"})
    return pd.DataFrame(rows)


@pytest.fixture
def index(fake_gtsrb):
    return load_train_index(fake_gtsrb)


def test_no_shared_tracks(index):
    train, val = split_by_track(index, val_fraction=0.34)
    assert set(train["track_id"]).isdisjoint(set(val["track_id"]))


def test_every_class_in_both(index):
    train, val = split_by_track(index, val_fraction=0.34)
    assert set(train["class_id"]) == {0, 1, 2}
    assert set(val["class_id"]) == {0, 1, 2}
    assert val["track_id"].nunique() == 3  # 1 track/clasa


def test_all_rows_kept(index):
    train, val = split_by_track(index, val_fraction=0.34)
    assert len(train) + len(val) == 36
    assert pd.concat([train, val])["path"].is_unique


def test_deterministic(index):
    _, v1 = split_by_track(index, val_fraction=0.34, seed=42)
    _, v2 = split_by_track(index, val_fraction=0.34, seed=42)
    assert list(v1["path"]) == list(v2["path"])


def test_different_seed_changes_split():
    df = _synthetic(n_tracks=10, classes=(0, 1, 2, 3, 4))
    _, v42 = split_by_track(df, val_fraction=0.3, seed=42)
    _, v7 = split_by_track(df, val_fraction=0.3, seed=7)
    assert set(v42["path"]) != set(v7["path"])


def test_single_track_class_raises():
    df = _synthetic(n_tracks=3, classes=(0, 1))
    df = df[~((df["class_id"] == 1) & (df["track_id"] != "00001_00000"))]
    with pytest.raises(ValueError, match="Clasa 1 are doar 1 track-uri"):
        split_by_track(df)


def test_missing_track_raises(index):
    index.loc[0, "track_id"] = None
    with pytest.raises(ValueError, match="track_id"):
        split_by_track(index, val_fraction=0.34)


def test_save_load_roundtrip(index, fake_gtsrb, tmp_path):
    train, val = split_by_track(index, val_fraction=0.34)
    out = tmp_path / "out" / "split.csv"
    save_split(train, val, out, fake_gtsrb)
    saved = pd.read_csv(out)
    assert set(saved["split"]) == {"train", "val"}
    assert not any(Path(p).is_absolute() for p in saved["path"])
    train2, val2 = load_split(out, fake_gtsrb)
    pd.testing.assert_frame_equal(train.reset_index(drop=True), train2)
    pd.testing.assert_frame_equal(val.reset_index(drop=True), val2)
