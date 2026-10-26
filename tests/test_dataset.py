import os
from pathlib import Path

import pandas as pd

from traffic_signs.dataset import load_test_index, load_train_index, parse_track_id

COLS = ["path", "class_id", "track_id", "width", "height",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2"]


def test_train_index_columns_and_size(fake_gtsrb):
    df = load_train_index(fake_gtsrb)
    assert len(df) == 36
    assert list(df.columns) == COLS
    assert df["class_id"].dtype.kind == "i"
    assert df["width"].dtype.kind == "i"
    assert df["roi_x2"].dtype.kind == "i"


def test_train_paths_absolute_and_exist(fake_gtsrb):
    df = load_train_index(fake_gtsrb)
    assert all(isinstance(p, str) for p in df["path"])
    assert all(Path(p).is_absolute() and Path(p).exists() for p in df["path"])


def test_paths_independent_of_cwd(fake_gtsrb, tmp_path, monkeypatch):
    expected = load_train_index(fake_gtsrb)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    pd.testing.assert_frame_equal(load_train_index(fake_gtsrb), expected)
    # data_dir relativ la directorul curent
    rel = Path(os.path.relpath(fake_gtsrb, elsewhere))
    assert not rel.is_absolute()
    pd.testing.assert_frame_equal(load_train_index(rel), expected)


def test_track_ids(fake_gtsrb):
    df = load_train_index(fake_gtsrb)
    assert df["track_id"].nunique() == 9
    assert (df.groupby("track_id").size() == 4).all()


def test_test_index_has_no_tracks(fake_gtsrb):
    df = load_test_index(fake_gtsrb)
    assert len(df) == 6
    assert list(df.columns) == COLS
    assert all(t is None for t in df["track_id"])
    assert df["class_id"].tolist() == [0, 1, 2, 0, 1, 2]


def test_parse_track_id():
    assert parse_track_id("00020_00005_00012.png") == "00020_00005"
    assert parse_track_id("00012.png") is None
