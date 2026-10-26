import numpy as np
import pandas as pd
from PIL import Image

from traffic_signs import config
from traffic_signs.dataset import load_test_index, load_train_index
from traffic_signs.eda import (class_distribution, plot_class_distribution,
                               plot_class_grid, plot_size_histogram,
                               plot_split_distribution)
from traffic_signs.split import split_by_track


def test_class_distribution_counts(fake_gtsrb):
    df = load_train_index(fake_gtsrb)
    counts = class_distribution(df)
    assert len(counts) == 43
    assert list(counts.index) == list(range(43))
    assert counts.iloc[:3].tolist() == [12, 12, 12]
    assert (counts.iloc[3:] == 0).all()
    assert np.issubdtype(counts.dtype, np.integer)


def test_plots_write_png(fake_gtsrb, tmp_path):
    train_all = load_train_index(fake_gtsrb)
    test = load_test_index(fake_gtsrb)
    train, val = split_by_track(train_all)
    meta = tmp_path / "Meta"
    meta.mkdir()
    for i in range(config.NUM_CLASSES):
        mode = "RGBA" if i % 2 else "RGB"
        Image.new(mode, (20, 20), (i * 5, 100, 50, 255)[: len(mode)]).save(meta / f"{i}.png")

    out = tmp_path / "figs"
    out.mkdir()
    calls = [
        plot_class_distribution(class_distribution(train_all), out / "a.png", "Titlu"),
        plot_size_histogram(train_all, out / "b.png"),
        plot_class_grid(meta, out / "c.png"),
        plot_split_distribution(train, val, test, out / "d.png"),
    ]
    assert calls == [None] * 4
    for name in "abcd":
        assert (out / f"{name}.png").stat().st_size > 0
