import os

# Reduce zgomotul din logurile C++ ale TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib.use("Agg")  # backend fara ecran in teste

COLUMNS = ["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"]
WIDTH, HEIGHT = 30, 32
ROI = (3, 3, 26, 28)


def _save_random_png(path, rng):
    """Salveaza o imagine RGB aleatoare de 30x32 (latime x inaltime)."""
    pixels = rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def _row(rel_path, class_id):
    return [WIDTH, HEIGHT, *ROI, class_id, rel_path]


@pytest.fixture
def fake_gtsrb(tmp_path):
    """Set GTSRB fals: 3 clase, 3 track-uri/clasa, 4 cadre/track; 6 imagini test."""
    rng = np.random.default_rng(42)
    train_rows = []
    for c in range(3):
        for t in range(3):
            for f in range(4):
                rel = f"Train/{c}/{c:05d}_{t:05d}_{f:05d}.png"
                _save_random_png(tmp_path / rel, rng)
                train_rows.append(_row(rel, c))
    pd.DataFrame(train_rows, columns=COLUMNS).to_csv(tmp_path / "Train.csv", index=False)

    test_rows = []
    for i, c in enumerate([0, 1, 2, 0, 1, 2]):
        rel = f"Test/{i:05d}.png"
        _save_random_png(tmp_path / rel, rng)
        test_rows.append(_row(rel, c))
    pd.DataFrame(test_rows, columns=COLUMNS).to_csv(tmp_path / "Test.csv", index=False)
    return tmp_path
