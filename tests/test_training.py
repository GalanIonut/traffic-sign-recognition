import json

import keras
import numpy as np
import pytest

from traffic_signs.dataset import load_train_index
from traffic_signs.models import build_cnn
from traffic_signs.pipeline import make_dataset
from traffic_signs.preprocessing import prepare
from traffic_signs.training import train_model


@pytest.fixture
def df(fake_gtsrb):
    return load_train_index(fake_gtsrb)


@pytest.fixture
def trained(df, tmp_path):
    """Antreneaza 2 epoci pe setul fals; intoarce (model, istoric, run_dir, val_ds)."""
    train_ds = make_dataset(df, 48, batch_size=8, training=True)
    val_ds = make_dataset(df, 48, batch_size=8, training=False)
    model = build_cnn()
    run_dir = tmp_path / "run"
    history = train_model(model, train_ds, val_ds, run_dir, epochs=2,
                          run_config={"run_name": "t"})
    return model, history, run_dir, val_ds


def test_train_model_writes_run_files(trained):
    _, history, run_dir, _ = trained
    assert (run_dir / "model.keras").exists()
    assert len(open(run_dir / "history.csv").read().strip().splitlines()) == 3
    cfg = json.loads((run_dir / "config.json").read_text())
    for key in ("run_name", "epochs_run", "best_epoch", "best_val_loss",
                "best_val_accuracy", "train_seconds", "params"):
        assert key in cfg
    assert cfg["epochs_run"] == 2
    assert list(history["epoch"]) == [1, 2]
    for col in ("loss", "accuracy", "val_loss", "val_accuracy", "learning_rate"):
        assert col in history.columns


def test_saved_model_reloads_same_predictions(trained):
    model, _, run_dir, val_ds = trained
    images, _ = next(iter(val_ds))
    images = images.numpy()
    reloaded = keras.models.load_model(run_dir / "model.keras")
    np.testing.assert_allclose(model.predict(images, verbose=0),
                               reloaded.predict(images, verbose=0), atol=1e-5)


def test_existing_run_dir_refused(df, trained):
    model, _, run_dir, val_ds = trained
    with pytest.raises(FileExistsError, match="run"):
        train_model(model, val_ds, val_ds, run_dir, epochs=1)


def test_make_dataset_shuffle_without_augment(df):
    ds = make_dataset(df, 48, batch_size=len(df), training=True, augment=False)
    images, labels = next(iter(ds))
    images = images.numpy()
    expected = np.stack([
        prepare(r.path, 48, (r.roi_x1, r.roi_y1, r.roi_x2, r.roi_y2))
        for r in df.itertuples()
    ])
    order = [int(np.argmin(np.abs(expected - img).sum(axis=(1, 2, 3)))) for img in images]
    for img, i in zip(images, order):
        np.testing.assert_allclose(img, expected[i], atol=1e-6)
    assert sorted(order) == list(range(len(df)))
    assert order != list(range(len(df)))
