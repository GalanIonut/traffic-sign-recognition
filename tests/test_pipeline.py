import numpy as np
import pytest

from traffic_signs.dataset import load_train_index
from traffic_signs.pipeline import make_dataset


@pytest.fixture
def df(fake_gtsrb):
    return load_train_index(fake_gtsrb)


def test_batch_shapes_and_dtypes(df):
    ds = make_dataset(df, image_size=48, batch_size=8)
    images, labels = next(iter(ds))
    assert images.shape == (8, 48, 48, 3)
    assert labels.shape == (8, 43)
    assert images.dtype.name == "float32"
    assert labels.dtype.name == "float32"


def test_labels_one_hot_match_df(df):
    ds = make_dataset(df, image_size=48, batch_size=8, training=False)
    ids = np.concatenate([lab.numpy().argmax(axis=1) for _, lab in ds])
    assert ids.tolist() == df.class_id.tolist()


def test_covers_all_rows(df):
    ds = make_dataset(df, image_size=48, batch_size=8)
    assert sum(img.shape[0] for img, _ in ds) == len(df) == 36


def test_eval_is_deterministic(df):
    ds = make_dataset(df, image_size=48, batch_size=8, training=False)
    a = np.concatenate([img.numpy() for img, _ in ds])
    b = np.concatenate([img.numpy() for img, _ in ds])
    np.testing.assert_array_equal(a, b)


def test_values_in_range_training(df):
    ds = make_dataset(df, image_size=48, batch_size=8, training=True)
    for img, _ in ds:
        assert img.numpy().min() >= 0.0
        assert img.numpy().max() <= 1.0


def test_crop_flag(df):
    one = df.iloc[:1]
    a = next(iter(make_dataset(one, 48, batch_size=1, crop=True)))[0].numpy()
    b = next(iter(make_dataset(one, 48, batch_size=1, crop=False)))[0].numpy()
    assert not np.array_equal(a, b)
