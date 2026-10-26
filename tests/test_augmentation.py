import numpy as np
from tensorflow import keras

from traffic_signs.augmentation import build_augmentation, compute_class_weights


def test_no_flip_layer():
    model = build_augmentation()
    assert not any(isinstance(layer, keras.layers.RandomFlip) for layer in model.layers)


def test_shape_and_range_preserved():
    x = np.random.default_rng(0).random((8, 48, 48, 3), dtype=np.float32)
    y = build_augmentation()(x, training=True).numpy()
    assert y.shape == x.shape
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_does_not_mirror():
    img = np.zeros((48, 48, 3), dtype=np.float32)
    img[:, :24] = 1.0
    batch = np.repeat(img[None], 32, axis=0)
    y = build_augmentation()(batch, training=True).numpy()
    left = y[:, :, 0:12].mean(axis=(1, 2, 3))
    right = y[:, :, 36:48].mean(axis=(1, 2, 3))
    assert np.all(left > right)


def test_inference_is_identity():
    x = np.random.default_rng(1).random((4, 48, 48, 3), dtype=np.float32)
    y = build_augmentation()(x, training=False).numpy()
    assert np.allclose(y, x)


def test_class_weights_balanced():
    w = compute_class_weights([0] * 10 + [1] * 30)
    assert set(w) == {0, 1}
    assert abs(w[0] - 2.0) < 1e-6
    assert abs(w[1] - 2 / 3) < 1e-3
    equal = compute_class_weights([0, 1, 2] * 5)
    assert all(abs(v - 1.0) < 1e-9 for v in equal.values())
    assert all(type(k) is int and type(v) is float for k, v in w.items())
