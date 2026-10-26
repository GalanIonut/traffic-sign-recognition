import numpy as np
from tensorflow import keras

from traffic_signs.models import build_cnn, model_summary_text, plot_architecture


def test_output_shape_and_softmax():
    model = build_cnn()
    x = np.random.default_rng(0).random((4, 48, 48, 3), dtype=np.float32)
    out = model.predict(x, verbose=0)
    assert out.shape == (4, 43)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_dropout_flag():
    no_drop = build_cnn(dropout=0)
    assert not any(isinstance(l, keras.layers.Dropout) for l in no_drop.layers)
    drops = [l for l in build_cnn().layers if isinstance(l, keras.layers.Dropout)]
    assert len(drops) == 1 and drops[0].rate == 0.5


def test_batch_norm_flag():
    def count(m):
        return sum(isinstance(l, keras.layers.BatchNormalization) for l in m.layers)

    assert count(build_cnn(batch_norm=False)) == 0
    assert count(build_cnn()) == 3


def test_can_overfit_tiny_batch():
    keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    x = rng.random((32, 48, 48, 3), dtype=np.float32)
    y = keras.utils.to_categorical(rng.integers(0, 4, 32), 4)
    model = build_cnn(num_classes=4)
    model.compile(keras.optimizers.Adam(1e-3), "categorical_crossentropy")
    first = model.train_on_batch(x, y)
    for _ in range(29):
        last = model.train_on_batch(x, y)
    assert last < 0.5 * first


def test_summary_and_diagram(tmp_path):
    model = build_cnn()
    text = model_summary_text(model)
    assert "cnn_propriu" in text and "Total params" in text
    out = tmp_path / "arch.png"
    plot_architecture(model, out)
    assert out.exists() and out.stat().st_size > 0
