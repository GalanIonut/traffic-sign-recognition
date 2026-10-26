"""Teste pentru ResNet50 cu preprocesare in model (fara greutati ImageNet)."""
import numpy as np
import pytest
from tensorflow import keras
from tensorflow.keras import layers

from traffic_signs.models import ResNetPreprocess, build_resnet50, unfreeze_top

SIZE = 32


@pytest.fixture(scope="module")
def shared_model():
    return build_resnet50(input_size=SIZE, weights=None)


def test_output_shape_and_softmax(shared_model):
    x = np.random.default_rng(0).random((2, SIZE, SIZE, 3), dtype="float32")
    y = shared_model.predict(x, verbose=0)
    assert y.shape == (2, 43)
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-4)


def test_preprocess_layer_matches_keras():
    x = np.random.default_rng(1).random((1, 8, 8, 3)).astype("float32")
    expected = keras.applications.resnet50.preprocess_input(x * 255.0)
    assert np.allclose(np.asarray(ResNetPreprocess()(x)), np.asarray(expected))


def test_base_frozen_initially(shared_model):
    names = {w.path.split("/")[0] for w in shared_model.trainable_weights}
    assert names == {"classifier"}
    assert len(shared_model.trainable_weights) == 2


def test_unfreeze_top_keeps_batchnorm_frozen():
    model = build_resnet50(input_size=SIZE, weights=None)
    n = unfreeze_top(model)
    assert n > 0
    base = model.get_layer("resnet50")
    conv5 = [l for l in base.layers if l.name.startswith("conv5_")]
    for l in conv5:
        if not isinstance(l, layers.BatchNormalization):
            assert l.trainable
    assert any(l.trainable and l.weights for l in conv5)
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            assert not l.trainable
        if l.name.startswith("conv4_"):
            assert not l.trainable
    trainable_ids = {id(w) for w in model.trainable_weights}
    assert not any(
        id(w) in trainable_ids
        for l in base.layers if isinstance(l, layers.BatchNormalization)
        for w in l.weights
    )


def test_save_and_reload_same_predictions(shared_model, tmp_path):
    x = np.random.default_rng(2).random((2, SIZE, SIZE, 3), dtype="float32")
    before = shared_model.predict(x, verbose=0)
    shared_model.save(tmp_path / "m.keras")
    loaded = keras.models.load_model(tmp_path / "m.keras")
    assert np.allclose(before, loaded.predict(x, verbose=0), atol=1e-5)
