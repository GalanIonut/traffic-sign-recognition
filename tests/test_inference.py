"""Teste pentru logica de inferenta a aplicatiei (fara archive/ si fara ResNet)."""
import io
from pathlib import Path

import numpy as np
import pytest
from keras import layers
import keras
from PIL import Image

from traffic_signs import config
from traffic_signs.inference import (
    LOW_CONFIDENCE, MODEL_SPECS, InvalidImageError, Prediction, available_models,
    classify, format_percent, is_uncertain, meta_image_path, read_image,
    sample_class_id, sample_images,
)


def _probs(order: list[int]) -> np.ndarray:
    p = np.full(43, 0.001, dtype=np.float32)
    for rank, cls in enumerate(order):
        p[cls] = [0.6, 0.25, 0.1][rank]
    return p / p.sum()


def test_model_specs_sizes():
    sizes = {s.key: s.image_size for s in MODEL_SPECS}
    assert sizes["cnn"] == config.IMG_SIZE_CNN
    assert sizes["resnet50"] == config.IMG_SIZE_RESNET
    assert [s.key for s in MODEL_SPECS] == ["cnn", "resnet50"]


def test_available_models_only_existing_files(tmp_path):
    assert available_models(tmp_path) == []
    (tmp_path / "cnn_no_crop.keras").write_bytes(b"")
    assert [s.key for s in available_models(tmp_path)] == ["cnn"]


def test_read_image_rgba_and_grayscale(tmp_path):
    Image.new("RGBA", (10, 7), (255, 0, 0, 128)).save(tmp_path / "a.png")
    Image.new("L", (6, 9), 100).save(tmp_path / "b.png")
    a = read_image(tmp_path / "a.png")
    b = read_image(tmp_path / "b.png")
    assert a.shape == (7, 10, 3) and a.dtype == np.uint8
    assert b.shape == (9, 6, 3) and b.dtype == np.uint8


def test_read_image_invalid_raises_friendly(tmp_path):
    fake = tmp_path / "fake.png"
    fake.write_text("nu sunt o imagine")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    for source in (fake, empty, io.BytesIO(b"nu e imagine")):
        with pytest.raises(InvalidImageError) as exc:
            read_image(source)
        assert "imagine validă" in str(exc.value)
        assert exc.value.__cause__ is not None


def test_read_image_decompression_bomb_is_friendly(tmp_path, monkeypatch):
    Image.new("RGB", (40, 40), (1, 2, 3)).save(tmp_path / "big.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)  # peste 2x limita -> DecompressionBombError
    with pytest.raises(InvalidImageError) as exc:
        read_image(tmp_path / "big.png")
    assert "imagine validă" in str(exc.value)
    assert isinstance(exc.value.__cause__, Image.DecompressionBombError)


def test_classify_topk_sorted_and_named():
    probs = _probs([14, 13, 1])
    result = classify(lambda x, training=False: probs[None, :], np.zeros((20, 20, 3), np.uint8), 48)
    assert [p.class_id for p in result] == [14, 13, 1]
    assert result[0].name == "Stop"
    confs = [p.confidence for p in result]
    assert confs == sorted(confs, reverse=True)


def test_classify_feeds_model_correct_shape():
    seen = []

    def fake(x, training=False):
        seen.append(np.asarray(x))
        return _probs([14, 13, 1])[None, :]

    classify(fake, np.random.randint(0, 256, (3000, 4000, 3), dtype=np.uint8), 48)
    x = seen[0]
    assert x.shape == (1, 48, 48, 3) and x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_classify_real_keras_model_probabilities():
    keras.utils.set_random_seed(config.SEED)
    model = keras.Sequential([keras.Input((8, 8, 3)), layers.Flatten(),
                              layers.Dense(43, activation="softmax")])
    result = classify(model, np.random.randint(0, 256, (30, 20, 3), dtype=np.uint8), 8, k=43)
    assert len(result) == 43
    assert abs(sum(p.confidence for p in result) - 1.0) < 1e-4


def test_is_uncertain_threshold():
    assert LOW_CONFIDENCE == 0.5
    assert is_uncertain(Prediction(1, "x", 0.49))
    assert not is_uncertain(Prediction(1, "x", 0.5))


def test_format_percent():
    assert format_percent(0.9734) == "97,3%"
    assert format_percent(1.0) == "100,0%"
    assert format_percent(0.0) == "0,0%"


def test_meta_image_path(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "14.png").write_bytes(b"x")
    assert meta_image_path(14, tmp_path) == tmp_path / "meta" / "14.png"
    assert meta_image_path(15, tmp_path) is None


def test_sample_images_and_class_id(tmp_path):
    (tmp_path / "samples").mkdir()
    for name in ("class_14.png", "class_1.png"):
        (tmp_path / "samples" / name).write_bytes(b"x")
    assert [p.name for p in sample_images(tmp_path)] == ["class_1.png", "class_14.png"]
    assert sample_class_id(Path("class_14.png")) == 14


def test_app_assets_present():
    assert len(list((config.APP_ASSETS_DIR / "meta").glob("*.png"))) == 43
    samples = sample_images()
    assert len(samples) == 8
    assert all(0 <= sample_class_id(p) <= 42 for p in samples)
