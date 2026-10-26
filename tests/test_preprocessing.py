import numpy as np
import pytest
from PIL import Image

from traffic_signs.preprocessing import crop_roi, load_image, preprocess


def test_load_image_rgb_shape_dtype(fake_gtsrb):
    path = next((fake_gtsrb / "Train" / "0").glob("*.png"))
    img = load_image(path)
    assert img.shape == (32, 30, 3)
    assert img.dtype == np.uint8


def test_load_image_accepts_str_and_bytes(fake_gtsrb):
    path = next((fake_gtsrb / "Train" / "0").glob("*.png"))
    ref = load_image(path)
    assert np.array_equal(load_image(str(path)), ref)
    assert np.array_equal(load_image(str(path).encode()), ref)


def test_load_grayscale_and_rgba_become_rgb(tmp_path):
    Image.fromarray(np.full((8, 6), 100, dtype=np.uint8), "L").save(tmp_path / "g.png")
    gray = load_image(tmp_path / "g.png")
    assert gray.shape == (8, 6, 3) and gray.dtype == np.uint8
    assert (gray == 100).all()

    rgba = np.zeros((8, 6, 4), dtype=np.uint8)
    rgba[..., :3] = (10, 20, 30)
    rgba[0, 0] = (10, 20, 30, 0)  # complet transparent
    rgba[1:, :, 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    out = load_image(tmp_path / "a.png")
    assert out.shape == (8, 6, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == (255, 255, 255)  # fundal alb
    assert tuple(out[1, 1]) == (10, 20, 30)


def test_load_palette_with_transparency_and_la(tmp_path):
    p = Image.new("P", (4, 4), 0)
    p.putpalette([0, 0, 0] * 256)
    p.save(tmp_path / "p.png", transparency=0)
    assert (load_image(tmp_path / "p.png") == 255).all()

    la = np.zeros((4, 4, 2), dtype=np.uint8)
    Image.fromarray(la, "LA").save(tmp_path / "la.png")
    assert (load_image(tmp_path / "la.png") == 255).all()


def test_crop_roi_inclusive():
    img = np.zeros((32, 30, 3), dtype=np.uint8)
    assert crop_roi(img, 3, 3, 26, 28).shape == (26, 24, 3)


@pytest.mark.parametrize("roi", [(-5, -5, 100, 100), (40, 40, 50, 50)])
def test_crop_roi_clamps_out_of_bounds(roi):
    img = np.arange(32 * 30 * 3, dtype=np.uint8).reshape(32, 30, 3)
    assert np.array_equal(crop_roi(img, *roi), img)


@pytest.mark.parametrize("size", [48, 96])
def test_preprocess_shape_range(size):
    img = np.random.default_rng(42).integers(0, 256, (26, 24, 3), dtype=np.uint8)
    out = preprocess(img, size)
    assert out.shape == (size, size, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_load_image_accepts_file_like(fake_gtsrb):
    import io
    path = next((fake_gtsrb / "Train" / "0").glob("*.png"))
    img = load_image(io.BytesIO(path.read_bytes()))
    assert img.shape == (32, 30, 3)
    assert np.array_equal(img, load_image(path))


def test_prepare_matches_pipeline(fake_gtsrb):
    from traffic_signs.dataset import load_train_index
    from traffic_signs.pipeline import make_dataset
    from traffic_signs.preprocessing import prepare
    df = load_train_index(fake_gtsrb)
    row = df.iloc[0]
    roi = (int(row.roi_x1), int(row.roi_y1), int(row.roi_x2), int(row.roi_y2))
    out = prepare(row.path, 48, roi)
    assert out.shape == (48, 48, 3) and out.dtype == np.float32
    img, _ = next(iter(make_dataset(df, 48, batch_size=1, training=False)))
    assert np.allclose(out, img[0].numpy())
    assert prepare(row.path, 48).shape == (48, 48, 3)
