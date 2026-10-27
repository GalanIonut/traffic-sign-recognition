import pandas as pd
import pytest
import yaml
from PIL import Image

from traffic_signs import config
from traffic_signs.detection import (
    SPLITS, build_yolo_dataset, load_gt, split_of, to_yolo_box, yolo_label_lines,
)

def _gt_text():
    rows = ["00001.ppm;10;20;30;40;7", "00001.ppm;50;60;70;80;14", "00001.ppm;1;2;9;9;3",
            "00000.ppm;5;5;20;20;1", "00500.ppm;5;5;20;20;1", "00600.ppm;5;5;20;20;2",
            "00700.ppm;6;6;30;30;4"]
    return "\n".join(rows) + "\n"


@pytest.fixture
def fake_all_images(tmp_path):
    """Toate cele 900 de poze (mici), cu 3 semne pe 00001 si 00002 fara semne."""
    images_dir = tmp_path / "raw900"
    for n in range(900):
        sub = "Train" if n < 600 else "Test"
        d = images_dir / sub / "images"
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(d / f"{n:05d}.jpg")
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text(_gt_text())
    return images_dir, load_gt(gt_path), tmp_path / "out900"


def test_load_gt_parses_crlf_and_spaces(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_bytes(b"00001.ppm;10;20;30;40;7 \r\n\n00002.ppm;1;2;3;4;14\r\n")
    gt = load_gt(p)
    assert len(gt) == 2
    assert gt["image_id"].tolist() == ["00001", "00002"]
    assert gt["class_id"].tolist() == [7, 14]


def test_to_yolo_box_formula():
    assert to_yolo_box(0, 0, 1359, 799, 1360, 800) == (0.5, 0.5, 1.0, 1.0)
    got = to_yolo_box(774, 411, 815, 446, 1360, 800)
    assert got == pytest.approx((0.584559, 0.536250, 0.030882, 0.045000), abs=1e-6)


def test_yolo_label_lines_all_class_zero():
    boxes = pd.DataFrame({"x1": [1, 2, 3], "y1": [1, 2, 3], "x2": [10, 20, 30],
                          "y2": [10, 20, 30], "class_id": [5, 6, 7]})
    lines = yolo_label_lines(boxes, 100, 100)
    assert len(lines) == 3
    assert all(l.startswith("0 ") and len(l.split()) == 5 for l in lines)


def test_split_of_boundaries():
    assert [split_of(n) for n in (0, 499, 500, 599, 600, 899)] == \
        ["train", "train", "val", "val", "test", "test"]
    assert set(SPLITS) == {"train", "val", "test"}


def test_build_dataset_multiple_boxes_same_image(fake_all_images):
    images_dir, gt, out = fake_all_images
    build_yolo_dataset(gt, images_dir, out)
    lines = (out / "labels" / "train" / "00001.txt").read_text().splitlines()
    assert len(lines) == 3


def test_build_dataset_empty_image_gets_empty_label(fake_all_images):
    images_dir, gt, out = fake_all_images
    counts = build_yolo_dataset(gt, images_dir, out)
    assert (out / "labels" / "train" / "00002.txt").read_text() == ""
    assert (out / "images" / "train" / "00002.jpg").exists()
    assert counts["train"]["images"] == 500 and counts["val"]["images"] == 100
    assert counts["test"]["images"] == 300
    assert counts["train"]["boxes"] == 4 and counts["train"]["empty"] == 498


def test_build_dataset_missing_image_raises(fake_all_images):
    images_dir, gt, out = fake_all_images
    (images_dir / "Train" / "images" / "00123.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="00123"):
        build_yolo_dataset(gt, images_dir, out)


def test_build_dataset_rerun_is_clean(fake_all_images):
    images_dir, gt, out = fake_all_images
    first = build_yolo_dataset(gt, images_dir, out)
    (out / "stray.txt").write_text("vechi")
    (out / "labels" / "train" / "99999.txt").write_text("")
    second = build_yolo_dataset(gt, images_dir, out)
    assert not (out / "stray.txt").exists()
    assert not (out / "labels" / "train" / "99999.txt").exists()
    assert first == second


def test_build_dataset_yaml(fake_all_images):
    images_dir, gt, out = fake_all_images
    build_yolo_dataset(gt, images_dir, out)
    cfg = yaml.safe_load((out / "dataset.yaml").read_text())
    assert cfg["names"] == {0: "semn"}
    assert cfg["train"] == "images/train"
    assert cfg["val"] == "images/val" and cfg["test"] == "images/test"
    assert cfg["path"] == str(out.resolve())


def test_real_gt_counts():
    path = config.GTSDB_DIR / "gt.txt"
    if not path.exists():
        pytest.skip("gt.txt real lipseste")
    gt = load_gt(path)
    assert len(gt) == 1213
    assert (gt["image_id"].astype(int) < 600).sum() == 852
    assert gt.groupby("image_id").size().max() == 6
