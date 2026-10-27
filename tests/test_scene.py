import numpy as np
import pandas as pd
import pytest

from traffic_signs import scene
from traffic_signs.inference import Prediction
from traffic_signs.scene import Detection, GroundTruth, PipelineConfig, SceneSign


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = np.array(xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.array(conf, dtype=float)


class _Result:
    def __init__(self, xyxy, conf):
        self.boxes = _Boxes(xyxy, conf)


class FakeYolo:
    def __init__(self, xyxy=(), conf=()):
        self.xyxy, self.conf = list(xyxy), list(conf)
        self.source = None
        self.kw = None

    def predict(self, source, **kw):
        self.source, self.kw = source, kw
        return [_Result(self.xyxy, self.conf)]


class FakeModel:
    def __init__(self, probs=None):
        self.calls = 0
        self.shape = None
        self.probs = probs

    def __call__(self, batch, training=False):
        self.calls += 1
        self.shape = batch.shape
        n = batch.shape[0]
        if self.probs is not None:
            return np.tile(np.asarray(self.probs, dtype=float), (n, 1))
        p = np.zeros((n, 43))
        p[:, 5], p[:, 7], p[:, 9] = 0.6, 0.3, 0.1
        return p


def _pred(cid, conf=0.9):
    return Prediction(cid, f"c{cid}", conf)


def _sign(number, box, score, cid):
    return SceneSign(number, box, score, (_pred(cid),))


def test_crop_with_margin_expands_each_side():
    img = np.zeros((100, 100, 3), np.uint8)
    assert scene.crop_with_margin(img, (40, 40, 60, 60), 0.1).shape == (24, 24, 3)


def test_crop_with_margin_clips_at_image_border():
    img = np.zeros((100, 100, 3), np.uint8)
    assert scene.crop_with_margin(img, (0, 0, 10, 10), 0.5).shape == (15, 15, 3)


def test_crop_never_empty():
    img = np.zeros((100, 100, 3), np.uint8)
    out = scene.crop_with_margin(img, (5, 5, 5, 5), 0.0)
    assert out.shape[0] >= 1 and out.shape[1] >= 1


def test_detect_sends_bgr_and_sorts_by_score():
    img = np.zeros((20, 20, 3), np.uint8)
    img[..., 0] = 255
    yolo = FakeYolo([(1, 1, 5, 5), (6, 6, 9, 9)], [0.3, 0.8])
    dets = scene.detect(yolo, img)
    assert (yolo.source[..., 2] == 255).all()
    assert yolo.kw["conf"] == 0.05
    assert yolo.kw["device"] == "cpu"
    assert yolo.kw["verbose"] is False
    assert [d.score for d in dets] == [0.8, 0.3]
    assert dets[0].box == (6.0, 6.0, 9.0, 9.0)


def test_classify_crops_single_batch():
    model = FakeModel()
    crops = [np.zeros((s, s + 2, 3), np.uint8) for s in (10, 20, 30)]
    tops = scene.classify_crops(model, crops, 48)
    assert model.calls == 1 and model.shape == (3, 48, 48, 3)
    assert len(tops) == 3
    assert [p.class_id for p in tops[0]] == [5, 7, 9]
    assert tops[0][0].confidence >= tops[0][1].confidence >= tops[0][2].confidence


def test_classify_crops_empty_makes_no_call():
    model = FakeModel()
    assert scene.classify_crops(model, [], 48) == []
    assert model.calls == 0


def test_run_pipeline_numbers_and_timings():
    img = np.zeros((50, 50, 3), np.uint8)
    yolo = FakeYolo([(1, 1, 10, 10), (20, 20, 30, 30)], [0.4, 0.9])
    cfg = PipelineConfig("cnn_no_crop.keras", 48, 0.1, 0.5)
    signs, timings = scene.run_pipeline(yolo, FakeModel(), img, cfg)
    assert [s.number for s in signs] == [1, 2]
    assert [s.det_score for s in signs] == [0.9, 0.4]
    assert signs[0].best.class_id == 5
    assert timings["detect_ms"] >= 0 and timings["classify_ms"] >= 0
    assert yolo.kw["conf"] == scene.DETECT_FLOOR


def test_filter_by_conf_keeps_numbers():
    signs = [_sign(1, (0, 0, 1, 1), 0.9, 1), _sign(2, (0, 0, 1, 1), 0.3, 2),
             _sign(3, (0, 0, 1, 1), 0.5, 3)]
    assert [s.number for s in scene.filter_by_conf(signs, 0.5)] == [1, 3]


def test_iou_basic():
    a = (0, 0, 10, 10)
    assert scene.iou(a, a) == 1.0
    assert scene.iou(a, (20, 20, 30, 30)) == 0.0
    assert scene.iou(a, (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_match_greedy_by_score():
    gts = [GroundTruth((0, 0, 10, 10), 1), GroundTruth((50, 50, 60, 60), 2)]
    signs = [_sign(1, (0, 0, 10, 10), 0.6, 1), _sign(2, (1, 0, 11, 10), 0.9, 1)]
    r = scene.match(signs, gts)
    assert r.pairs == ((1, 0),)
    assert r.false_signs == (0,)
    assert r.missed == (1,)


def test_statuses_correct_wrong_false():
    gts = [GroundTruth((0, 0, 10, 10), 1), GroundTruth((50, 50, 60, 60), 2)]
    signs = [_sign(1, (0, 0, 10, 10), 0.9, 1), _sign(2, (50, 50, 60, 60), 0.8, 9),
             _sign(3, (80, 80, 90, 90), 0.7, 3)]
    st = scene.sign_statuses(signs, gts, scene.match(signs, gts))
    assert st == {1: "correct", 2: "wrong", 3: "false"}


def test_counts_and_summarize():
    gts = [GroundTruth((0, 0, 10, 10), 1), GroundTruth((50, 50, 60, 60), 2),
           GroundTruth((70, 70, 80, 80), 3)]
    signs = [_sign(1, (0, 0, 10, 10), 0.9, 1), _sign(2, (50, 50, 60, 60), 0.8, 9),
             _sign(3, (90, 90, 99, 99), 0.7, 3)]
    c = scene.count(signs, gts)
    assert c == scene.Counts(3, 3, 2, 1)
    assert c + c == scene.Counts(6, 6, 4, 2)
    s = scene.summarize(c)
    assert s["detection_recall"] == pytest.approx(2 / 3)
    assert s["classification_accuracy"] == pytest.approx(1 / 2)
    assert s["e2e_recall"] == pytest.approx(1 / 3)
    assert s["e2e_precision"] == pytest.approx(1 / 3)
    assert s["e2e_f1"] == pytest.approx(1 / 3)


def test_summarize_zero_division_is_zero():
    s = scene.summarize(scene.Counts(0, 0, 0, 0))
    assert set(s) == {"detection_recall", "classification_accuracy", "e2e_recall",
                      "e2e_precision", "e2e_f1"}
    assert all(v == 0.0 for v in s.values())


def test_gt_for_image_makes_corners_exclusive():
    df = pd.DataFrame([("00001", 10, 20, 29, 39, 14), ("00002", 0, 0, 5, 5, 1)],
                      columns=["image_id", "x1", "y1", "x2", "y2", "class_id"])
    assert scene.gt_for_image(df, "00001") == [GroundTruth((10, 20, 30, 40), 14)]
    assert scene.gt_for_image(df, "00099") == []


def test_choose_config_tie_break():
    rows = [{"classifier": "cnn_baseline.keras", "margin": 0.10, "accuracy": 0.950},
            {"classifier": "cnn_no_crop.keras", "margin": 0.05, "accuracy": 0.9495},
            {"classifier": "cnn_no_crop.keras", "margin": 0.20, "accuracy": 0.90}]
    best = scene.choose_config(rows)
    assert best["classifier"] == "cnn_no_crop.keras" and best["margin"] == 0.05
    rows = [{"classifier": "cnn_baseline.keras", "margin": 0.05, "accuracy": 0.9500},
            {"classifier": "cnn_no_crop.keras", "margin": 0.05, "accuracy": 0.9495}]
    assert scene.choose_config(rows)["classifier"] == "cnn_no_crop.keras"


def _image(scores_correct):
    gt = [GroundTruth((0, 0, 10, 10), 1)]
    signs = [_sign(i + 1, (0, 0, 10, 10) if ok else (50, 50, 60, 60), s, 1)
             for i, (s, ok) in enumerate(scores_correct)]
    return signs, gt


def test_best_threshold_prefers_lower_on_tie():
    img = _image([(0.9, True)])
    assert scene.best_threshold([img], grid=(0.1, 0.5, 0.8)) == 0.1


def test_best_threshold_picks_max_f1():
    img = _image([(0.9, True), (0.3, False)])
    assert scene.best_threshold([img], grid=(0.1, 0.5, 0.95)) == 0.5


def test_pipeline_config_roundtrip(tmp_path):
    cfg = PipelineConfig("cnn_no_crop.keras", 48, 0.05, 0.35)
    path = tmp_path / "pipeline.json"
    scene.save_pipeline_config(cfg, path)
    assert scene.load_pipeline_config(path) == cfg
