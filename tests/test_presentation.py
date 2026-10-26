"""Teste pentru graficele de prezentare (date false, fără archive/ și fără model)."""
import numpy as np
import pandas as pd

from traffic_signs import presentation as p


def _img(rng, size=40, channels=3):
    return rng.integers(0, 256, size=(size, size, channels), dtype=np.uint8)


def _probs(rng, n, k=43):
    raw = rng.random((n, k)) ** 6
    return raw / raw.sum(axis=1, keepdims=True)


def _assert_png(path):
    assert path.exists() and path.stat().st_size > 5000


def test_top_k_valori_exacte():
    probs = np.array([0.1, 0.6, 0.05, 0.25])
    assert p.top_k(probs, 3) == [(1, 0.6), (3, 0.25), (0, 0.1)]


def test_top_confused_pairs_exact():
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0])
    pairs = p.top_confused_pairs(y_true, y_pred, k=2)
    # (0->1): 2 din 4; (1->2) si (2->0): 1 din 2 / 1 din 3
    assert pairs[0] == (0, 1, 2, 0.5)
    assert pairs[1] == (1, 2, 1, 0.5)
    assert len(p.top_confused_pairs(y_true, y_pred, k=10)) == 3


def test_select_examples_determinist():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 100)
    y_pred = y_true.copy()
    y_pred[:10] = (y_pred[:10] + 1) % 5
    probs = _probs(rng, 100, 5)
    a = p.select_examples(y_true, y_pred, probs, 6, 2, seed=42)
    b = p.select_examples(y_true, y_pred, probs, 6, 2, seed=42)
    assert a == b and len(a) == 8
    assert sum(y_true[i] == y_pred[i] for i in a) == 6
    assert sum(y_true[i] != y_pred[i] for i in a) == 2


def test_plot_pipeline(tmp_path):
    rng = np.random.default_rng(0)
    out = tmp_path / "a.png"
    p.plot_pipeline(_img(rng, 60), _img(rng, 40), _img(rng, 48),
                    [("Stop", 0.97), ("Cedează trecerea", 0.02), ("Altceva", 0.01)],
                    "Stop", 1331275, out)
    _assert_png(out)


def test_plot_predictions(tmp_path):
    rng = np.random.default_rng(1)
    items = []
    for i in range(8):
        top3 = [(f"Clasa {i}", 0.8), ("Alta", 0.15), ("A treia", 0.05)]
        items.append({"image": _img(rng, 48), "top3": top3,
                      "correct": i < 6, "true_name": f"Clasa {i}"})
    out = tmp_path / "b.png"
    p.plot_predictions(items, out, "Titlu", "Subtitlu")
    _assert_png(out)


def test_plot_per_class(tmp_path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"class_id": range(43), "name": [f"Semn {i}" for i in range(43)],
                       "recall": np.linspace(0.85, 1.0, 43)})
    metas = {i: _img(rng, 30) for i in range(43)}
    out = tmp_path / "c.png"
    p.plot_per_class(df, metas, out)
    _assert_png(out)


def test_plot_experiments(tmp_path):
    rows = [("De bază — decupate", 0.9903, True), ("De bază — nedecupate", 0.9177, False),
            ("Fără regularizare — decupate", 0.9831, True),
            ("Fără decupare — nedecupate", 0.9846, False)]
    out = tmp_path / "d.png"
    p.plot_experiments(rows, out)
    _assert_png(out)


def test_plot_overfitting(tmp_path):
    ep = np.arange(1, 11)
    h1 = pd.DataFrame({"epoch": ep, "accuracy": 1 - 0.1 / ep, "val_accuracy": 1 - 0.12 / ep})
    h2 = pd.DataFrame({"epoch": ep, "accuracy": 1 - 0.05 / ep, "val_accuracy": 0.95 - 0.1 / ep})
    out = tmp_path / "e.png"
    p.plot_overfitting(h1, h2, out)
    _assert_png(out)


def test_plot_confusions(tmp_path):
    rng = np.random.default_rng(3)
    pairs = [(i, i + 1, 10 - i, 0.1 * (i + 1)) for i in range(5)]
    metas = {i: _img(rng, 30) for i in range(7)}
    names = {i: f"Semn {i}" for i in range(7)}
    out = tmp_path / "f.png"
    p.plot_confusions(pairs, metas, names, out, "Titlu", "Subtitlu")
    _assert_png(out)


def test_plot_dataset(tmp_path):
    rng = np.random.default_rng(4)
    stats = {"n_train": 39209, "n_test": 12630, "n_classes": 43, "avg_px": 50}
    metas = [_img(rng, 30) for _ in range(12)]
    out = tmp_path / "g.png"
    p.plot_dataset(stats, metas, out)
    _assert_png(out)
