import json
import warnings

import numpy as np
import pandas as pd
import pytest

from traffic_signs.dataset import load_train_index
from traffic_signs.evaluation import (compare_runs, compute_metrics, load_history,
                                      plot_confusion_matrix, plot_history,
                                      plot_misclassified, predict_classes)
from traffic_signs.models import build_cnn


def test_compute_metrics_known_values():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # orice avertisment devine eroare
        metrics, per_class = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], num_classes=3)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert set(metrics) == {"accuracy", "precision_macro", "recall_macro",
                            "f1_macro", "f1_weighted"}
    assert len(per_class) == 3
    assert list(per_class["support"]) == [2, 2, 0]


def test_predict_classes_keeps_row_order(fake_gtsrb):
    df = load_train_index(fake_gtsrb).sample(frac=1, random_state=0)
    y_true, y_pred, probs = predict_classes(build_cnn(), df, batch_size=8)
    assert list(y_true) == list(df.class_id)
    assert len(y_pred) == len(df)
    assert probs.shape == (len(df), 43)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_plots_write_png(fake_gtsrb, tmp_path):
    history = pd.DataFrame({
        "epoch": [1, 2, 3], "loss": [2.0, 1.0, 0.5], "accuracy": [0.2, 0.5, 0.8],
        "val_loss": [2.1, 1.2, 1.3], "val_accuracy": [0.2, 0.4, 0.45],
        "learning_rate": [1e-3] * 3})
    plot_history(history, tmp_path / "curves.png", "test")

    df = load_train_index(fake_gtsrb)
    rng = np.random.default_rng(0)
    y_true = df.class_id.to_numpy()
    y_pred = rng.integers(0, 3, len(df))
    probs = rng.dirichlet(np.ones(43), len(df))
    plot_confusion_matrix(y_true, y_pred, tmp_path / "cm.png")
    plot_misclassified(df, y_true, y_pred, probs, tmp_path / "mis.png", n=4)
    for name in ("curves.png", "cm.png", "mis.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_compare_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "epochs_run": 5, "best_epoch": 3, "best_val_accuracy": 0.9,
            "train_seconds": 12.0, "params": 1000}))
    (a / "metrics.json").write_text(json.dumps({"accuracy": 0.95, "f1_macro": 0.9}))
    df = compare_runs([a, b])
    assert list(df["run"]) == ["a", "b"]
    assert df.loc[0, "test_accuracy"] == pytest.approx(0.95)
    assert np.isnan(df.loc[1, "test_accuracy"])


def test_load_history_adds_one_to_zero_based(tmp_path):
    zero = pd.DataFrame({"epoch": [0, 1, 2], "val_loss": [3.0, 2.0, 1.0]})
    zero.to_csv(tmp_path / "history.csv", index=False)
    assert list(load_history(tmp_path)["epoch"]) == [1, 2, 3]
    one = pd.DataFrame({"epoch": [1, 2], "val_loss": [3.0, 2.0]})
    one.to_csv(tmp_path / "history.csv", index=False)
    assert list(load_history(tmp_path)["epoch"]) == [1, 2]


def test_compare_runs_adds_crop_and_extra_metrics_rows(tmp_path):
    d = tmp_path / "baseline"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"epochs_run": 5, "best_epoch": 3}))
    (d / "metrics.json").write_text(json.dumps(
        {"accuracy": 0.99, "f1_macro": 0.9, "crop": True, "ms_per_image_batch": 0.1}))
    (d / "metrics_nocrop.json").write_text(json.dumps(
        {"accuracy": 0.8, "f1_macro": 0.7, "crop": False, "ms_per_image_batch": 0.2}))
    df = compare_runs([d])
    assert list(df["run"]) == ["baseline", "baseline (nedecupat)"]
    assert list(df["crop"]) == [True, False]
    assert df.loc[1, "test_accuracy"] == pytest.approx(0.8)
    assert df.loc[1, "ms_per_image_batch"] == pytest.approx(0.2)


def test_misclassified_without_errors_writes_png(fake_gtsrb, tmp_path):
    df = load_train_index(fake_gtsrb)
    y = df.class_id.to_numpy()
    probs = np.full((len(df), 43), 1 / 43)
    plot_misclassified(df, y, y, probs, tmp_path / "none.png")
    assert (tmp_path / "none.png").stat().st_size > 0


def test_misclassified_limits_two_per_pair(fake_gtsrb, tmp_path):
    from traffic_signs.evaluation import select_misclassified
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 1])
    conf = np.array([0.9, 0.8, 0.95, 0.7, 0.99])
    idx = select_misclassified(y_true, y_pred, conf, n=16, per_pair=2)
    assert list(idx) == [2, 0, 3]


def test_plot_history_marks_stage_boundary(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    history = pd.DataFrame({
        "epoch": [1, 2, 3, 4], "stage": [1, 1, 2, 2], "loss": [2.0, 1.0, 0.5, 0.4],
        "accuracy": [0.2, 0.5, 0.8, 0.9], "val_loss": [2.1, 1.2, 1.0, 0.9],
        "val_accuracy": [0.2, 0.4, 0.5, 0.6], "learning_rate": [1e-3] * 4})
    captured = []
    monkeypatch.setattr("traffic_signs.evaluation._save",
                        lambda fig, path: captured.append(fig))
    plot_history(history, tmp_path / "c.png", "t")
    fig = captured[0]
    for ax in fig.axes:
        xs = [l.get_xdata()[0] for l in ax.lines if isinstance(l, Line2D)
              and len(set(l.get_xdata())) == 1]
        assert 2.5 in xs
        assert "dezghețare conv5" in ax.get_legend_handles_labels()[1]
    plt.close(fig)
    # fara coloana `stage` -> fara linie de etapa
    plot_history(history.drop(columns="stage"), tmp_path / "d.png", "t")
    for ax in captured[1].axes:
        assert "dezghețare conv5" not in ax.get_legend_handles_labels()[1]
    plt.close(captured[1])
