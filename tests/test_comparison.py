"""Teste pentru comparația CNN vs. ResNet50 și exportul figurilor M4 (date false)."""
import importlib.util
import json
import shutil

import pandas as pd
import pytest

from traffic_signs import config
from traffic_signs.comparison import COLUMNS, model_comparison_table, plot_cnn_vs_resnet

ROOT = config.PROJECT_ROOT


def _fake_run(base, name, *, size_bytes, acc, f1, params, trainable, seconds, epochs,
              ms_batch, ms_single, with_trainable=True):
    run_dir = base / "runs" / name
    metrics_dir = base / "reports" / name
    run_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    cfg = {"params": params, "train_seconds": seconds, "epochs_run": epochs}
    if with_trainable:
        cfg["trainable_params"] = trainable
    (run_dir / "config.json").write_text(json.dumps(cfg))
    (run_dir / "model.keras").write_bytes(b"x" * size_bytes)
    (metrics_dir / "metrics.json").write_text(json.dumps(
        {"accuracy": acc, "f1_macro": f1, "ms_per_image_batch": ms_batch,
         "ms_per_image_single": ms_single}))
    return {"label": name, "run_dir": run_dir, "metrics_dir": metrics_dir}


@pytest.fixture
def entries(tmp_path):
    cnn = _fake_run(tmp_path, "cnn", size_bytes=2 * 1024 * 1024, acc=0.99, f1=0.98,
                    params=1_000_000, trainable=1_000_000, seconds=600.0, epochs=30,
                    ms_batch=0.3, ms_single=13.0, with_trainable=False)
    res = _fake_run(tmp_path, "resnet", size_bytes=36 * 1024 * 1024, acc=0.995, f1=0.99,
                    params=23_000_000, trainable=9_000_000, seconds=7200.0, epochs=12,
                    ms_batch=2.4, ms_single=40.0)
    return [cnn, res]


def test_comparison_table(entries):
    t = model_comparison_table(entries)
    assert list(t.columns) == COLUMNS
    cnn, res = t.iloc[0], t.iloc[1]
    assert cnn["model"] == "cnn" and res["model"] == "resnet"
    assert cnn["model_mb"] == pytest.approx(2.0) and res["model_mb"] == pytest.approx(36.0)
    assert cnn["train_minutes"] == pytest.approx(10.0)
    assert res["train_minutes"] == pytest.approx(120.0)
    assert cnn["test_accuracy"] == 0.99 and res["f1_macro"] == 0.99
    assert cnn["params"] == 1_000_000
    assert cnn["trainable_params"] == 1_000_000  # lipsește din config -> = params
    assert res["trainable_params"] == 9_000_000
    assert res["epochs_run"] == 12
    assert res["ms_per_image_batch"] == 2.4 and res["ms_per_image_single"] == 40.0


@pytest.mark.parametrize("res_acc", [0.995, 0.97])  # câștigă ResNet / câștigă CNN
def test_plot_writes_png(entries, tmp_path, res_acc):
    t = model_comparison_table(entries)
    t.loc[1, "test_accuracy"] = res_acc
    out = tmp_path / "out" / "fig.png"
    plot_cnn_vs_resnet(t, out)
    assert out.exists() and out.stat().st_size > 5000


def _load_script():
    spec = importlib.util.spec_from_file_location("compare_runs_script",
                                                  ROOT / "scripts" / "compare_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _populate(out_dir):
    for run, src, _ in _load_script().EXPORTS:
        (out_dir / run).mkdir(parents=True, exist_ok=True)
        (out_dir / run / src).write_bytes(b"png")


def test_compare_runs_does_not_export_for_other_out(tmp_path, monkeypatch):
    figures = tmp_path / "figures"
    figures.mkdir()
    monkeypatch.setattr(config, "FIGURES_DIR", figures)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    other = tmp_path / "reports" / "m5"
    _populate(other)
    assert _load_script().export_m4_figures(other) == []
    assert list(figures.iterdir()) == []


def test_compare_runs_exports_for_m4(tmp_path, monkeypatch):
    figures = tmp_path / "figures"
    figures.mkdir()
    monkeypatch.setattr(config, "FIGURES_DIR", figures)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    m4 = tmp_path / "reports" / "m4"
    _populate(m4)
    copied = _load_script().export_m4_figures(m4)
    assert sorted(p.name for p in figures.iterdir()) == sorted(
        ["m4_curves_baseline.png", "m4_curves_no_reg.png",
         "m4_confusion_baseline.png", "m4_misclassified_baseline.png"])
    assert len(copied) == 4


def test_title_is_truthful_both_ways(entries):
    from traffic_signs.comparison import comparison_title
    t = model_comparison_table(entries)
    win = comparison_title(t)
    assert "ResNet50 câștigă 0,5 pp" in win and "de 18× mai mare" in win
    assert "de 12× mai lent la antrenare" in win
    t.loc[1, "test_accuracy"] = 0.98
    lose = comparison_title(t)
    assert lose.startswith("CNN-ul propriu câștigă 1,0 pp") and "mai mare" in lose
