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
    assert cnn["file_mb"] == pytest.approx(2.0) and res["file_mb"] == pytest.approx(36.0)
    assert cnn["weights_mb"] == pytest.approx(1_000_000 * 4 / 2**20)
    assert res["weights_mb"] == pytest.approx(23_000_000 * 4 / 2**20)
    assert "model_mb" not in t.columns
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
    assert "ResNet50 câștigă 0,5 pp" in win and "de 23× mai mare" in win
    assert "de 12× mai lent la antrenare" in win
    t.loc[1, "test_accuracy"] = 0.98
    lose = comparison_title(t)
    assert lose.startswith("CNN-ul propriu câștigă 1,0 pp") and "mai mare" in lose


def _row(model, acc, mb, minutes):
    return {"model": model, "test_accuracy": acc, "weights_mb": mb, "train_minutes": minutes}


def _tbl(res_acc, res_mb, res_min):
    return pd.DataFrame([_row("cnn", 0.99, 10.0, 10.0), _row("resnet", res_acc, res_mb, res_min)])


def test_title_combinations():
    from traffic_signs.comparison import comparison_title
    # ResNet câștigă, dar mai mare și mai lent
    assert comparison_title(_tbl(0.995, 100.0, 40.0)) == (
        "ResNet50 câștigă 0,5 pp acuratețe, dar e de 10× mai mare și de 4,0× mai lent la antrenare")
    # ResNet câștigă și e mai mic și mai rapid -> fără „dar”
    t = comparison_title(_tbl(0.995, 5.0, 5.0))
    assert ", și e de 2,0× mai mic și de 2,0× mai rapid la antrenare" in t and "dar" not in t
    # mixt: mai mic, dar mai lent
    t = comparison_title(_tbl(0.995, 5.0, 40.0))
    assert ", dar e de 2,0× mai mic și de 4,0× mai lent la antrenare" in t
    # aceeași mărime
    t = comparison_title(_tbl(0.995, 10.0, 40.0))
    assert "are aceeași mărime, iar e de 4,0× mai lent" in t
    # CNN câștigă
    t = comparison_title(_tbl(0.98, 100.0, 40.0))
    assert t.startswith("CNN-ul propriu câștigă 1,0 pp acuratețe; ResNet50 e de 10× mai mare")
    # egalitate
    assert comparison_title(_tbl(0.99, 100.0, 40.0)).startswith(
        "ResNet50 și CNN-ul au aceeași acuratețe;")
    # diferență foarte mică: 2 zecimale
    assert "câștigă 0,05 pp" in comparison_title(_tbl(0.9905, 100.0, 40.0))


def test_rows_selected_by_label(entries, tmp_path):
    from traffic_signs.comparison import comparison_title
    t = model_comparison_table(entries)
    swapped = t.iloc[::-1].reset_index(drop=True)
    assert comparison_title(t, "cnn", "resnet") == comparison_title(swapped, "cnn", "resnet")
    plot_cnn_vs_resnet(swapped, tmp_path / "s.png", "cnn", "resnet")
    assert (tmp_path / "s.png").exists()
    with pytest.raises(ValueError, match="lipsește"):
        comparison_title(t, "cnn", "nope")
    with pytest.raises(ValueError, match="lipsește"):
        plot_cnn_vs_resnet(t, tmp_path / "x.png", "nope", "resnet")


def test_default_figure_path_and_relative_out(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    mod = _load_script()
    assert mod.default_figure_path("reports/m4").name == "m4_val_accuracy_comparison.png"
    assert mod.default_figure_path("reports/m5").name == "m5_val_accuracy_comparison.png"
    (tmp_path / "figures").mkdir()
    _populate(tmp_path / "reports" / "m4")
    assert len(mod.export_m4_figures("reports/m4")) == 4  # cale relativa
    assert mod.export_m4_figures("reports/m5") == []


def test_title_fits_in_figure(entries):
    import matplotlib.pyplot as plt
    from traffic_signs.comparison import build_comparison_figure
    t = model_comparison_table(entries)
    t.loc[1, "test_accuracy"] = 0.97  # varianta cu titlul cel mai lung
    fig = build_comparison_figure(t)
    fig.canvas.draw()
    box = fig.title_text.get_window_extent()
    assert "\n" in fig.title_text.get_text()
    assert fig.bbox.x0 <= box.x0 and box.x1 <= fig.bbox.x1
    assert fig.bbox.y0 <= box.y0 and box.y1 <= fig.bbox.y1
    sub = [x for x in fig.texts if x is not fig.title_text][0].get_window_extent()
    assert box.y0 >= sub.y1  # nu se suprapune cu subtitlul
    plt.close(fig)
