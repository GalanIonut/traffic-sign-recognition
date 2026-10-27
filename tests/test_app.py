"""Teste pentru interfata Streamlit (AppTest ruleaza scriptul in acelasi proces)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from traffic_signs import config

APP = str(Path(config.PROJECT_ROOT) / "app" / "streamlit_app.py")
CNN_FILE = "cnn_no_crop.keras"


@pytest.fixture
def cnn_only_dir(tmp_path, monkeypatch):
    """MODELS_DIR cu doar CNN (symlink): testele nu incarca ResNet-ul de 260 MB."""
    (tmp_path / CNN_FILE).symlink_to(Path(config.PROJECT_ROOT) / "models" / CNN_FILE)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    return tmp_path


def _run(**_):
    return AppTest.from_file(APP, default_timeout=120).run()


def test_app_starts_without_errors(cnn_only_dir):
    at = _run()
    assert not at.exception
    assert "Recunoașterea semnelor" in at.title[0].value
    assert "CNN propriu" in at.radio[0].options


def test_app_classifies_stop_sample(cnn_only_dir):
    at = _run()
    at.selectbox[0].select("Stop").run()
    assert not at.exception
    assert any("Stop" in h.value for h in at.subheader)
    assert any(m.label == "Încredere" for m in at.metric)


def test_app_without_resnet_shows_info(cnn_only_dir):
    at = _run()
    assert len(at.radio[0].options) == 1
    assert any("promote_model.py" in i.value for i in at.info)


def test_app_without_models_stops_with_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    at = _run()
    assert not at.exception
    assert len(at.error) == 1
    assert len(at.radio) == 0


def test_app_corrupt_model_shows_friendly_error(tmp_path, monkeypatch):
    (tmp_path / CNN_FILE).write_bytes(b"")  # fisier gol, nu se poate incarca
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    at = _run()
    at.selectbox[0].select("Stop").run()
    assert not at.exception
    assert any("nu a putut fi încărcat" in e.value for e in at.error)


# --- tabul „Scenă de stradă” ---

SCENE_SELECT = "scene_sample"
TRUTH_TOGGLE = "scene_truth"


@pytest.fixture
def scene_dir(tmp_path, monkeypatch):
    """MODELS_DIR cu CNN-ul tabului 1, clasificatorul din pipeline.json si YOLO (symlink-uri)."""
    import shutil

    from traffic_signs.scene import load_pipeline_config

    models = Path(config.PROJECT_ROOT) / "models"
    cfg = load_pipeline_config(models / "pipeline.json")
    for name in {CNN_FILE, cfg.classifier, "yolo_signs.pt"}:
        (tmp_path / name).symlink_to(models / name)
    shutil.copy(models / "pipeline.json", tmp_path / "pipeline.json")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "YOLO_WEIGHTS", tmp_path / "yolo_signs.pt")
    monkeypatch.setattr(config, "PIPELINE_CONFIG", tmp_path / "pipeline.json")
    return tmp_path


def _first_scene(at):
    box = at.selectbox(key=SCENE_SELECT)
    return box.select(box.options[1]).run()


def test_app_has_two_tabs(scene_dir):
    at = _run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Un semn", "Scenă de stradă"]


def test_scene_tab_runs_on_sample(scene_dir):
    at = _first_scene(_run())
    assert not at.exception
    scene_tab = at.tabs[1]
    assert len(scene_tab.get("image")) >= 1
    texts = [m.value for m in scene_tab.markdown] + [h.value for h in scene_tab.subheader]
    assert any(t.lstrip("*#  ").startswith("1. ") for t in texts)
    captions = [c.value for c in scene_tab.caption]
    assert any("detecție" in c and "ms" in c for c in captions)


def test_scene_truth_toggle_only_for_samples(scene_dir):
    at = _run()
    assert not [t for t in at.toggle if t.key == TRUTH_TOGGLE]
    at = _first_scene(at)
    toggle = at.toggle(key=TRUTH_TOGGLE)
    assert toggle.label == "Arată răspunsul corect"
    at = toggle.set_value(True).run()
    assert not at.exception


def test_scene_tab_without_yolo_shows_info(scene_dir, monkeypatch):
    monkeypatch.setattr(config, "YOLO_WEIGHTS", scene_dir / "lipsa.pt")
    at = _run()
    assert not at.exception
    assert any("detector" in i.value.lower() for i in at.tabs[1].info)
    at.selectbox[0].select("Stop").run()
    assert not at.exception
    assert any("Stop" in h.value for h in at.subheader)


def test_scene_tab_keeps_pil_open_unpatched(scene_dir):
    """ultralytics inlocuieste PIL.Image.open (incearca `pip install pi-heif` la fisiere invalide);
    aplicatia il restaureaza, ca o imagine invalida sa dea tot mesajul prietenos."""
    from PIL import Image

    at = _first_scene(_run())
    assert not at.exception
    assert Image.open.__module__ == "PIL.Image"
