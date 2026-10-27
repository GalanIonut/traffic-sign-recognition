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
