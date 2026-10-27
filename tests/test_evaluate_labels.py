"""Etichete specifice run-ului pentru curbele ResNet50 (scripts/evaluate.py)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_script", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage_label_conv4():
    assert _load().stage_label_for({"unfreeze": "conv4+conv5"}) == "dezghețare conv4+conv5"


def test_stage_label_default_conv5():
    mod = _load()
    assert mod.stage_label_for({}) == "dezghețare conv5"
    assert mod.stage_label_for({"unfreeze": "conv5"}) == "dezghețare conv5"


def test_display_names():
    mod = _load()
    assert mod.curves_title("resnet50_conv4") == "Curbe de antrenare: ResNet50 (conv4+conv5)"
    assert mod.curves_title("resnet50") == "Curbe de antrenare: ResNet50 (conv5)"
