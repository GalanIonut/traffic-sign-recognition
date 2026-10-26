"""Regresie: incarcarea unui model salvat intr-un proces proaspat (fara import explicit al modelelor)."""
import subprocess
import sys

from traffic_signs.models import build_resnet50

SNIPPET = """
import sys
{load}
m = load({path!r})
print(m.name)
"""


def _run(load: str, path) -> subprocess.CompletedProcess:
    code = SNIPPET.format(load=load, path=str(path))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_plain_keras_load_fails_in_fresh_process(tmp_path):
    """Documenteaza bug-ul: fara import, ResNetPreprocess nu e inregistrat."""
    path = tmp_path / "m.keras"
    build_resnet50(input_size=32, weights=None).save(path)
    r = _run("from keras.models import load_model as load", path)
    assert r.returncode != 0
    assert "ResNetPreprocess" in r.stderr


def test_load_trained_model_works_in_fresh_process(tmp_path):
    path = tmp_path / "m.keras"
    build_resnet50(input_size=32, weights=None).save(path)
    r = _run("from traffic_signs.models import load_trained_model as load", path)
    assert r.returncode == 0, r.stderr[-800:]
    assert "resnet50_finetune" in r.stdout
