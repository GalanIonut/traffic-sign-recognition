import pytest

from traffic_signs.classes import CLASS_NAMES, class_name
from traffic_signs.config import NUM_CLASSES


def test_has_43_classes():
    assert sorted(CLASS_NAMES) == list(range(NUM_CLASSES))


def test_names_unique():
    assert len(set(CLASS_NAMES.values())) == NUM_CLASSES


def test_known_names():
    assert class_name(14) == "Stop"
    assert class_name(13) == "Cedează trecerea"


@pytest.mark.parametrize("bad_id", [-1, NUM_CLASSES])
def test_invalid_id_raises(bad_id):
    with pytest.raises(ValueError):
        class_name(bad_id)
