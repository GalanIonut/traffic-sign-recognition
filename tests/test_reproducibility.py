import numpy as np

from traffic_signs.reproducibility import set_global_seed


def test_set_global_seed_is_reproducible():
    set_global_seed()
    a = np.random.rand()
    set_global_seed()
    assert np.random.rand() == a
