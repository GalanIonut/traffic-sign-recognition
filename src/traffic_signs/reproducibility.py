"""Reproductibilitate: seed global pentru Python, NumPy si backend-ul Keras."""
from tensorflow import keras

from traffic_signs.config import SEED


def set_global_seed(seed: int = SEED) -> None:
    """Fixeaza seed-urile random, NumPy si Keras/TensorFlow."""
    keras.utils.set_random_seed(seed)
