"""Augmentare Keras (fara oglindire) si ponderi pe clase."""
from collections import Counter
from collections.abc import Sequence

from tensorflow import keras

from traffic_signs.config import NUM_CLASSES, SEED


@keras.utils.register_keras_serializable(package="traffic_signs")
class ClipToUnitRange(keras.layers.Layer):
    """Limiteaza valorile la [0, 1]; identitate pentru intrari deja in interval."""

    def call(self, inputs):
        return keras.ops.clip(inputs, 0.0, 1.0)


def build_augmentation(seed: int = SEED) -> keras.Sequential:
    """Augmentari geometrice si de culoare; fara RandomFlip (semnele nu se oglindesc)."""
    return keras.Sequential(
        [
            keras.layers.RandomRotation(factor=15 / 360, fill_mode="nearest", seed=seed),
            keras.layers.RandomZoom(0.1, fill_mode="nearest", seed=seed + 1),
            keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest", seed=seed + 2),
            keras.layers.RandomBrightness(0.2, value_range=(0, 1), seed=seed + 3),
            keras.layers.RandomContrast(0.2, seed=seed + 4),
            ClipToUnitRange(),
        ],
        name="augmentation",
    )


def compute_class_weights(
    labels: Sequence[int], num_classes: int = NUM_CLASSES
) -> dict[int, float]:
    """Pondere = n_total / (n_clase_prezente * count_c); clasele absente lipsesc."""
    counts = Counter(int(label) for label in labels)
    n_total = sum(counts.values())
    n_present = len(counts)
    return {
        c: float(n_total / (n_present * counts[c]))
        for c in range(num_classes)
        if c in counts
    }
