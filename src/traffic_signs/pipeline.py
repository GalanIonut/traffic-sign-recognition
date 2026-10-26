"""Pipeline tf.data: citire, decupare ROI, redimensionare, augmentare."""
import numpy as np
import pandas as pd
import tensorflow as tf

from traffic_signs.augmentation import build_augmentation
from traffic_signs.config import BATCH_SIZE, NUM_CLASSES, SEED
from traffic_signs.preprocessing import prepare


def make_dataset(df: pd.DataFrame, image_size: int, batch_size: int = BATCH_SIZE,
                 training: bool = False, crop: bool = True,
                 seed: int = SEED) -> tf.data.Dataset:
    """Dataset (imagini float32 in [0,1], etichete one-hot); ordine din df daca nu antrenam."""

    def load(path, x1, y1, x2, y2):
        # Ruleaza in NumPy/Pillow; calea ajunge ca bytes
        roi = (int(x1), int(y1), int(x2), int(y2)) if crop else None
        return prepare(path, image_size, roi)

    def load_tf(path, x1, y1, x2, y2, label):
        img = tf.numpy_function(load, [path, x1, y1, x2, y2], tf.float32)
        img.set_shape((image_size, image_size, 3))
        return img, tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)

    if len(df) == 0:
        raise ValueError("DataFrame-ul este gol")
    ds = tf.data.Dataset.from_tensor_slices((
        df["path"].astype(str).to_numpy(),
        df["roi_x1"].to_numpy(np.int32), df["roi_y1"].to_numpy(np.int32),
        df["roi_x2"].to_numpy(np.int32), df["roi_y2"].to_numpy(np.int32),
        df["class_id"].to_numpy(np.int32),
    ))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load_tf, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size)
    if training:
        augment = build_augmentation(seed)  # o singura data pe dataset
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
