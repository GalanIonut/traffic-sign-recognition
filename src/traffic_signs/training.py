"""Bucla de antrenare: callback-uri, checkpoint si salvarea unui run."""
import json
import time
from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf


class _LogLearningRate(keras.callbacks.Callback):
    """Adauga `learning_rate` in logs daca Keras nu l-a pus deja."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and "learning_rate" not in logs:
            logs["learning_rate"] = float(self.model.optimizer.learning_rate)


def train_model(model: keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset,
                run_dir: Path, epochs: int = 30, learning_rate: float = 1e-3,
                class_weight: dict[int, float] | None = None, patience: int = 5,
                run_config: dict | None = None) -> pd.DataFrame:
    """Antreneaza modelul si salveaza model.keras, history.csv, config.json in run_dir."""
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Directorul de run {run_dir} exista deja si nu e gol; alege alt --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        _LogLearningRate(),
        keras.callbacks.ModelCheckpoint(run_dir / "model.keras", monitor="val_loss",
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                          min_lr=1e-5),
        keras.callbacks.CSVLogger(run_dir / "history.csv"),
    ]
    start = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     class_weight=class_weight, callbacks=callbacks, shuffle=False, verbose=2)
    seconds = time.perf_counter() - start

    history = pd.DataFrame(hist.history)
    history.insert(0, "epoch", range(1, len(history) + 1))
    history = history[["epoch", "loss", "accuracy", "val_loss", "val_accuracy",
                       "learning_rate"]]
    history.to_csv(run_dir / "history.csv", index=False)  # epoci 1-based (CSVLogger e 0-based)
    best = int(history["val_loss"].idxmin())
    config = dict(run_config or {})
    config.update(
        epochs_run=len(history),
        best_epoch=int(history.loc[best, "epoch"]),
        best_val_loss=float(history.loc[best, "val_loss"]),
        best_val_accuracy=float(history.loc[best, "val_accuracy"]),
        train_seconds=round(seconds, 1),
        params=int(model.count_params()),
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    return history
