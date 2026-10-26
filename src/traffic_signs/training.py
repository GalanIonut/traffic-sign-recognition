"""Bucla de antrenare: callback-uri, checkpoint si salvarea unui run."""
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

from traffic_signs.models import unfreeze_top


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


def train_two_stage(model: keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset,
                    run_dir: Path, stage1: dict, stage2: dict,
                    class_weight: dict[int, float] | None = None,
                    unfreeze: Callable[[keras.Model], int] = unfreeze_top,
                    run_config: dict | None = None) -> pd.DataFrame:
    """Fine-tuning in doua etape (cap, apoi blocul superior); scrie si agregatul in run_dir."""
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Directorul de run {run_dir} exista deja si nu e gol; alege alt --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)

    h1 = train_model(model, train_ds, val_ds, run_dir / "stage1",
                     class_weight=class_weight, **stage1)
    unfrozen = unfreeze(model)  # inainte de stage 2, care recompileaza modelul
    h2 = train_model(model, train_ds, val_ds, run_dir / "stage2",
                     class_weight=class_weight, **stage2)

    shutil.copy2(run_dir / "stage2" / "model.keras", run_dir / "model.keras")
    history = pd.concat([h1.assign(stage=1), h2.assign(stage=2)], ignore_index=True)
    history["epoch"] = range(1, len(history) + 1)
    history = history[["epoch", "stage", "loss", "accuracy", "val_loss", "val_accuracy",
                       "learning_rate"]]
    history.to_csv(run_dir / "history.csv", index=False)

    cfg1 = json.loads((run_dir / "stage1" / "config.json").read_text(encoding="utf-8"))
    cfg2 = json.loads((run_dir / "stage2" / "config.json").read_text(encoding="utf-8"))
    # Modelul salvat e cel din etapa 2: cea mai buna epoca se cauta doar acolo
    stage2_rows = history[history["stage"] == 2]
    best = stage2_rows.loc[stage2_rows["val_loss"].idxmin()]
    config = dict(run_config or {})
    config.update(
        epochs_run=len(history),
        best_epoch=int(best["epoch"]),
        best_val_loss=float(best["val_loss"]),
        best_val_accuracy=float(best["val_accuracy"]),
        train_seconds=round(cfg1["train_seconds"] + cfg2["train_seconds"], 1),
        params=int(model.count_params()),
        trainable_params=int(sum(int(w.numpy().size) for w in model.trainable_weights)),
        unfrozen_layers=int(unfrozen),
        stage1=stage1,
        stage2=stage2,
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    return history
