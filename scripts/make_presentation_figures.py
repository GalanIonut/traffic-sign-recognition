"""Generează cele 7 grafice intuitive pentru prezentare în reports/figures/pres_*.png."""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from traffic_signs.classes import class_name
from traffic_signs.config import DATA_DIR, FIGURES_DIR, IMG_SIZE_CNN, PROJECT_ROOT
from traffic_signs.dataset import load_test_index, load_train_index
from traffic_signs.evaluation import load_history, predict_classes
from traffic_signs.preprocessing import crop_roi, load_image, prepare
from traffic_signs import presentation as pres

import pandas as pd

REPORTS = PROJECT_ROOT / "reports" / "m4"
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_baseline.keras"
# 12 tipuri variate de semne pentru figura cu setul de date
SHOWCASE = [14, 13, 17, 27, 28, 12, 33, 40, 25, 18, 3, 38]


def meta_image(class_id: int) -> np.ndarray:
    return load_image(DATA_DIR / "Meta" / f"{class_id}.png")


def roi_of(row) -> tuple[int, int, int, int]:
    return int(row.roi_x1), int(row.roi_y1), int(row.roi_x2), int(row.roi_y2)


def main(out_dir: Path = FIGURES_DIR) -> None:
    train = load_train_index(DATA_DIR)
    test = load_test_index(DATA_DIR)
    model = tf.keras.models.load_model(MODEL_PATH)
    y_true, y_pred, probs = predict_classes(model, test, IMG_SIZE_CNN, crop=True)
    print(f"acuratete test: {(y_true == y_pred).mean():.4f}")

    # 1. Pipeline: un exemplu bine clasificat (Stop, imagine mare, încredere mare)
    conf = probs.max(axis=1)
    ok = np.flatnonzero((y_true == 14) & (y_pred == 14) & (conf > 0.99) & (test["width"].to_numpy() >= 60))
    row = test.iloc[int(ok[0])]
    full = load_image(row["path"])
    top3 = [(class_name(c), p) for c, p in pres.top_k(probs[ok[0]], 3)]
    pres.plot_pipeline(full, crop_roi(full, *roi_of(row)), prepare(row["path"], IMG_SIZE_CNN, roi_of(row)),
                       top3, class_name(14), int(model.count_params()), out_dir / "pres_pipeline.png")

    # 2. Predicții: 6 corecte + 2 greșite
    idx = pres.select_examples(y_true, y_pred, probs, 6, 2, seed=42)
    items = []
    for i in idx:
        r = test.iloc[i]
        items.append({"image": prepare(r["path"], IMG_SIZE_CNN, roi_of(r)),
                      "top3": [(pres.short_name(c), p) for c, p in pres.top_k(probs[i], 3)],
                      "correct": bool(y_true[i] == y_pred[i]), "true_name": pres.short_name(int(y_true[i]))})
    wrong_conf = [float(conf[i]) for i in idx if y_true[i] != y_pred[i]]
    print("incredere pe cele 2 greșite:", wrong_conf,
          "| medie corecte:", conf[y_true == y_pred].mean(), "| medie greșite:", conf[y_true != y_pred].mean())
    pres.plot_predictions(
        items, out_dir / "pres_predictions.png",
        "Când are dreptate, modelul este aproape sigur pe răspuns",
        "Opt exemple din setul de test: probabilitățile celor mai probabile 3 semne")

    # 3. Acuratețea pe clasă
    per_class = pd.read_csv(REPORTS / "baseline" / "per_class.csv")
    metas = {c: meta_image(c) for c in range(43)}
    pres.plot_per_class(per_class, metas, out_dir / "pres_per_class.png")

    # 4. Experimente
    comp = pd.read_csv(REPORTS / "comparison.csv").set_index("run")
    rows = [("De bază — semne decupate", comp.loc["baseline", "test_accuracy"], True),
            ("De bază — imagini nedecupate", comp.loc["baseline (nedecupat)", "test_accuracy"], False),
            ("Fără regularizare — semne decupate", comp.loc["no_reg", "test_accuracy"], True),
            ("Fără decupare — imagini nedecupate", comp.loc["no_crop", "test_accuracy"], False)]
    pres.plot_experiments(rows, out_dir / "pres_experiments.png")

    # 5. Supraînvățare
    pres.plot_overfitting(load_history(REPORTS / "baseline"), load_history(REPORTS / "no_reg"),
                          out_dir / "pres_overfitting.png")

    # 6. Confuzii
    pairs = pres.top_confused_pairs(y_true, y_pred, 5)
    print("perechi confundate (real, prezis, n, fracție):", pairs)
    names = {c: class_name(c) for c in range(43)}
    pres.plot_confusions(pairs, metas, names, out_dir / "pres_confusions.png",
                         "Modelul confundă semne care arată aproape la fel",
                         "Cele mai frecvente 5 confuzii pe setul de test")

    # 7. Setul de date
    sizes = pd.concat([train, test])
    stats = {"n_train": len(train), "n_test": len(test), "n_classes": int(train["class_id"].nunique()),
             "avg_px": float(np.mean([sizes["width"].mean(), sizes["height"].mean()]))}
    print("statistici:", stats)
    pres.plot_dataset(stats, [metas[c] for c in SHOWCASE], out_dir / "pres_dataset.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else FIGURES_DIR)
