"""Evalueaza un run pe setul de test; scrie rezultatele in runs/<nume> si reports/m4/<nume>."""
import argparse
import json
import shutil
import sys
import time

import keras
import pandas as pd

from traffic_signs import config
from traffic_signs.dataset import load_test_index
from traffic_signs.evaluation import (compute_metrics, plot_confusion_matrix,
                                      plot_history, plot_misclassified,
                                      predict_classes)
from traffic_signs.pipeline import make_dataset


def ms_per_image(model, df, crop: bool, batch_size: int = 256) -> float:
    """Timp de inferenta pe imagine (ms); imaginile sunt incarcate dinainte, cu o rulare de incalzire."""
    ds = make_dataset(df, config.IMG_SIZE_CNN, batch_size=batch_size, crop=crop)
    images = [x.numpy() for x, _ in ds]
    model.predict(images[0], verbose=0)  # incalzire, netimata
    start = time.perf_counter()
    for x in images:
        model.predict(x, verbose=0)
    return (time.perf_counter() - start) / len(df) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--no-crop", action="store_true",
                        help="evalueaza pe imagini nedecupate")
    args = parser.parse_args()

    run_dir = config.PROJECT_ROOT / "runs" / args.run_name
    if not (run_dir / "config.json").exists() or not (run_dir / "model.keras").exists():
        sys.exit(f"Eroare: rularea '{args.run_name}' este incompletă "
                 f"(lipsește config.json sau model.keras în {run_dir}). "
                 "Antrenarea a fost probabil întreruptă; rulați-o din nou.")
    run_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    trained_crop = not run_cfg.get("no_crop", False)
    crop = trained_crop and not args.no_crop
    suffix = "_nocrop" if (trained_crop and args.no_crop) else ""

    model = keras.models.load_model(run_dir / "model.keras")
    test_df = load_test_index(config.DATA_DIR)
    y_true, y_pred, probs = predict_classes(model, test_df, config.IMG_SIZE_CNN, crop)
    metrics, per_class = compute_metrics(y_true, y_pred)
    metrics["crop"] = crop
    metrics["ms_per_image"] = ms_per_image(model, test_df, crop)

    names = [f"metrics{suffix}.json", f"per_class{suffix}.csv",
             f"confusion_matrix{suffix}.png", f"misclassified{suffix}.png"]
    (run_dir / names[0]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    per_class.to_csv(run_dir / names[1], index=False)
    plot_confusion_matrix(y_true, y_pred, run_dir / names[2])
    plot_misclassified(test_df, y_true, y_pred, probs, run_dir / names[3], crop=crop)
    if not suffix:  # curbele nu depind de decupare
        history = pd.read_csv(run_dir / "history.csv")
        plot_history(history, run_dir / "curves.png", f"Curbe de antrenare: {args.run_name}")
        names.append("curves.png")

    out_dir = config.PROJECT_ROOT / "reports" / "m4" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(run_dir / name, out_dir / name)
    print(f"acuratete test: {metrics['accuracy']:.4f}, F1 macro: {metrics['f1_macro']:.4f}, "
          f"{metrics['ms_per_image']:.3f} ms/imagine")
    print(f"salvat in {run_dir} si {out_dir}")


if __name__ == "__main__":
    main()
