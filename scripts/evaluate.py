"""Evalueaza un run pe setul de test; scrie rezultatele in runs/<nume> si <out>/<nume>.

<out> se alege cu --out (implicit reports/m4; pentru ResNet50 se foloseste reports/m5).
Cu --curves-only se regenereaza doar curbele de antrenare, fara reevaluare.
"""
import argparse
import json
import shutil
import sys
import time

import pandas as pd

from traffic_signs import config
from traffic_signs.dataset import load_test_index
from traffic_signs.evaluation import (compute_metrics, load_history, plot_confusion_matrix,
                                      plot_history, plot_misclassified,
                                      predict_classes)
from traffic_signs.models import load_trained_model
from traffic_signs.pipeline import make_dataset


def ms_per_image_batch(model, df, crop: bool, image_size: int, batch_size: int = 256) -> float:
    """Debit: ms pe imagine la batch 256; imaginile sunt incarcate dinainte, cu incalzire."""
    ds = make_dataset(df, image_size, batch_size=batch_size, crop=crop)
    images = [x.numpy() for x, _ in ds]
    model.predict(images[0], verbose=0)  # incalzire, netimata
    start = time.perf_counter()
    for x in images:
        model.predict(x, verbose=0)
    return (time.perf_counter() - start) / len(df) * 1000


def ms_per_image_single(model, df, crop: bool, image_size: int, count: int = 50) -> float:
    """Latenta: ms medii pe imagine la batch=1 (primele `count` imagini, dupa incalzire)."""
    ds = make_dataset(df.head(count), image_size, batch_size=1, crop=crop)
    images = [x.numpy() for x, _ in ds]
    model.predict(images[0], verbose=0)  # incalzire, netimata
    start = time.perf_counter()
    for x in images:
        model.predict(x, verbose=0)
    return (time.perf_counter() - start) / len(images) * 1000


# Nume afisabile pentru titlul curbelor; fallback pe id-ul run-ului
DISPLAY_NAMES = {"resnet50": "ResNet50 (conv5)", "resnet50_conv4": "ResNet50 (conv4+conv5)",
                 "baseline": "CNN propriu — de bază",
                 "no_reg": "CNN propriu — fără regularizare",
                 "no_crop": "CNN propriu — fără decupare"}


def curves_title(run_name: str) -> str:
    return f"Curbe de antrenare: {DISPLAY_NAMES.get(run_name, run_name)}"


def stage_label_for(cfg: dict) -> str:
    """Eticheta liniei de etapa, dupa cheia `unfreeze` din config (lipsa = conv5)."""
    return f"dezghețare {cfg.get('unfreeze', 'conv5')}"


def _stage_label(run_dir) -> str:
    path = run_dir / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return stage_label_for(cfg)


def write_curves(run_dir, out_dir, run_name: str) -> None:
    """Deseneaza curbele din history.csv in run_dir si le copiaza in out_dir."""
    plot_history(load_history(run_dir), run_dir / "curves.png", curves_title(run_name),
                 _stage_label(run_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "curves.png", out_dir / "curves.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--no-crop", action="store_true",
                        help="evalueaza pe imagini nedecupate")
    parser.add_argument("--out", default="reports/m4",
                        help="folder (relativ la radacina proiectului) pentru rezultate")
    parser.add_argument("--curves-only", action="store_true",
                        help="regenereaza doar curves.png (fara reevaluare)")
    args = parser.parse_args()

    run_dir = config.PROJECT_ROOT / "runs" / args.run_name
    if not (run_dir / "config.json").exists() or not (run_dir / "model.keras").exists():
        sys.exit(f"Eroare: rularea '{args.run_name}' este incompletă "
                 f"(lipsește config.json sau model.keras în {run_dir}). "
                 "Antrenarea a fost probabil întreruptă; rulați-o din nou.")
    if args.curves_only:
        write_curves(run_dir, config.PROJECT_ROOT / args.out / args.run_name, args.run_name)
        print(f"curbe salvate in {run_dir} si {config.PROJECT_ROOT / args.out / args.run_name}")
        return
    run_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    trained_crop = not run_cfg.get("no_crop", False)
    crop = trained_crop and not args.no_crop
    suffix = "_nocrop" if (trained_crop and args.no_crop) else ""

    model = load_trained_model(run_dir / "model.keras")
    test_df = load_test_index(config.DATA_DIR)
    image_size = model.input_shape[1]
    y_true, y_pred, probs = predict_classes(model, test_df, image_size, crop)
    metrics, per_class = compute_metrics(y_true, y_pred)
    metrics["crop"] = crop
    metrics["ms_per_image_batch"] = ms_per_image_batch(model, test_df, crop, image_size)
    metrics["ms_per_image_single"] = ms_per_image_single(model, test_df, crop, image_size)

    names = [f"metrics{suffix}.json", f"per_class{suffix}.csv",
             f"confusion_matrix{suffix}.png", f"misclassified{suffix}.png"]
    (run_dir / names[0]).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    per_class.to_csv(run_dir / names[1], index=False)
    plot_confusion_matrix(y_true, y_pred, run_dir / names[2])
    plot_misclassified(test_df, y_true, y_pred, probs, run_dir / names[3], crop=crop)
    if not suffix:  # curbele nu depind de decupare
        plot_history(load_history(run_dir), run_dir / "curves.png", curves_title(args.run_name),
                     _stage_label(run_dir))
        names.append("curves.png")

    out_dir = config.PROJECT_ROOT / args.out / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(run_dir / name, out_dir / name)
    if not suffix:  # config si istoric (1-based) pentru raport
        shutil.copy2(run_dir / "config.json", out_dir / "config.json")
        load_history(run_dir).to_csv(out_dir / "history.csv", index=False)
    print(f"acuratete test: {metrics['accuracy']:.4f}, F1 macro: {metrics['f1_macro']:.4f}, "
          f"{metrics['ms_per_image_batch']:.3f} ms/imagine (batch), "
          f"{metrics['ms_per_image_single']:.2f} ms/imagine (singular)")
    print(f"salvat in {run_dir} si {out_dir}")


if __name__ == "__main__":
    main()
