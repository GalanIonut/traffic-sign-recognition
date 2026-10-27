"""Regleaza lantul YOLO -> CNN pe GTSDB 0-599: clasificator, marja de decupaj, prag de detectie."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from traffic_signs import config
from traffic_signs.detection import load_gt
from traffic_signs.models import load_trained_model
from traffic_signs.scene import (DETECT_FLOOR, THRESHOLD_GRID, PipelineConfig, Counts,
                                 best_threshold, build_signs, choose_config, classify_crops,
                                 count, crop_with_margin, detect, filter_by_conf, gt_for_image,
                                 match, save_pipeline_config, summarize)

CLASSIFIERS = ("cnn_baseline.keras", "cnn_no_crop.keras")
MARGINS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
EXTRA_MARGINS = (0.40, 0.50)
TUNE_RANGE = range(0, 600)   # reglaj marja/clasificator: toate pozele 0-599 (CNN nu a vazut GTSDB)
THRESH_RANGE = range(500, 600)  # reglaj prag: imaginile de validare


def collect(images_dir: Path, ids: list[int]) -> dict[str, Path]:
    found = {}
    for i in ids:
        name = f"{i:05d}"
        for split in ("train", "val"):
            p = images_dir / split / f"{name}.jpg"
            if p.exists():
                found[name] = p
                break
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=config.GTSDB_YOLO_DIR / "images")
    ap.add_argument("--gt", type=Path, default=config.GTSDB_DIR / "gt.txt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, default=config.M8_REPORTS_DIR)
    args = ap.parse_args()

    if not args.gt.exists():
        sys.exit(f"Lipsește fișierul de adnotări {args.gt}.")
    paths = collect(args.images_dir, list(range(600)))
    if len(paths) < 600:
        sys.exit(f"Lipsesc pozele: găsite {len(paths)} din 600 în {args.images_dir}.")
    gt = load_gt(args.gt)

    from ultralytics import YOLO
    yolo = YOLO(str(config.YOLO_WEIGHTS))

    print("Detecție YOLO pe imaginile 0-599 ...")
    images, dets, gts = {}, {}, {}
    for k, (name, p) in enumerate(sorted(paths.items())):
        img = np.array(Image.open(p).convert("RGB"))
        images[name] = img
        dets[name] = detect(yolo, img, conf=DETECT_FLOOR, device=args.device)
        gts[name] = gt_for_image(gt, name)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/600")

    models = {c: load_trained_model(config.MODELS_DIR / c) for c in CLASSIFIERS}

    def classify_image(model, name: str, margin: float):
        crops = [crop_with_margin(images[name], d.box, margin) for d in dets[name]]
        return build_signs(dets[name], classify_crops(model, crops, config.IMG_SIZE_CNN))

    def tune(margins) -> list[dict]:
        rows = []
        for c in CLASSIFIERS:
            for m in margins:
                n_matched = n_ok = 0
                for name in (f"{i:05d}" for i in TUNE_RANGE):
                    signs = classify_image(models[c], name, m)
                    mr = match(signs, gts[name])
                    n_matched += len(mr.pairs)
                    n_ok += sum(signs[si].best.class_id == gts[name][gi].class_id
                                for si, gi in mr.pairs)
                acc = n_ok / n_matched if n_matched else 0.0
                rows.append({"classifier": c, "margin": m, "n_matched": n_matched,
                             "accuracy": acc})
                print(f"  {c} marja={m:.2f} potrivite={n_matched} acuratete={acc:.4f}")
        return rows

    print("Reglaj clasificator x marjă (imaginile 0-599) ...")
    rows = tune(MARGINS)
    chosen = choose_config(rows)
    if chosen["margin"] == max(MARGINS):
        print("Marja aleasă e 0.30; extind grila cu 0.40 și 0.50 ...")
        rows += tune(EXTRA_MARGINS)
        chosen = choose_config(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "tuning.csv", index=False)

    print("Reglaj prag de detecție (imaginile 500-599) ...")
    model = models[chosen["classifier"]]
    per_image = []
    for i in THRESH_RANGE:
        name = f"{i:05d}"
        per_image.append((classify_image(model, name, chosen["margin"]), gts[name]))
    trows = []
    for t in THRESHOLD_GRID:
        total = Counts(0, 0, 0, 0)
        for signs, g in per_image:
            total = total + count(filter_by_conf(signs, t), g)
        s = summarize(total)
        trows.append({"conf": t, **{k: s[k] for k in (
            "e2e_precision", "e2e_recall", "e2e_f1", "detection_recall",
            "classification_accuracy")}})
    pd.DataFrame(trows).to_csv(args.out / "thresholds.csv", index=False)
    conf = best_threshold(per_image)

    cfg = PipelineConfig(chosen["classifier"], config.IMG_SIZE_CNN, chosen["margin"], conf)
    save_pipeline_config(cfg, config.PIPELINE_CONFIG)
    f1 = next(r["e2e_f1"] for r in trows if r["conf"] == conf)
    print(f"Clasificator ales: {cfg.classifier}")
    print(f"Marjă de decupaj: {cfg.margin}")
    print(f"Prag de detecție: {cfg.conf}")
    print(f"Acuratețe de reglaj: {chosen['accuracy']:.4f} · F1 cap-coadă (500-599): {f1:.4f}")
    print(f"Salvat în {config.PIPELINE_CONFIG}, {args.out / 'tuning.csv'}, {args.out / 'thresholds.csv'}")


if __name__ == "__main__":
    main()
