"""Evaluare finala cap-coada (YOLO -> CNN) pe imaginile de test GTSDB 600-899.

Se ruleaza O SINGURA data pe test, dupa ce pipeline.json a fost fixat pe imaginile 0-599.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from traffic_signs import config  # noqa: E402
from traffic_signs.classes import CLASS_NAMES  # noqa: E402
from traffic_signs.detection import load_gt  # noqa: E402
from traffic_signs.models import load_trained_model  # noqa: E402
from traffic_signs.scene import (Counts, annotate, count, filter_by_conf, gt_for_image,  # noqa: E402
                                 load_pipeline_config, match, pick_examples, run_pipeline,
                                 sign_statuses, summarize)

TEST_RANGE = range(600, 900)

NOTE = (
    "Evaluare finală cap-coadă (YOLO apoi CNN) pe cele 300 de imagini de test GTSDB (600–899), "
    "folosită o singură dată. Un semn detectat se potrivește cu un semn real dacă IoU ≥ 0,5 "
    "(potrivire greedy, fiecare semn real cel mult o dată). "
    "detection_recall = ponderea semnelor reale găsite de detector; "
    "classification_accuracy = ponderea semnelor găsite al căror tip e corect; "
    "e2e_recall = semne găsite ȘI clasificate corect / semne reale; "
    "e2e_precision = semne găsite și clasificate corect / toate detecțiile păstrate; "
    "e2e_f1 = media armonică a lor. Semnele ratate de detector contează ca greșeli "
    "(scad e2e_recall), iar detecțiile fără semn real scad e2e_precision. "
    "Pragul de detecție, marja de decupaj și clasificatorul au fost alese pe imaginile 0–599 "
    "(pragul pe 500–599), fără să se folosească testul. "
    "detect_ms_mean = timpul mediu YOLO per imagine, classify_ms_mean = timpul mediu de decupare "
    "și clasificare CNN per imagine, ambele măsurate pe dispozitivul din câmpul device, "
    "după 2 apeluri de încălzire."
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=config.GTSDB_YOLO_DIR / "images" / "test")
    ap.add_argument("--gt", type=Path, default=config.GTSDB_DIR / "gt.txt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=config.M8_REPORTS_DIR)
    args = ap.parse_args()

    if not args.gt.exists():
        sys.exit(f"Lipsește fișierul de adnotări {args.gt}.")
    names = [f"{i:05d}" for i in TEST_RANGE]
    missing = [n for n in names if not (args.images_dir / f"{n}.jpg").exists()]
    if missing:
        sys.exit(f"Lipsesc {len(missing)} poze de test în {args.images_dir}.")
    cfg = load_pipeline_config()
    gt = load_gt(args.gt)

    from ultralytics import YOLO
    yolo = YOLO(str(config.YOLO_WEIGHTS))
    model = load_trained_model(config.MODELS_DIR / cfg.classifier)

    def load(n: str) -> np.ndarray:
        return np.array(Image.open(args.images_dir / f"{n}.jpg").convert("RGB"))

    for n in names[:2]:  # incalzire (nu intra in timpi)
        run_pipeline(yolo, model, load(n), cfg, device=args.device)

    total = Counts(0, 0, 0, 0)
    per_image, rows = {}, []
    det_ms, cls_ms = [], []
    for k, n in enumerate(names):
        img = load(n)
        signs, t = run_pipeline(yolo, model, img, cfg, device=args.device)
        signs = filter_by_conf(signs, cfg.conf)
        gts = gt_for_image(gt, n)
        total = total + count(signs, gts)
        det_ms.append(t["detect_ms"])
        cls_ms.append(t["classify_ms"])
        per_image[n] = (signs, gts, img)
        m = match(signs, gts)
        for si, gi in m.pairs:
            if signs[si].best.class_id != gts[gi].class_id:
                rows.append([n, "wrong", gts[gi].class_id, signs[si].best.class_id,
                             f"{signs[si].det_score:.4f}", f"{signs[si].best.confidence:.4f}"])
        for si in m.false_signs:
            rows.append([n, "false", "", signs[si].best.class_id,
                         f"{signs[si].det_score:.4f}", f"{signs[si].best.confidence:.4f}"])
        for gi in m.missed:
            rows.append([n, "missed", gts[gi].class_id, "", "", ""])
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(names)}")

    examples = pick_examples({n: (s, g) for n, (s, g, _) in per_image.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "split": "test (GTSDB 600-899)", "n_images": len(names), **vars(total),
        **summarize(total), "conf": cfg.conf, "classifier": cfg.classifier, "margin": cfg.margin,
        "detect_ms_mean": float(np.mean(det_ms)), "classify_ms_mean": float(np.mean(cls_ms)),
        "device": args.device, "example_ids": examples, "note": NOTE,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    with open(args.out / "errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "kind", "true_class", "pred_class", "det_score", "cnn_conf"])
        w.writerows(rows)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    for ax, n in zip(axes.ravel(), examples):
        signs, gts, img = per_image[n]
        m = match(signs, gts)
        st = sign_statuses(signs, gts, m)
        ok = sum(v == "correct" for v in st.values())
        ax.imshow(annotate(img, signs, st, [gts[i] for i in m.missed]))
        ax.set_title(f"{n} — {ok}/{len(gts)} corecte")
        ax.axis("off")
    for ax in axes.ravel()[len(examples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out / "examples.png", dpi=150)

    for k, v in metrics.items():
        if k != "note":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
