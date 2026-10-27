"""Evalueaza o rulare YOLO: metrici, curbe si exemple cu patratele prezise vs reale."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from traffic_signs import config
from traffic_signs.detection import read_yolo_boxes, summarize_metrics

GREEN = "#2ca02c"
ORANGE = "#eb6834"


def draw_examples(model, split: str, conf: float, imgsz: int, device: str, out_path: Path) -> None:
    images = sorted((config.GTSDB_YOLO_DIR / "images" / split).glob("*.jpg"))
    chosen = random.Random(config.SEED).sample(images, 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 7.6))
    for ax, img_path in zip(axes.ravel(), chosen):
        img = Image.open(img_path).convert("RGB")
        pred = model.predict(str(img_path), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        ax.imshow(img)
        gt_boxes = read_yolo_boxes(
            config.GTSDB_YOLO_DIR / "labels" / split / f"{img_path.stem}.txt", *img.size)
        n_gt = len(gt_boxes)
        for x1, y1, x2, y2 in gt_boxes:
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor=ORANGE, linewidth=2.5, linestyle="--"))
        for x1, y1, x2, y2 in pred.boxes.xyxy.cpu().numpy():
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor=GREEN, linewidth=1.5, linestyle="--"))
        ax.set_title(f"{img_path.name} · real: {n_gt} · găsite: {len(pred.boxes)}", fontsize=11)
        ax.axis("off")
    handles = [patches.Patch(edgecolor=GREEN, fill=False, linestyle="--", label="Prezis (model)"),
               patches.Patch(edgecolor=ORANGE, fill=False, linestyle="--", label="Real (etichetă)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=13, frameon=False)
    fig.suptitle(f"Detecția semnelor: exemple din setul de {split}", fontsize=17,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=2.5)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def time_single_image(model, split: str, imgsz: int, conf: float, device: str,
                      n_images: int = 20, n_warmup: int = 3) -> float:
    """Timp mediu (ms) pentru model.predict pe o imagine, incl. citire, preprocesare si NMS."""
    images = sorted((config.GTSDB_YOLO_DIR / "images" / split).glob("*.jpg"))[:n_images]
    for p in images[:n_warmup]:
        model.predict(str(p), imgsz=imgsz, conf=conf, device=device, verbose=False)
    times = []
    for p in images:
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=imgsz, conf=conf, device=device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


TIMING_NOTE = (
    "inference_ms_per_image_batched = doar propagarea prin rețea, medie pe imagine în validarea "
    "Ultralytics (loturi, fără citirea imaginii); preprocess_ms și postprocess_ms = redimensionare "
    "și, respectiv, NMS, în aceeași validare; end_to_end_ms_single_image = timp total mediu "
    "(wall-clock) pentru model.predict pe o singură imagine (citire, preprocesare, rețea, NMS) pe 20 de "
    "imagini de test, după 3 apeluri de încălzire."
)
CONF_NOTE = (
    "Precizia și recall-ul sunt calculate de Ultralytics la pragul de încredere care maximizează "
    "F1; exemplele din examples.png folosesc pragul conf 0.25."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--promote", action="store_true",
                        help="copiaza best.pt in models/yolo_signs.pt")
    args = parser.parse_args()

    run_dir = config.PROJECT_ROOT / "runs" / "yolo" / args.run_name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Lipseste {best}. Antreneaza intai cu scripts/train_yolo.py")
    out_dir = config.PROJECT_ROOT / "reports" / "m7" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("BoxPR_curve.png", "PR_curve.png", "test_PR_curve.png", "test_F1_curve.png"):
        (out_dir / stale).unlink(missing_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(best))
    results = model.val(data=str(config.GTSDB_YOLO_DIR / "dataset.yaml"), split=args.split,
                        imgsz=args.imgsz, device=args.device, conf=0.001,
                        project=str(out_dir / "_val"), name="val", exist_ok=True, plots=True)
    metrics = summarize_metrics(results)
    metrics.update(
        split=args.split, imgsz=args.imgsz,
        inference_ms_per_image_batched=float(results.speed["inference"]),
        preprocess_ms=float(results.speed["preprocess"]),
        postprocess_ms=float(results.speed["postprocess"]),
        end_to_end_ms_single_image=time_single_image(
            model, args.split, args.imgsz, args.conf, args.device),
        note=TIMING_NOTE, conf_note=CONF_NOTE)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Curbele de test vin doar din folderul evaluarii; din rularea de antrenare doar results.png
    val_dir = Path(results.save_dir)
    for src_name, dst_name in (("BoxPR_curve.png", "test_PR_curve.png"),
                               ("PR_curve.png", "test_PR_curve.png"),
                               ("BoxF1_curve.png", "test_F1_curve.png"),
                               ("F1_curve.png", "test_F1_curve.png")):
        if (val_dir / src_name).exists() and not (out_dir / dst_name).exists():
            shutil.copy2(val_dir / src_name, out_dir / dst_name)
    if (run_dir / "results.png").exists():
        shutil.copy2(run_dir / "results.png", out_dir / "results.png")
    shutil.rmtree(out_dir / "_val", ignore_errors=True)

    draw_examples(model, args.split, args.conf, args.imgsz, args.device, out_dir / "examples.png")
    if args.promote:
        shutil.copy2(best, config.MODELS_DIR / "yolo_signs.pt")
        print(f"best.pt copiat in {config.MODELS_DIR / 'yolo_signs.pt'}")
    print(json.dumps(metrics, indent=2))
    print(f"salvat in {out_dir}")


if __name__ == "__main__":
    main()
