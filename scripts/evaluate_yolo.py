"""Evalueaza o rulare YOLO: metrici, curbe si exemple cu patratele prezise vs reale."""
import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from traffic_signs import config
from traffic_signs.detection import summarize_metrics

GREEN = "#2ca02c"
ORANGE = "#eb6834"


def read_gt_boxes(label_file: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    """Etichete YOLO normalizate -> (x1, y1, x2, y2) in pixeli."""
    boxes = []
    for line in label_file.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        boxes.append(((cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def draw_examples(model, split: str, conf: float, imgsz: int, device: str, out_path: Path) -> None:
    images = sorted((config.GTSDB_YOLO_DIR / "images" / split).glob("*.jpg"))
    chosen = random.Random(config.SEED).sample(images, 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 7.6))
    for ax, img_path in zip(axes.ravel(), chosen):
        img = Image.open(img_path).convert("RGB")
        pred = model.predict(str(img_path), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        ax.imshow(img)
        for x1, y1, x2, y2 in read_gt_boxes(
                config.GTSDB_YOLO_DIR / "labels" / split / f"{img_path.stem}.txt", *img.size):
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor=ORANGE, linewidth=2.5))
        n_pred = 0
        for x1, y1, x2, y2 in pred.boxes.xyxy.cpu().numpy():
            n_pred += 1
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor=GREEN, linewidth=1.5, linestyle="--"))
        ax.set_title(f"{img_path.name} - {n_pred} prezise", fontsize=11)
        ax.axis("off")
    handles = [patches.Patch(edgecolor=GREEN, fill=False, label="Prezis (model)"),
               patches.Patch(edgecolor=ORANGE, fill=False, label="Real (etichetă)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=13, frameon=False)
    fig.suptitle(f"Detecția semnelor: exemple din setul de {split}", fontsize=17,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=2.5)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


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

    from ultralytics import YOLO
    model = YOLO(str(best))
    results = model.val(data=str(config.GTSDB_YOLO_DIR / "dataset.yaml"), split=args.split,
                        imgsz=args.imgsz, device=args.device, conf=0.001,
                        project=str(out_dir / "_val"), name="val", exist_ok=True, plots=True)
    metrics = summarize_metrics(results)
    metrics.update(split=args.split, imgsz=args.imgsz,
                   ms_per_image=float(results.speed["inference"]))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    for name in ("results.png", "PR_curve.png", "BoxPR_curve.png"):
        for src in (run_dir / name, Path(results.save_dir) / name):
            if src.exists():
                shutil.copy2(src, out_dir / name)
                break
    shutil.rmtree(out_dir / "_val", ignore_errors=True)

    draw_examples(model, args.split, args.conf, args.imgsz, args.device, out_dir / "examples.png")
    if args.promote:
        shutil.copy2(best, config.MODELS_DIR / "yolo_signs.pt")
        print(f"best.pt copiat in {config.MODELS_DIR / 'yolo_signs.pt'}")
    print(json.dumps(metrics, indent=2))
    print(f"salvat in {out_dir}")


if __name__ == "__main__":
    main()
