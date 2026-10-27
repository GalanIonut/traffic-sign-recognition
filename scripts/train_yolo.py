"""Antreneaza YOLO (o singura clasa: semn) pe setul GTSDB convertit."""
import argparse
import time
from pathlib import Path

from traffic_signs import config
from traffic_signs.detection import train_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    # Calea in models/ face ca ponderile descarcate sa nu ajunga in radacina
    parser.add_argument("--model", default=str(config.MODELS_DIR / "yolo11n.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    data_yaml = config.GTSDB_YOLO_DIR / "dataset.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Lipseste {data_yaml}. Ruleaza intai: "
                         "uv run python scripts/make_yolo_dataset.py")

    from ultralytics import YOLO
    model = YOLO(args.model)
    kwargs = train_args(data_yaml, args.run_name, args.epochs, args.imgsz,
                        args.batch, args.device)
    start = time.time()
    model.train(**kwargs)
    seconds = time.time() - start
    run_dir = Path(kwargs["project"]) / args.run_name
    (run_dir / "train_seconds.txt").write_text(f"{seconds:.1f}\n")
    print(f"Antrenare terminata in {seconds:.0f} s ({seconds / args.epochs:.1f} s/epoca, "
          f"{args.device}); rezultate in {run_dir}")


if __name__ == "__main__":
    main()
