"""Converteste GTSDB (gt.txt + poze) in setul YOLO cu o singura clasa (data/gtsdb_yolo)."""
import argparse
from pathlib import Path

from traffic_signs import config
from traffic_signs.detection import build_yolo_dataset, load_gt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path,
                        default=Path.home() / "Downloads" / "GTSDB_Train_and_Test")
    parser.add_argument("--gt", type=Path, default=config.GTSDB_DIR / "gt.txt")
    parser.add_argument("--out", type=Path, default=config.GTSDB_YOLO_DIR)
    args = parser.parse_args()

    gt = load_gt(args.gt)
    counts = build_yolo_dataset(gt, args.images_dir, args.out)
    print(f"{'set':<6}{'poze':>6}{'patrate':>9}{'fara semne':>12}")
    for split, c in counts.items():
        print(f"{split:<6}{c['images']:>6}{c['boxes']:>9}{c['empty']:>12}")
    print(f"total patrate: {sum(c['boxes'] for c in counts.values())}")
    print(f"salvat in {args.out}")


if __name__ == "__main__":
    main()
