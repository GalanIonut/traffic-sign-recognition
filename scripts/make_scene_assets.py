"""Copiaza in app/assets/scenes cele 6 poze de test alese de evaluate_pipeline.py + scenes_gt.json."""
import argparse
import json
import shutil
import sys
from pathlib import Path

from traffic_signs import config
from traffic_signs.detection import load_gt
from traffic_signs.scene import gt_for_image, write_scene_gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=config.GTSDB_YOLO_DIR / "images" / "test")
    ap.add_argument("--gt", type=Path, default=config.GTSDB_DIR / "gt.txt")
    ap.add_argument("--metrics", type=Path, default=config.M8_REPORTS_DIR / "metrics.json")
    ap.add_argument("--assets-dir", type=Path, default=config.APP_ASSETS_DIR)
    args = ap.parse_args()

    if not args.metrics.exists():
        sys.exit(f"Lipsește {args.metrics}; rulează întâi evaluate_pipeline.py.")
    ids = json.loads(args.metrics.read_text())["example_ids"]
    gt = load_gt(args.gt)
    out = args.assets_dir / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    for p in list(out.glob("*.jpg")) + [out / "scenes_gt.json"]:
        p.unlink(missing_ok=True)
    for n in ids:
        shutil.copy(args.images_dir / f"{n}.jpg", out / f"{n}.jpg")
    write_scene_gt(args.assets_dir, {n: gt_for_image(gt, n) for n in ids})
    print(f"{len(ids)} poze copiate în {out}: {', '.join(ids)}")


if __name__ == "__main__":
    main()
