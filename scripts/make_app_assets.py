"""Genereaza resursele aplicatiei in app/assets/: imaginile oficiale (Meta) si exemple din Test.

Pentru fiecare clasa din SAMPLE_CLASSES alege imaginea cu cea mai mare arie (Width*Height) din Test.csv.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from traffic_signs import config

SAMPLE_CLASSES = (1, 9, 12, 13, 14, 17, 25, 33)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.APP_ASSETS_DIR)
    args = parser.parse_args()

    if not (args.data_dir / "Test.csv").is_file() or not (args.data_dir / "Meta").is_dir():
        sys.exit(f"Nu găsesc setul de date în {args.data_dir} (lipsește Test.csv sau Meta/).")

    meta_out = args.out_dir / "meta"
    samples_out = args.out_dir / "samples"
    meta_out.mkdir(parents=True, exist_ok=True)
    samples_out.mkdir(parents=True, exist_ok=True)

    for class_id in range(config.NUM_CLASSES):
        shutil.copyfile(args.data_dir / "Meta" / f"{class_id}.png", meta_out / f"{class_id}.png")

    test = pd.read_csv(args.data_dir / "Test.csv")
    test["area"] = test["Width"] * test["Height"]
    for class_id in SAMPLE_CLASSES:
        rows = test[test["ClassId"] == class_id].sort_values(["area", "Path"], ascending=[False, True])
        best = rows.iloc[0]
        with Image.open(args.data_dir / best["Path"]) as im:
            im.convert("RGB").save(samples_out / f"class_{class_id}.png")
        print(f"clasa {class_id}: {best['Path']} ({best['Width']}x{best['Height']})")
    print(f"salvat in {args.out_dir}")


if __name__ == "__main__":
    main()
