"""Copiaza un run antrenat in models/ (model + config) pentru folosire in M5."""
import argparse
import shutil
import sys

from traffic_signs import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="run din runs/")
    parser.add_argument("--as", dest="target", required=True,
                        help="numele de baza in models/ (ex. cnn_baseline)")
    args = parser.parse_args()

    run_dir = config.PROJECT_ROOT / "runs" / args.run_name
    for name in ("model.keras", "config.json"):
        if not (run_dir / name).exists():
            sys.exit(f"Eroare: lipsește {run_dir / name}")
    models_dir = config.PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    shutil.copy2(run_dir / "model.keras", models_dir / f"{args.target}.keras")
    shutil.copy2(run_dir / "config.json", models_dir / f"{args.target}.json")
    print(f"copiat în {models_dir}/{args.target}.keras și .json")


if __name__ == "__main__":
    main()
