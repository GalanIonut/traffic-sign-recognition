"""Antreneaza ResNet50 in doua etape (cap, apoi conv5_); salveaza in runs/<nume>."""
import argparse

from traffic_signs import config
from traffic_signs.augmentation import compute_class_weights
from traffic_signs.models import build_resnet50
from traffic_signs.pipeline import make_dataset
from traffic_signs.reproducibility import set_global_seed
from traffic_signs.split import load_split
from traffic_signs.training import train_two_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="resnet50")
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    set_global_seed(config.SEED)
    train_df, val_df = load_split(config.SPLITS_DIR / "train_val_split.csv",
                                  config.DATA_DIR)
    size = config.IMG_SIZE_RESNET
    train_ds = make_dataset(train_df, size, training=True, crop=True)
    val_ds = make_dataset(val_df, size, training=False, crop=True)
    weights = None if args.no_class_weights else compute_class_weights(
        train_df["class_id"].tolist())
    model = build_resnet50(input_size=size)
    run_dir = config.PROJECT_ROOT / "runs" / args.run_name
    print(f"director run: {run_dir}")
    history = train_two_stage(
        model, train_ds, val_ds, run_dir,
        stage1={"epochs": args.stage1_epochs, "learning_rate": 1e-3, "patience": 2},
        stage2={"epochs": args.stage2_epochs, "learning_rate": 1e-4, "patience": 3},
        class_weight=weights, run_config=vars(args))
    best = history[history["stage"] == 2].pipe(lambda h: h.loc[h["val_loss"].idxmin()])
    print(f"cea mai buna epoca: {int(best['epoch'])}, "
          f"acuratete validare: {best['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
