"""Antreneaza CNN-ul propriu pe impartirea train/validare; salveaza in runs/<nume>."""
import argparse

from traffic_signs import config
from traffic_signs.augmentation import compute_class_weights
from traffic_signs.models import build_cnn
from traffic_signs.pipeline import make_dataset
from traffic_signs.reproducibility import set_global_seed
from traffic_signs.split import load_split
from traffic_signs.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    set_global_seed(config.SEED)
    train_df, val_df = load_split(config.SPLITS_DIR / "train_val_split.csv",
                                  config.DATA_DIR)
    crop = not args.no_crop
    # Train se amesteca mereu; augmentarea poate fi oprita separat
    train_ds = make_dataset(train_df, config.IMG_SIZE_CNN, training=True,
                            augment=not args.no_augment, crop=crop)
    val_ds = make_dataset(val_df, config.IMG_SIZE_CNN, training=False, crop=crop)
    weights = None if args.no_class_weights else compute_class_weights(
        train_df["class_id"].tolist())
    model = build_cnn(dropout=args.dropout)
    run_dir = config.PROJECT_ROOT / "runs" / args.run_name
    print(f"director run: {run_dir}")
    history = train_model(model, train_ds, val_ds, run_dir=run_dir,
                          epochs=args.epochs, class_weight=weights,
                          run_config=vars(args))
    best = history.loc[history["val_loss"].idxmin()]
    print(f"cea mai buna epoca: {int(best['epoch'])}, "
          f"acuratete validare: {best['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
