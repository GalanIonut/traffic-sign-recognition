"""Genereaza splits/train_val_split.csv din setul real (archive/)."""
from traffic_signs import config
from traffic_signs.dataset import load_train_index
from traffic_signs.split import save_split, split_by_track


def main() -> None:
    df = load_train_index(config.DATA_DIR)
    train, val = split_by_track(df, config.VAL_FRACTION, config.SEED)
    out_path = config.SPLITS_DIR / "train_val_split.csv"
    save_split(train, val, out_path, config.DATA_DIR)
    min_val = val.groupby("class_id")["track_id"].nunique().min()
    print(f"train: {len(train)} imagini, {train['track_id'].nunique()} track-uri")
    print(f"val:   {len(val)} imagini, {val['track_id'].nunique()} track-uri")
    print(f"val: {len(val) / len(df):.1%} din imagini")
    print(f"minim track-uri val pe clasa: {min_val}")
    print(f"salvat in {out_path}")


if __name__ == "__main__":
    main()
