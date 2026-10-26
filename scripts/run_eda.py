"""Genereaza figurile EDA in reports/figures/ si tipareste statistici."""
from traffic_signs import config
from traffic_signs.dataset import load_test_index, load_train_index
from traffic_signs.eda import (class_distribution, plot_class_distribution,
                               plot_class_grid, plot_size_histogram,
                               plot_split_distribution)
from traffic_signs.split import load_split


def main() -> None:
    fig_dir = config.FIGURES_DIR
    train_all = load_train_index(config.DATA_DIR)
    test = load_test_index(config.DATA_DIR)
    train, val = load_split(config.SPLITS_DIR / "train_val_split.csv", config.DATA_DIR)

    counts = class_distribution(train_all)
    print(f"imagini train (total): {len(train_all)}, test: {len(test)}")
    print(f"imagini/clasa: min={counts.min()} (clasa {counts.idxmin()}), "
          f"max={counts.max()} (clasa {counts.idxmax()}), medie={counts.mean():.1f}")
    for col in ("width", "height"):
        s = train_all[col]
        print(f"{col}: min={s.min()}, max={s.max()}, medie={s.mean():.1f}")

    plot_class_distribution(counts, fig_dir / "class_distribution_train.png",
                            "Distribuția claselor în setul de antrenare")
    plot_size_histogram(train_all, fig_dir / "image_sizes.png")
    plot_class_grid(config.DATA_DIR / "Meta", fig_dir / "class_grid.png")
    plot_split_distribution(train, val, test, fig_dir / "split_distribution.png")
    print(f"figuri salvate in {fig_dir}")


if __name__ == "__main__":
    main()
