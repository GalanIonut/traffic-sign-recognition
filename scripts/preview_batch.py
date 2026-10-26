"""Salveaza reports/figures/augmented_batch.png: 16 imagini augmentate din train."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traffic_signs import config
from traffic_signs.classes import class_name
from traffic_signs.pipeline import make_dataset
from traffic_signs.split import load_split


def main() -> None:
    train, _ = load_split(config.SPLITS_DIR / "train_val_split.csv", config.DATA_DIR)
    ds = make_dataset(train, config.IMG_SIZE_CNN, batch_size=16, training=True)
    images, labels = next(iter(ds))
    fig, axes = plt.subplots(4, 4, figsize=(9, 10))
    for ax, img, lab in zip(axes.ravel(), images.numpy(), labels.numpy()):
        cid = int(lab.argmax())
        ax.imshow(img)
        ax.set_title(textwrap.fill(f"{cid}: {class_name(cid)}", 24), fontsize=7)
        ax.axis("off")
    fig.suptitle("Batch augmentat (antrenare)")
    fig.tight_layout()
    out = config.FIGURES_DIR / "augmented_batch.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"salvat in {out}")


if __name__ == "__main__":
    main()
