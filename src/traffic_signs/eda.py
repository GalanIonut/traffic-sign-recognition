"""Explorarea datelor: statistici si grafice (titluri si axe in romana)."""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from traffic_signs.classes import class_name  # noqa: E402
from traffic_signs.config import NUM_CLASSES  # noqa: E402

DPI = 150
COLOR = "#2b6cb0"


def class_distribution(df: pd.DataFrame, num_classes: int = NUM_CLASSES) -> pd.Series:
    """Numarul de imagini pe clasa; clasele lipsa au 0."""
    counts = df["class_id"].value_counts().reindex(range(num_classes), fill_value=0)
    counts = counts.astype(int).sort_index()
    counts.index.name = "class_id"
    return counts


def _save(fig: plt.Figure, out_path: Path) -> None:
    """Salveaza figura si o inchide (fara scurgeri de memorie)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(counts: pd.Series, out_path: Path, title: str) -> None:
    """Grafic cu bare: numarul de imagini pe clasa."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(counts.index, counts.values, color=COLOR)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("ID clasă", fontsize=13)
    ax.set_ylabel("Număr de imagini", fontsize=13)
    ax.set_xticks(list(counts.index))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(-0.8, len(counts) - 0.2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, out_path)


def plot_size_histogram(df: pd.DataFrame, out_path: Path) -> None:
    """Histograme pentru latimea si inaltimea imaginilor."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, col, label in zip(axes, ["width", "height"], ["Lățime (px)", "Înălțime (px)"]):
        ax.hist(df[col], bins=40, color=COLOR, edgecolor="white")
        ax.axvline(df[col].mean(), color="crimson", linestyle="--",
                   label=f"medie = {df[col].mean():.1f}")
        ax.set_xlabel(label, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Număr de imagini", fontsize=13)
    fig.suptitle("Distribuția dimensiunilor imaginilor", fontsize=16)
    _save(fig, out_path)


def _short_name(class_id: int, width: int = 20, max_lines: int = 4) -> str:
    """Nume scurt, pe cel mult `max_lines` randuri."""
    lines = textwrap.wrap(class_name(class_id), width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 1].rstrip() + "…"
    return "\n".join(lines)


def plot_class_grid(meta_dir: Path, out_path: Path) -> None:
    """Grila cu cele 43 de semne din Meta/, cu ID si nume scurt."""
    cols, rows = 8, 6
    fig, axes = plt.subplots(rows, cols, figsize=(17, 14))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= NUM_CLASSES:
            continue
        with Image.open(Path(meta_dir) / f"{i}.png") as im:
            ax.imshow(np.asarray(im.convert("RGB")))
        ax.set_title(f"{i}\n{_short_name(i)}", fontsize=9.5, pad=4)
    fig.suptitle("Cele 43 de clase de semne de circulație", fontsize=18)
    fig.subplots_adjust(hspace=0.75, wspace=0.15, top=0.92)
    _save(fig, out_path)


def plot_split_distribution(train: pd.DataFrame, val: pd.DataFrame,
                            test: pd.DataFrame, out_path: Path) -> None:
    """Proportia pe clasa (%) in fiecare set: train, validare, test."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(NUM_CLASSES)
    width = 0.27
    sets = [("Antrenare", train, "#2b6cb0"), ("Validare", val, "#dd6b20"),
            ("Test", test, "#38a169")]
    for k, (name, df, color) in enumerate(sets):
        share = class_distribution(df) / max(len(df), 1) * 100
        ax.bar(x + (k - 1) * width, share.values, width, label=name, color=color)
    ax.set_title("Proporția claselor în seturile de antrenare, validare și test",
                 fontsize=16)
    ax.set_xlabel("ID clasă", fontsize=13)
    ax.set_ylabel("Proporție din set (%)", fontsize=13)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(-0.8, NUM_CLASSES - 0.2)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, out_path)
