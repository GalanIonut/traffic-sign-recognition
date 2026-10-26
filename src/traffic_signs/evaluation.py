"""Evaluare pe test: predictii, metrici, grafice si comparatie intre run-uri."""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from traffic_signs.classes import class_name
from traffic_signs.config import IMG_SIZE_CNN, NUM_CLASSES
from traffic_signs.pipeline import make_dataset
from traffic_signs.preprocessing import crop_roi, load_image

DPI = 150
FONT = 16  # lizibil si pe un slide 16:9


def predict_classes(model, df: pd.DataFrame, image_size: int = IMG_SIZE_CNN,
                    crop: bool = True, batch_size: int = 256):
    """(y_true, y_pred, probs) in ordinea randurilor din df."""
    ds = make_dataset(df, image_size, batch_size=batch_size, training=False, crop=crop)
    probs = model.predict(ds, verbose=0)
    return df["class_id"].to_numpy(), probs.argmax(axis=1), probs


def compute_metrics(y_true, y_pred, num_classes: int = NUM_CLASSES):
    """Metrici globale (dict) si pe clasa (DataFrame); mereu toate clasele."""
    labels = list(range(num_classes))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
    }
    per_class = pd.DataFrame({
        "class_id": labels, "name": [class_name(c) for c in labels],
        "precision": prec, "recall": rec, "f1": f1, "support": support})
    return metrics, per_class


def load_history(run_dir: Path) -> pd.DataFrame:
    """Citeste history.csv; epocile sunt mereu 1-based (run-urile vechi erau 0-based)."""
    history = pd.read_csv(Path(run_dir) / "history.csv")
    if history["epoch"].iloc[0] == 0:
        history["epoch"] += 1
    return history


def _save(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: pd.DataFrame, out_path: Path, title: str) -> None:
    """Curbe loss si acuratete (antrenare vs. validare), cu epoca cea mai buna."""
    best = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, key, label in ((axes[0], "loss", "Pierdere (loss)"),
                           (axes[1], "accuracy", "Acuratețe")):
        ax.plot(history["epoch"], history[key], label="antrenare", linewidth=2.5)
        ax.plot(history["epoch"], history[f"val_{key}"], label="validare", linewidth=2.5)
        if key == "loss":
            ax.set_yscale("log")
        ax.axvline(best, color="gray", linestyle="--", label=f"cea mai bună epocă ({best})")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epoci intregi
        ax.set_xlabel("Epocă", fontsize=FONT)
        ax.set_ylabel(label, fontsize=FONT)
        ax.tick_params(labelsize=FONT - 2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=FONT - 2)
    fig.suptitle(title, fontsize=FONT + 4)
    fig.tight_layout()
    _save(fig, out_path)


def plot_confusion_matrix(y_true, y_pred, out_path: Path,
                          num_classes: int = NUM_CLASSES) -> None:
    """Matrice de confuzie normalizata pe randuri (procent), etichete = ID clasa."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(float)
    sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, sums, out=np.zeros_like(cm), where=sums > 0) * 100
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ticks = range(0, num_classes, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=11, rotation=90)
    ax.set_yticklabels(ticks, fontsize=11)
    ax.set_title("Matricea de confuzie (normalizată pe rânduri)", fontsize=FONT + 2)
    ax.set_xlabel("Clasa prezisă", fontsize=FONT)
    ax.set_ylabel("Clasa reală", fontsize=FONT)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("% din clasa reală", fontsize=FONT - 2)
    cbar.ax.tick_params(labelsize=FONT - 4)
    _save(fig, out_path)


def _short(class_id: int, width: int = 22) -> str:
    return textwrap.shorten(class_name(int(class_id)), width, placeholder="…")


def select_misclassified(y_true, y_pred, conf, n: int = 16, per_pair: int = 2):
    """Indicii celor mai sigure greseli, cu cel mult per_pair pe pereche (real, prezis)."""
    y_true, y_pred, conf = np.asarray(y_true), np.asarray(y_pred), np.asarray(conf)
    wrong = np.flatnonzero(y_true != y_pred)
    ordered = wrong[np.argsort(-conf[wrong], kind="stable")]
    seen: dict[tuple[int, int], int] = {}
    chosen = []
    for i in ordered:
        pair = (int(y_true[i]), int(y_pred[i]))
        if seen.get(pair, 0) < per_pair:
            seen[pair] = seen.get(pair, 0) + 1
            chosen.append(int(i))
        if len(chosen) == n:
            break
    return np.array(chosen, dtype=int)


def plot_misclassified(df: pd.DataFrame, y_true, y_pred, probs, out_path: Path,
                       n: int = 16, crop: bool = True) -> None:
    """Grila cu cele mai sigure n greseli (max. 2 per pereche real/prezis)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    conf = np.asarray(probs).max(axis=1)
    top = select_misclassified(y_true, y_pred, conf, n)
    cols = 4
    rows = max(1, -(-n // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    if len(top) == 0:
        fig.text(0.5, 0.5, "Nicio greșeală", ha="center", va="center", fontsize=FONT + 8)
    for ax, i in zip(axes.ravel(), top):
        row = df.iloc[i]
        img = load_image(row["path"])
        if crop:
            img = crop_roi(img, int(row["roi_x1"]), int(row["roi_y1"]),
                           int(row["roi_x2"]), int(row["roi_y2"]))
        ax.imshow(img)
        ax.set_title(f"real: {_short(y_true[i])}\nprezis: {_short(y_pred[i])} "
                     f"({conf[i] * 100:.0f}%)", fontsize=FONT - 2)
    fig.tight_layout()
    _save(fig, out_path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def compare_runs(run_dirs: list[Path]) -> pd.DataFrame:
    """Un rand pe run (+ un rand pentru fiecare metrics_*.json, ex. evaluare nedecupata)."""
    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        cfg = _read_json(run_dir / "config.json")
        variants = [(run_dir.name, run_dir / "metrics.json")]
        for extra in sorted(run_dir.glob("metrics_*.json")):
            tag = extra.stem.removeprefix("metrics_")
            label = "nedecupat" if tag == "nocrop" else tag
            variants.append((f"{run_dir.name} ({label})", extra))
        for name, path in variants:
            metrics = _read_json(path)
            rows.append({
                "run": name,
                "crop": metrics.get("crop", np.nan),
                "epochs_run": cfg.get("epochs_run"),
                "best_epoch": cfg.get("best_epoch"),
                "best_val_accuracy": cfg.get("best_val_accuracy"),
                "train_seconds": cfg.get("train_seconds"),
                "params": cfg.get("params"),
                "test_accuracy": metrics.get("accuracy", np.nan),
                "f1_macro": metrics.get("f1_macro", np.nan),
                "ms_per_image_batch": metrics.get("ms_per_image_batch", np.nan),
            })
    return pd.DataFrame(rows)
