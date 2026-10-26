"""Compara run-urile din runs/: tabel CSV, figura de validare si export de figuri pentru slide.

Doar cu --out reports/m4 (implicit) copiaza in reports/figures/ (daca exista) figurile de
prezentare: m4_curves_baseline.png, m4_curves_no_reg.png, m4_confusion_baseline.png,
m4_misclassified_baseline.png. Cu alt --out nu se copiaza nimic (nu suprascrie figurile M4).
"""
import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from traffic_signs import config
from traffic_signs.evaluation import DPI, FONT, compare_runs, load_history

# id run -> eticheta in romana; run-urile necunoscute isi pastreaza numele
LABELS = {"baseline": "de bază", "no_reg": "fără regularizare", "no_crop": "fără decupare*"}
EXPORTS = [("baseline", "curves.png", "m4_curves_baseline.png"),
           ("no_reg", "curves.png", "m4_curves_no_reg.png"),
           ("baseline", "confusion_matrix.png", "m4_confusion_baseline.png"),
           ("baseline", "misclassified.png", "m4_misclassified_baseline.png")]


def plot_comparison(run_dirs, fig_path) -> None:
    """Doua panouri: acuratete pe validare (zoom 0,9-1,0) si pierdere pe validare (log)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for d in run_dirs:
        h = load_history(d)
        label = LABELS.get(d.name, d.name)
        for ax, key in zip(axes, ("val_accuracy", "val_loss")):
            ax.plot(h["epoch"], h[key], linewidth=2.5, label=label)
    axes[0].set_ylim(0.9, 1.0)
    axes[0].set_ylabel("Acuratețe pe validare", fontsize=FONT)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Pierdere pe validare (scară logaritmică)", fontsize=FONT)
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Epocă", fontsize=FONT)
        ax.tick_params(labelsize=FONT - 2)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=FONT - 2, loc="lower right")
    fig.suptitle("Comparație între run-uri", fontsize=FONT + 4)
    if any(d.name == "no_crop" for d in run_dirs):
        fig.text(0.5, 0.0, "*validare pe imagini nedecupate", ha="center", va="top",
                 fontsize=FONT - 2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def default_figure_path(out_dir) -> Path:
    """Figura de validare: numele M4 doar pentru reports/m4, altfel dupa folderul --out."""
    out_dir = config.PROJECT_ROOT / out_dir
    if out_dir.resolve() == (config.PROJECT_ROOT / "reports" / "m4").resolve():
        name = "m4"
    else:
        name = out_dir.name
    return config.FIGURES_DIR / f"{name}_val_accuracy_comparison.png"


def export_m4_figures(out_dir, figures_dir=None) -> list[Path]:
    """Copiaza figurile m4_* in figures_dir, dar doar daca out_dir este reports/m4."""
    out_dir = config.PROJECT_ROOT / out_dir  # cale relativa -> fata de PROJECT_ROOT
    if out_dir.resolve() != (config.PROJECT_ROOT / "reports" / "m4").resolve():
        return []
    figures_dir = Path(figures_dir or config.FIGURES_DIR)
    copied = []
    for run, src, dst in EXPORTS:
        if (out_dir / run / src).exists():
            copied.append(Path(shutil.copy2(out_dir / run / src, figures_dir / dst)))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/m4", help="folderul cu rezultate")
    parser.add_argument("--figure", default=None,
                        help="implicit: m4_val_accuracy_comparison.png pentru reports/m4, "
                             "altfel <folder>_val_accuracy_comparison.png")
    parser.add_argument("--runs", nargs="*", help="doar aceste run-uri (implicit toate)")
    args = parser.parse_args()

    runs_dir = config.PROJECT_ROOT / "runs"
    # Ignora directoarele temporare (_) si pe cele fara config.json
    run_dirs = sorted(d for d in runs_dir.iterdir()
                      if d.is_dir() and not d.name.startswith("_")
                      and (d / "config.json").exists()
                      and (not args.runs or d.name in args.runs))
    if not run_dirs:
        raise SystemExit("Nu există run-uri valide în runs/.")
    out_dir = config.PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    table = compare_runs(run_dirs)
    csv_path = out_dir / "comparison.csv"
    table.to_csv(csv_path, index=False)

    fig_path = (config.PROJECT_ROOT / args.figure if args.figure
                else default_figure_path(args.out))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(run_dirs, fig_path)
    export_m4_figures(out_dir)
    print(table.to_string(index=False))
    print(f"salvat in {csv_path} si {fig_path}")


if __name__ == "__main__":
    main()
