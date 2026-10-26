"""Compara toate run-urile din runs/: tabel CSV si curbele de acuratete pe validare."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from traffic_signs import config
from traffic_signs.evaluation import DPI, FONT, compare_runs


def main() -> None:
    runs_dir = config.PROJECT_ROOT / "runs"
    # Ignora directoarele temporare (_) si pe cele fara config.json
    run_dirs = sorted(d for d in runs_dir.iterdir()
                      if d.is_dir() and not d.name.startswith("_")
                      and (d / "config.json").exists())
    if not run_dirs:
        raise SystemExit("Nu există run-uri valide în runs/.")
    table = compare_runs(run_dirs)
    csv_path = config.PROJECT_ROOT / "reports" / "m4" / "comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(16, 9))
    for d in run_dirs:
        h = pd.read_csv(d / "history.csv")
        ax.plot(h["epoch"], h["val_accuracy"], linewidth=2.5, label=d.name)
    ax.set_xlabel("Epocă", fontsize=FONT + 2)
    ax.set_ylabel("Acuratețe pe validare", fontsize=FONT + 2)
    ax.set_title("Comparație între run-uri", fontsize=FONT + 6)
    ax.tick_params(labelsize=FONT)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=FONT)
    fig_path = config.FIGURES_DIR / "m4_val_accuracy_comparison.png"
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(table.to_string(index=False))
    print(f"salvat in {csv_path} si {fig_path}")


if __name__ == "__main__":
    main()
