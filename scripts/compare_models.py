"""Compara CNN-ul propriu (baseline) cu ResNet50 (conv5 si, daca exista, conv4+conv5): tabel CSV si figura pentru slide.

Scrie reports/m5/comparison.csv si reports/figures/pres_cnn_vs_resnet.png.
"""
from traffic_signs import config
from traffic_signs.comparison import model_comparison_table, plot_cnn_vs_resnet


def build_entries(root) -> list[dict]:
    """CNN + ResNet (conv5); a treia intrare (conv4+conv5) doar dacă run-ul și rapoartele există."""
    entries = [
        {"label": "cnn", "run_dir": root / "runs" / "baseline",
         "metrics_dir": root / "reports" / "m4" / "baseline"},
        {"label": "resnet", "run_dir": root / "runs" / "resnet50",
         "metrics_dir": root / "reports" / "m5" / "resnet50"},
    ]
    conv4 = {"label": "resnet_conv4", "run_dir": root / "runs" / "resnet50_conv4",
             "metrics_dir": root / "reports" / "m5" / "resnet50_conv4"}
    if conv4["run_dir"].exists() and conv4["metrics_dir"].exists():
        entries.append(conv4)
    return entries


def main() -> None:
    root = config.PROJECT_ROOT
    entries = build_entries(root)
    table = model_comparison_table(entries)
    csv_path = root / "reports" / "m5" / "comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    fig_path = config.FIGURES_DIR / "pres_cnn_vs_resnet.png"
    plot_cnn_vs_resnet(table, fig_path, "cnn", [e["label"] for e in entries[1:]])
    print(table.to_string(index=False))
    print(f"salvat in {csv_path} si {fig_path}")


if __name__ == "__main__":
    main()
