"""Scrie sumarul text si diagrama CNN-ului propriu."""
from traffic_signs import config
from traffic_signs.models import build_cnn, model_summary_text, plot_architecture


def main() -> None:
    model = build_cnn()
    summary_path = config.PROJECT_ROOT / "reports" / "m4" / "cnn_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(model_summary_text(model) + "\n", encoding="utf-8")
    figure_path = config.FIGURES_DIR / "m4_cnn_architecture.png"
    plot_architecture(model, figure_path)
    print(f"parametri: {model.count_params():,}")
    print(f"salvat in {summary_path} si {figure_path}")


if __name__ == "__main__":
    main()
