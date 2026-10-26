"""Comparația CNN propriu vs. ResNet50: tabel și figură pentru slide (16x9)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from traffic_signs.presentation import BG, BLUE, GRID, INK, INK2, ORANGE, W, H, _save, _style_axes

COLUMNS = ["model", "test_accuracy", "f1_macro", "params", "trainable_params", "model_mb",
           "train_minutes", "epochs_run", "ms_per_image_batch", "ms_per_image_single"]
_MB = 1024 * 1024  # mărimea fișierului în MiB
_NAMES = {"cnn": "CNN propriu", "resnet": "ResNet50", "resnet50": "ResNet50"}


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def model_comparison_table(entries: list[dict]) -> pd.DataFrame:
    """Un rând per model; entries = [{"label", "run_dir", "metrics_dir"}, ...]."""
    rows = []
    for e in entries:
        run_dir = Path(e["run_dir"])
        cfg = _read_json(run_dir / "config.json")
        metrics = _read_json(Path(e["metrics_dir"]) / "metrics.json")
        rows.append({
            "model": e["label"],
            "test_accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "params": cfg["params"],
            "trainable_params": cfg.get("trainable_params", cfg["params"]),
            "model_mb": (run_dir / "model.keras").stat().st_size / _MB,
            "train_minutes": cfg["train_seconds"] / 60,
            "epochs_run": cfg["epochs_run"],
            "ms_per_image_batch": metrics["ms_per_image_batch"],
            "ms_per_image_single": metrics["ms_per_image_single"],
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def _ro(x: float, decimals: int = 1) -> str:
    """Număr cu virgulă zecimală: 12.34 -> '12,3'."""
    return f"{x:.{decimals}f}".replace(".", ",")


def _times(ratio: float) -> str:
    """Raport ca „18×” sau „2,5×”."""
    return f"{ratio:.0f}×" if ratio >= 10 else f"{_ro(ratio)}×"


def _pick(table: pd.DataFrame, label: str) -> pd.Series:
    """Rândul modelului după etichetă; ValueError dacă lipsește."""
    rows = table[table["model"] == label]
    if rows.empty:
        raise ValueError(f"Modelul '{label}' lipsește din tabel (există: {list(table['model'])}).")
    return rows.iloc[0]


def comparison_title(table: pd.DataFrame, cnn_label: str = "cnn",
                     resnet_label: str = "resnet") -> str:
    """Titlu-concluzie calculat din cifre, adevărat oricare model câștigă."""
    cnn, res = _pick(table, cnn_label), _pick(table, resnet_label)
    diff_pp = (res["test_accuracy"] - cnn["test_accuracy"]) * 100
    dec = 2 if abs(diff_pp) < 0.5 else 1  # diferențe mici: 2 zecimale
    if abs(diff_pp) < 0.005:
        head = "ResNet50 și CNN-ul au aceeași acuratețe"
    elif diff_pp > 0:
        head = f"ResNet50 câștigă {_ro(diff_pp, dec)} pp acuratețe"
    else:
        head = f"CNN-ul propriu câștigă {_ro(-diff_pp, dec)} pp acuratețe"

    size = res["model_mb"] / cnn["model_mb"]
    time = res["train_minutes"] / cnn["train_minutes"]
    size_txt = (f"e de {_times(size)} mai mare" if size >= 1.05 else
                f"e de {_times(1 / size)} mai mic" if size <= 1 / 1.05 else "are aceeași mărime")
    time_txt = (f"e de {_times(time)} mai lent la antrenare" if time >= 1.05 else
                f"e de {_times(1 / time)} mai rapid la antrenare" if time <= 1 / 1.05
                else "se antrenează în același timp")
    if size_txt.startswith("e de") and time_txt.startswith("e de"):
        tail = f"{size_txt} și {time_txt[2:].strip()}"  # „e de 18× mai mare și de 3× mai lent…”
    else:
        tail = f"{size_txt}, iar {time_txt}"
    worse = size >= 1.05 or time >= 1.05  # ResNet50 pierde la cel puțin un cost
    if diff_pp > 0.005:
        return f"{head}, {'dar' if worse else 'și'} {tail}"
    return f"{head}; ResNet50 {tail}"


PANELS = [
    ("error_pct", "Eroare pe test (%)", lambda v: _ro(v, 2) + "%"),
    ("train_minutes", "Timp de antrenare (minute)", lambda v: _ro(v, 1)),
    ("model_mb", "Mărimea modelului (MB)", lambda v: _ro(v, 1)),
    ("ms_per_image_batch", "Timp de inferență (ms/imagine, batch)", lambda v: _ro(v, 2)),
]


def plot_cnn_vs_resnet(table: pd.DataFrame, out_path: Path, cnn_label: str = "cnn",
                       resnet_label: str = "resnet") -> None:
    """Patru panouri de bare orizontale, fiecare cu axa ei; rândurile se aleg după etichetă."""
    title_txt = comparison_title(table, cnn_label, resnet_label)
    table = pd.DataFrame([_pick(table, cnn_label), _pick(table, resnet_label)]).reset_index(drop=True)
    table["error_pct"] = (1 - table["test_accuracy"].astype(float)) * 100
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    fig.text(0.6 / W, 1 - 0.75 / H, title_txt, fontsize=20, fontweight="bold",
             color=INK, ha="left", va="center")
    fig.text(0.6 / W, 1 - 1.35 / H, "Același set de test (12.630 de imagini, semne decupate)",
             fontsize=15, color=INK2, ha="left", va="center")
    n = len(table)
    names = [_NAMES.get(m, m) for m in table["model"]]
    colors = [ORANGE if "resnet" in str(m).lower() else BLUE for m in table["model"]]
    for i, (col, title, fmt) in enumerate(PANELS):
        r, c = divmod(i, 2)
        x0 = 2.3 + c * 7.6
        y0 = 4.35 - r * 3.85
        ax = fig.add_axes([x0 / W, y0 / H, 5.1 / W, 2.45 / H])
        _style_axes(ax)
        vals = table[col].astype(float).tolist()
        ys = list(range(n))[::-1]
        ax.barh(ys, vals, height=0.55, color=colors, zorder=2)
        top = max(vals) * 1.3 if max(vals) > 0 else 1
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _ro(v, 0 if v == int(v) else 1)))
        for y, v in zip(ys, vals):
            ax.text(v + top * 0.015, y, fmt(v), fontsize=16, fontweight="bold", color=INK,
                    ha="left", va="center", zorder=3)
        ax.set_yticks(ys, names)
        ax.tick_params(axis="y", labelsize=14, pad=8)
        ax.set_xlim(0, top)
        ax.set_ylim(-0.6, n - 0.4)
        ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_title(title, fontsize=16, fontweight="bold", color=INK, loc="left", pad=10)
    _save(fig, out_path)
