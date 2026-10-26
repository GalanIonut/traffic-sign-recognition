"""Grafice intuitive pentru prezentare (slide-uri 16:9).

Fiecare funcție `plot_*` primește date deja calculate și scrie un PNG.
Desenăm pe un strat transparent în „unități de slide” (16 x 9), ca poziționarea
textului și a imaginilor să fie exactă și fără suprapuneri.
"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from traffic_signs.classes import class_name

# --- Paleta (ghidul de vizualizare) ---
BG = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#c3c2b7"
TILE = "#f4f3ee"

DPI = 150
W, H = 16.0, 9.0  # unități de slide = inci la figsize 16x9

# Nume scurte pentru clasele cu nume lung (etichete pe grafice)
_SHORT = {
    6: "Sfârșit limită 80 km/h",
    10: "Depășire interzisă (>3,5 t)",
    11: "Prioritate la intersecție",
    15: "Circulație interzisă",
    16: "Acces interzis (>3,5 t)",
    18: "Atenție, alte pericole",
    24: "Drum îngustat la dreapta",
    32: "Sfârșit restricții",
    41: "Sfârșit interzicere depășire",
    42: "Sfârșit interz. depășire (>3,5 t)",
}


def short_name(class_id: int) -> str:
    """Numele clasei, prescurtat dacă e prea lung pentru o etichetă."""
    return _SHORT.get(int(class_id), class_name(int(class_id)))


def pct(x: float, decimals: int = 1) -> str:
    """Procent cu virgulă zecimală: 0.9903 -> '99,0%'."""
    return f"{x * 100:.{decimals}f}".replace(".", ",") + "%"


def fmt_int(n: int) -> str:
    """Număr întreg cu punct ca separator de mii: 39209 -> '39.209'."""
    return f"{int(n):,}".replace(",", ".")


# --- Calcule ---
def top_k(probs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Primele k clase după probabilitate: [(clasă, probabilitate), ...]."""
    probs = np.asarray(probs)
    idx = np.argsort(-probs, kind="stable")[:k]
    return [(int(i), float(probs[i])) for i in idx]


def top_confused_pairs(y_true, y_pred, k: int = 5) -> list[tuple[int, int, int, float]]:
    """Cele mai frecvente confuzii (real, prezis, număr, fracție din clasa reală)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    wrong = y_true != y_pred
    totals = np.bincount(y_true)
    pairs: dict[tuple[int, int], int] = {}
    for t, q in zip(y_true[wrong], y_pred[wrong]):
        pairs[(int(t), int(q))] = pairs.get((int(t), int(q)), 0) + 1
    ordered = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [(t, q, n, n / float(totals[t])) for (t, q), n in ordered]


def select_examples(y_true, y_pred, probs, n_correct: int = 6, n_wrong: int = 2,
                    seed: int = 42) -> list[int]:
    """Indici determiniști: n_correct predicții corecte, apoi n_wrong greșite."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    right = np.flatnonzero(y_true == y_pred)
    wrong = np.flatnonzero(y_true != y_pred)
    pick_r = rng.choice(right, size=min(n_correct, len(right)), replace=False)
    pick_w = rng.choice(wrong, size=min(n_wrong, len(wrong)), replace=False)
    return [int(i) for i in pick_r] + [int(i) for i in pick_w]


# --- Helpere de desen ---
def _canvas(height: float = H):
    """Figură 16 x height in, cu un strat în unități de slide (y în sus)."""
    fig = plt.figure(figsize=(W, height), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def _reset(ax, height: float = H) -> None:
    ax.set_xlim(0, W)
    ax.set_ylim(0, height)


def _title(ax, title: str, subtitle: str | None = None, height: float = H) -> None:
    ax.text(0.6, height - 0.75, title, fontsize=23, fontweight="bold", color=INK,
            ha="left", va="center")
    if subtitle:
        ax.text(0.6, height - 1.35, subtitle, fontsize=15, color=INK2,
                ha="left", va="center")


def _image(ax, img, x: float, y: float, w: float, h: float, pixelated: bool = False,
           frame: bool = True) -> None:
    """Pune o imagine în dreptunghiul (x, y, w, h) din unități de slide."""
    ax.imshow(img, extent=(x, x + w, y, y + h), aspect="auto", zorder=2,
              interpolation="nearest" if pixelated else "antialiased")
    if frame:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=BASE, lw=1.2, zorder=3))
    _reset(ax, ax.get_ylim()[1])


def _arrow(ax, x0, y0, x1, y1, color=MUTED, lw=3) -> None:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=26, color=color, lw=lw, zorder=3))


def _hbars(ax, items, x, y_top, width, bar_h=0.26, row_h=0.62, label_size=13.5,
           value_size=13.5, scale_frac=0.78) -> None:
    """Bare orizontale desenate manual: items = [(nume, probabilitate, culoare)].

    Numele sta deasupra barei; procentul la capătul ei. Bara de 100% ocupă
    scale_frac din lățime, ca să rămână loc pentru procent.
    """
    for i, (name, p, color) in enumerate(items):
        y = y_top - i * row_h
        ax.text(x, y, name, fontsize=label_size, color=INK, ha="left", va="bottom")
        w = max(p, 0.004) * width * scale_frac
        ax.add_patch(Rectangle((x, y - bar_h - 0.04), w, bar_h, fc=color, ec="none", zorder=2))
        label = "<1%" if p < 0.005 else pct(p, 0)
        ax.text(x + w + 0.08, y - 0.04 - bar_h / 2, label, fontsize=value_size,
                color=INK, ha="left", va="center")


def _save(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=BG)
    plt.close(fig)


def _style_axes(ax) -> None:
    """Fără chenar sus/dreapta, axă de bază discretă, etichete lizibile."""
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=13, length=0)
    ax.tick_params(axis="y", labelcolor=INK)


# --- 1. Pipeline ---
def plot_pipeline(original: np.ndarray, cropped: np.ndarray, model_input: np.ndarray,
                  top3: list[tuple[str, float]], correct_name: str, n_params: int,
                  out_path: Path) -> None:
    """Pașii de la imaginea din test la probabilitățile top-3."""
    fig, ax = _canvas()
    _title(ax, "Rețeaua transformă pixelii în probabilități și alege semnul",
           "Cinci pași pentru un exemplu real din setul de test")
    cy, s = 4.5, 2.2
    steps = [(0.6, original, "1. Fotografia\noriginală", False),
             (3.4, cropped, "2. Decupăm\nsemnul", False),
             (6.2, model_input, "3. Micșorăm la 48×48\n(ce vede rețeaua)", True)]
    for x, img, cap, pix in steps:
        _image(ax, img, x, cy - s / 2, s, s, pixelated=pix)
        ax.text(x + s / 2, cy - s / 2 - 0.2, cap, fontsize=15, color=INK, ha="center",
                va="top", linespacing=1.3)
        h, w = np.asarray(img).shape[:2]
        ax.text(x + s / 2, cy + s / 2 + 0.15, f"{w}×{h} px", fontsize=13, color=MUTED,
                ha="center", va="bottom")
    for x0 in (2.85, 5.65, 8.5):
        _arrow(ax, x0, cy, x0 + 0.4, cy)
    bx, bw, bh = 9.1, 2.4, 3.0
    ax.add_patch(FancyBboxPatch((bx, cy - bh / 2), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=BLUE, ec="none", zorder=2))
    ax.text(bx + bw / 2, cy + 0.5, "4. CNN propriu", fontsize=18, fontweight="bold",
            color="white", ha="center", va="center", zorder=3)
    ax.text(bx + bw / 2, cy - 0.35, f"{fmt_int(n_params)}\nde parametri", fontsize=16,
            color="white", ha="center", va="center", linespacing=1.3, zorder=3)
    _arrow(ax, 11.6, cy, 12.0, cy)
    items = [(f"{n}", p, BLUE if n == correct_name else GRAY) for n, p in top3]
    ax.text(12.2, cy + 1.9, "5. Cele mai probabile\n3 semne", fontsize=15, color=INK,
            ha="left", va="center", linespacing=1.3)
    _hbars(ax, items, 12.2, cy + 0.95, 3.4, bar_h=0.3, row_h=1.0, label_size=14,
           value_size=15, scale_frac=0.72)
    ax.text(0.6, 0.75, "Albastru = semnul corect. Rețeaua este sigură pe răspuns când o singură bară domină.",
            fontsize=14, color=INK2, ha="left", va="center")
    _save(fig, out_path)


# --- 2. Predicții ---
def plot_predictions(items: list[dict], out_path: Path, title: str, subtitle: str) -> None:
    """Grilă 2x4: imagine + top-3 probabilități.

    item = {image, top3: [(nume, p)], correct: bool, true_name: str}
    """
    fig, ax = _canvas()
    _title(ax, title, subtitle)
    cols, cell_w, cell_h = 4, 3.9, 3.35
    x0, y_top = 0.45, 7.35
    for k, it in enumerate(items[:8]):
        r, c = divmod(k, cols)
        cx = x0 + c * (cell_w + 0.0) + cell_w / 2
        top = y_top - r * cell_h
        mark = "✔ corect" if it["correct"] else "✘ greșit"
        ax.text(cx, top - 0.05, mark, fontsize=16, fontweight="bold", color=INK,
                ha="center", va="top")
        s = 1.0
        _image(ax, it["image"], cx - s / 2, top - 0.45 - s, s, s, pixelated=True)
        bars = []
        for name, p in it["top3"]:
            if name == it["true_name"]:
                col = BLUE
            elif not it["correct"] and name == it["top3"][0][0]:
                col = ORANGE
            else:
                col = GRAY
            bars.append((textwrap.shorten(name, 27, placeholder="…"), p, col))
        _hbars(ax, bars, cx - cell_w / 2 + 0.25, top - 1.95, cell_w - 0.5, bar_h=0.16,
               row_h=0.42, label_size=12.5, value_size=12.5, scale_frac=0.78)
        if not it["correct"]:
            ax.text(cx, top - 3.25, "era: " + textwrap.shorten(it["true_name"], 30, placeholder="…"),
                    fontsize=12.5, color=INK2, ha="center", va="center")
    ax.text(0.6, 0.3, "Albastru = semnul corect · portocaliu = răspunsul greșit al modelului · gri = alte variante",
            fontsize=13, color=INK2, ha="left", va="center")
    _save(fig, out_path)


# --- 3. Acuratețea pe clasă ---
def plot_per_class(per_class: pd.DataFrame, metas: dict[int, np.ndarray], out_path: Path,
                   n_weak: int = 5, axis_min: float = 0.70) -> None:
    """Recall pe fiecare clasă, două coloane (sortat crescător), cele mai slabe în portocaliu."""
    df = per_class.sort_values(["recall", "class_id"]).reset_index(drop=True)
    weak_ids = set(df["class_id"].iloc[:n_weak])
    mean_recall = float(df["recall"].mean())
    fig, ax = _canvas()
    n_low = int((df["recall"] < 0.8).sum())
    _title(ax, f"Aproape toate semnele sunt recunoscute; doar {n_low} cad sub 80%",
           "Acuratețea pe fiecare tip de semn (din 100 de semne de acest tip, câte sunt recunoscute)")
    n = len(df)
    left_n = (n + 1) // 2
    row_h, y_top = 0.29, 7.0
    cols = [(0.5, 0, left_n), (8.3, left_n, n)]
    thumb = 0.25
    for cx, a, b in cols:
        bx0, bw = cx + 3.3, 3.55
        for gv in np.arange(axis_min, 1.0001, 0.1):
            gx = bx0 + (gv - axis_min) / (1 - axis_min) * bw
            ax.plot([gx, gx], [y_top - (b - a) * row_h, y_top + 0.08], color=GRID, lw=1, zorder=0)
            ax.text(gx, y_top + 0.15, f"{gv * 100:.0f}%", fontsize=12, color=MUTED,
                    ha="center", va="bottom")
        for j, i in enumerate(range(a, b)):
            row = df.iloc[i]
            cid = int(row["class_id"])
            y = y_top - (j + 0.5) * row_h
            ax.imshow(metas[cid], extent=(cx, cx + thumb, y - thumb / 2, y + thumb / 2),
                      aspect="auto", zorder=2)
            ax.text(cx + thumb + 0.1, y, short_name(cid), fontsize=12.5, color=INK,
                    ha="left", va="center")
            weak = cid in weak_ids
            w = max((row["recall"] - axis_min) / (1 - axis_min), 0.0) * bw
            ax.add_patch(Rectangle((bx0, y - 0.085), w, 0.17, fc=ORANGE if weak else BLUE,
                                   ec="none", zorder=2))
            if weak:
                ax.text(bx0 + w + 0.08, y, pct(row["recall"]), fontsize=13, color=INK,
                        fontweight="bold", ha="left", va="center")
    _reset(ax)
    ax.text(0.6, 0.3, f"Atenție: axa începe de la {axis_min * 100:.0f}%, nu de la 0. "
            f"Media pe cele 43 de tipuri: {pct(mean_recall)}. Sortat de la cele mai slabe la cele mai bune.",
            fontsize=13.5, color=INK2, ha="left", va="center")
    _save(fig, out_path)


# --- 4. Experimente ---
def plot_experiments(rows: list[tuple[str, float, bool]], out_path: Path,
                     axis_min: float = 0.85, target: float = 0.95) -> None:
    """Acuratețe pe test; rows = (nume, acuratețe, test_pe_semne_decupate)."""
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    fig.text(0.6 / W, 1 - 0.75 / H, "Modelul merge bine doar pe felul de imagini pe care a învățat",
             fontsize=23, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.6 / W, 1 - 1.35 / H, "Acuratețea pe setul de test pentru fiecare experiment",
             fontsize=15, color=INK2, ha="left", va="center")
    ax = fig.add_axes([4.6 / W, 1.5 / H, 10.4 / W, 5.6 / H])
    _style_axes(ax)
    y = np.arange(len(rows))[::-1]
    for yi, (name, acc, cropped) in zip(y, rows):
        ax.barh(yi, acc, height=0.5, color=BLUE if cropped else ORANGE, zorder=2)
        ax.text(acc - 0.003, yi, pct(acc, 2), fontsize=17, fontweight="bold", color="white",
                ha="right", va="center", zorder=3)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=15, pad=10)
    ax.set_xlim(axis_min, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(np.arange(0.85, 1.001, 0.05), [f"{v * 100:.0f}%" for v in np.arange(0.85, 1.001, 0.05)])
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.axvline(target, color=INK2, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(target + 0.002, len(rows) - 0.45, "ținta: 95%", fontsize=14, color=INK2,
            ha="left", va="center")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=BLUE), Rectangle((0, 0), 1, 1, fc=ORANGE)],
              labels=["testat pe semne decupate", "testat pe imagini nedecupate"],
              loc="lower left", bbox_to_anchor=(0.0, -0.2), ncol=2, fontsize=14, frameon=False)
    fig.text(0.6 / W, 0.55 / H, f"Atenție: axa începe de la {axis_min * 100:.0f}%, nu de la 0.",
             fontsize=13.5, color=INK2, ha="left", va="center")
    _save(fig, out_path)


# --- 5. Supraînvățare ---
def plot_overfitting(base: pd.DataFrame, no_reg: pd.DataFrame, out_path: Path,
                     ylim: tuple[float, float] = (0.90, 1.0)) -> None:
    """Acuratețe antrenare vs. validare pentru două run-uri, aceeași axă y."""
    gap_noreg = (no_reg["accuracy"].iloc[-1] - no_reg["val_accuracy"].iloc[-1]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True, facecolor=BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.76, bottom=0.14, wspace=0.08)
    fig.text(0.6 / W, 1 - 0.75 / H,
             "Fără regularizare, modelul «tocește» datele de antrenare",
             fontsize=23, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.6 / W, 1 - 1.35 / H,
             f"Fără regularizare, antrenarea ajunge la 100%, dar validarea rămâne cu {gap_noreg:.1f} puncte mai jos".replace(".", ","),
             fontsize=15, color=INK2, ha="left", va="center")
    for ax, hist, name in ((axes[0], base, "De bază (cu regularizare)"),
                           (axes[1], no_reg, "Fără regularizare")):
        _style_axes(ax)
        ep = hist["epoch"].to_numpy()
        tr, va = hist["accuracy"].to_numpy(), hist["val_accuracy"].to_numpy()
        ax.fill_between(ep, tr, va, facecolor=GRID, edgecolor=BASE, hatch="///",
                        lw=0, alpha=0.9, zorder=1)
        ax.plot(ep, tr, color=BLUE, lw=3, label="antrenare", zorder=3)
        ax.plot(ep, va, color=ORANGE, lw=3, label="validare", zorder=3)
        ax.set_title(name, fontsize=18, fontweight="bold", color=INK, loc="left", pad=14)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, ep.max() + 1)
        ax.set_xlabel("Epoca", fontsize=14, color=INK2)
        ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
        gap = (tr[-1] - va[-1]) * 100
        mid = (tr[-1] + va[-1]) / 2
        txt = f"decalaj: {abs(gap):.1f} puncte procentuale".replace(".", ",")
        if gap < 0:
            txt += "\n(validarea e chiar mai sus)"
        ax.annotate(txt, xy=(ep.max(), mid), xytext=(ep.max() * 0.7, 0.962),
                    fontsize=14, color=INK, ha="center", va="center", linespacing=1.3,
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.8,
                                    connectionstyle="arc3,rad=0"), zorder=5)
    axes[0].set_yticks(np.arange(0.90, 1.001, 0.02),
                       [f"{v * 100:.0f}%" for v in np.arange(0.90, 1.001, 0.02)])
    axes[0].set_ylabel("Acuratețe", fontsize=14, color=INK2)
    axes[0].legend(loc="lower right", fontsize=14, frameon=False)
    fig.text(0.6 / W, 0.45 / H, f"Axa verticală pornește de la {ylim[0] * 100:.0f}% (primele epoci, sub această valoare, nu apar). "
             "Zona hașurată = decalajul dintre antrenare și validare.",
             fontsize=13, color=INK2, ha="left", va="center")
    _save(fig, out_path)


# --- 6. Confuzii ---
def plot_confusions(pairs: list[tuple[int, int, int, float]], metas: dict[int, np.ndarray],
                    names: dict[int, str], out_path: Path, title: str, subtitle: str) -> None:
    """Primele perechi confundate: semn real -> semn prezis."""
    fig, ax = _canvas()
    _title(ax, title, subtitle)
    for x, label, ha in ((0.6, "Semnul real", "left"), (7.35, "Modelul a răspuns", "left"),
                         (15.4, "Cât de des", "right")):
        ax.text(x, 7.2, label, fontsize=14, color=MUTED, ha=ha, va="center")
    ax.plot([0.6, 15.4], [6.95, 6.95], color=BASE, lw=1)
    row_h, s = 1.2, 0.95
    for k, (t, q, n, frac) in enumerate(pairs):
        cy = 6.3 - k * row_h
        _image(ax, metas[t], 0.6, cy - s / 2, s, s)
        ax.text(1.75, cy, textwrap.fill(names[t], 30), fontsize=15, color=INK, ha="left",
                va="center", linespacing=1.25)
        _arrow(ax, 5.55, cy, 6.55, cy, color=ORANGE, lw=3.5)
        _image(ax, metas[q], 7.35, cy - s / 2, s, s)
        ax.text(8.5, cy, textwrap.fill(names[q], 30), fontsize=15, color=INK, ha="left",
                va="center", linespacing=1.25)
        ax.text(15.4, cy + 0.17, f"{n} imagini", fontsize=19, fontweight="bold", color=INK,
                ha="right", va="center")
        ax.text(15.4, cy - 0.27, f"({frac * 100:.0f}% din clasa reală)", fontsize=13.5,
                color=INK2, ha="right", va="center")
    _save(fig, out_path)


# --- 7. Setul de date ---
def plot_dataset(stats: dict, metas: list[np.ndarray], out_path: Path) -> None:
    """Patru „stat tiles” + un rând de semne oficiale."""
    total = stats["n_train"] + stats["n_test"]
    fig, ax = _canvas()
    _title(ax, f"Peste {fmt_int(total // 1000 * 1000)} de fotografii de semne, în {stats['n_classes']} de tipuri",
           "Setul de date GTSRB, pe scurt")
    tiles = [(fmt_int(stats["n_train"]), "imagini de antrenare"),
             (fmt_int(stats["n_test"]), "imagini de test"),
             (str(stats["n_classes"]), "tipuri de semne"),
             (f"~{stats['avg_px']:.0f}×{stats['avg_px']:.0f}", "pixeli, în medie")]
    tw, gap, th, ty = 3.4, 0.4, 2.7, 4.5
    for i, (num, lab) in enumerate(tiles):
        x = 0.6 + i * (tw + gap)
        ax.add_patch(FancyBboxPatch((x, ty), tw, th, boxstyle="round,pad=0.0,rounding_size=0.18",
                                    fc=TILE, ec=GRID, lw=1.5, zorder=1))
        ax.text(x + tw / 2, ty + th * 0.6, num, fontsize=44, fontweight="bold", color=INK,
                ha="center", va="center", zorder=2)
        ax.text(x + tw / 2, ty + th * 0.23, lab, fontsize=17, color=INK2, ha="center",
                va="center", zorder=2)
    ax.text(0.6, 3.7, "Așa arată semnele oficiale (câte unul din 12 tipuri)", fontsize=16,
            color=INK, ha="left", va="center")
    s, g = 1.0, (14.8 - 12 * 1.0) / 11
    for i, img in enumerate(metas[:12]):
        x = 0.6 + i * (s + g)
        ax.imshow(img, extent=(x, x + s, 1.7, 1.7 + s), aspect="auto", zorder=2)
    _reset(ax)
    _save(fig, out_path)
