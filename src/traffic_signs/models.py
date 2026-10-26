"""CNN propriu pentru semne de circulatie + sumar text si diagrama."""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from tensorflow import keras
from tensorflow.keras import layers

from traffic_signs.config import IMG_SIZE_CNN, NUM_CLASSES

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def build_cnn(
    input_size: int = IMG_SIZE_CNN,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.5,
    batch_norm: bool = True,
) -> keras.Model:
    """3 blocuri convolutionale + clasificator dens; BN si Dropout optionale."""
    inputs = keras.Input((input_size, input_size, 3), name="input")
    x = inputs
    # (nr. convolutii, filtre) per bloc
    for block, (n_convs, filters) in enumerate([(2, 32), (2, 64), (1, 128)], start=1):
        for i in range(1, n_convs + 1):
            x = layers.Conv2D(
                filters, 3, padding="same", activation="relu", name=f"block{block}_conv{i}"
            )(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"block{block}_bn")(x)
        x = layers.MaxPooling2D(2, name=f"block{block}_pool")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="fc")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return keras.Model(inputs, outputs, name="cnn_propriu")


def model_summary_text(model: keras.Model) -> str:
    """Tabelul `model.summary()` ca text simplu (fara coduri ANSI)."""
    lines: list[str] = []
    model.summary(print_fn=lambda line, **_: lines.append(_ANSI.sub("", line)))
    return "\n".join(lines)


# (eticheta in legenda, culoare caseta) per tip de strat
_KINDS = {
    "input": ("Intrare", "#d9d9d9"),
    "conv": ("Convolutie", "#9ecae1"),
    "pool": ("Max pooling", "#fdd0a2"),
    "norm": ("Normalizare (BatchNorm)", "#c7e9c0"),
    "flatten": ("Flatten", "#fff2a8"),
    "dense": ("Dens", "#bcbddc"),
    "dropout": ("Dropout", "#fcbba1"),
}
_TYPE_NAMES = {
    "input": "Intrare",
    "conv": "Conv2D 3×3, same",
    "pool": "MaxPooling2D 2×2",
    "norm": "BatchNormalization",
    "flatten": "Flatten",
    "dense": "Dense",
    "dropout": "Dropout",
}


def _kind(layer: keras.layers.Layer) -> str:
    for cls, kind in [
        (layers.InputLayer, "input"),
        (layers.Conv2D, "conv"),
        (layers.MaxPooling2D, "pool"),
        (layers.BatchNormalization, "norm"),
        (layers.Flatten, "flatten"),
        (layers.Dense, "dense"),
        (layers.Dropout, "dropout"),
    ]:
        if isinstance(layer, cls):
            return kind
    return "dense"


def _shape_text(layer: keras.layers.Layer) -> str:
    shape = layer.output.shape[1:]
    return "×".join(str(d) for d in shape)


def plot_architecture(model: keras.Model, out_path: Path) -> None:
    """Diagrama verticala: o caseta per strat (tip, forma iesirii, activare)."""
    items = []
    for layer in model.layers:
        kind = _kind(layer)
        title = _TYPE_NAMES[kind]
        if kind == "conv":
            title = f"Conv2D {layer.filters} filtre 3×3, same"
        elif kind == "dense":
            title = f"Dense {layer.units} neuroni"
        elif kind == "dropout":
            title = f"Dropout (rata {layer.rate:g})"
        act = getattr(layer, "activation", None)
        act_name = act.__name__ if act is not None and act.__name__ != "linear" else None
        detail = f"iesire: {_shape_text(layer)}"
        if act_name:
            detail += f"   |   activare: {act_name}"
        items.append((layer.name, kind, title, detail))

    n = len(items)
    step, box_h, box_w = 1.0, 0.7, 7.0
    fig_h = n * step * 0.62 + 3.8
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.set_xlim(-box_w / 2 - 1.2, box_w / 2 + 1.2)
    ax.set_ylim(-(n - 1) * step - 2.8, 1.9)
    ax.axis("off")

    for i, (name, kind, title, detail) in enumerate(items):
        y = -i * step
        color = _KINDS[kind][1]
        ax.add_patch(
            FancyBboxPatch(
                (-box_w / 2, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=color, edgecolor="#333333", linewidth=1.0,
            )
        )
        ax.text(0, y + 0.14, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(0, y - 0.17, detail, ha="center", va="center", fontsize=9.5)
        ax.text(-box_w / 2 - 0.1, y, name, ha="right", va="center", fontsize=8, color="#555555")
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (0, y - box_h / 2 - 0.02), (0, y - step + box_h / 2 + 0.02),
                    arrowstyle="-|>", mutation_scale=10, color="#333333",
                )
            )

    ax.text(0, 1.45, "Arhitectura CNN propriu", ha="center", fontsize=16, fontweight="bold")
    ax.text(
        0, 1.0, f"Total parametri: {model.count_params():,}".replace(",", "."),
        ha="center", fontsize=11, color="#444444",
    )
    used = [k for k in _KINDS if any(it[1] == k for it in items)]
    handles = [
        Patch(facecolor=_KINDS[k][1], edgecolor="#333333", label=_KINDS[k][0]) for k in used
    ]
    ax.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
