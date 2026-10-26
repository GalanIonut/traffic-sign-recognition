"""CNN propriu pentru semne de circulatie + sumar text si diagrama."""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from tensorflow import keras
from tensorflow.keras import layers

from traffic_signs.config import IMG_SIZE_CNN, IMG_SIZE_RESNET, NUM_CLASSES

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


@keras.utils.register_keras_serializable(package="traffic_signs")
class ResNetPreprocess(keras.layers.Layer):
    """Preprocesare ImageNet pentru ResNet50: [0, 1] -> [0, 255] -> RGB->BGR, minus medie."""

    def call(self, x):
        return keras.applications.resnet50.preprocess_input(x * 255.0)


def build_resnet50(
    input_size: int = IMG_SIZE_RESNET,
    num_classes: int = NUM_CLASSES,
    weights: str | None = "imagenet",
    dropout: float = 0.3,
) -> keras.Model:
    """ResNet50 cu baza inghetata, preprocesare in model si cap nou de clasificare."""
    base = keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
        pooling="avg",
        name="resnet50",
    )
    base.trainable = False
    inputs = keras.Input((input_size, input_size, 3), name="input")
    x = ResNetPreprocess(name="preprocess")(inputs)
    x = base(x, training=False)  # BN ramane in mod inferenta chiar si dupa dezghetare
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return keras.Model(inputs, outputs, name="resnet50_finetune")


def unfreeze_top(model: keras.Model, prefix: str = "conv5_") -> int:
    """Dezgheata straturile bazei cu numele `prefix*`, fara BatchNormalization."""
    base = model.get_layer("resnet50")
    base.trainable = True  # altfel trainable-ul straturilor interne nu conteaza
    count = 0
    for layer in base.layers:
        unfreeze = layer.name.startswith(prefix) and not isinstance(
            layer, layers.BatchNormalization
        )
        layer.trainable = unfreeze
        count += int(unfreeze)
    return count


def model_summary_text(model: keras.Model) -> str:
    """Tabelul `model.summary()` ca text simplu (fara coduri ANSI)."""
    lines: list[str] = []
    model.summary(print_fn=lambda line, **_: lines.append(_ANSI.sub("", line)))
    return "\n".join(lines)


# (eticheta in legenda, culoare caseta) per tip de strat
_KINDS = {
    "input": ("Intrare", "#d9d9d9"),
    "conv": ("Convoluție", "#9ecae1"),
    "pool": ("Max pooling", "#fdd0a2"),
    "norm": ("Normalizare (BatchNorm)", "#c7e9c0"),
    "flatten": ("Flatten", "#fff2a8"),
    "dense": ("Dens", "#bcbddc"),
    "dropout": ("Dropout", "#fcbba1"),
}
_ACT_NAMES = {"relu": "ReLU", "softmax": "softmax"}


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
    return "×".join(str(d) for d in layer.output.shape[1:])


def _title(layer: keras.layers.Layer, kind: str) -> str:
    if kind == "input":
        return "Imagine RGB"
    if kind == "conv":
        return f"Conv2D {layer.filters}×(3×3)"
    if kind == "pool":
        return "MaxPooling2D 2×2"
    if kind == "norm":
        return "BatchNormalization"
    if kind == "dense":
        return f"Dense {layer.units}"
    if kind == "dropout":
        return f"Dropout (rată {layer.rate:g})".replace(".", ",")
    return "Flatten"


def _column(name: str) -> int:
    """Coloana diagramei: 0 intrare, 1-3 blocurile convolutionale, 4 clasificator."""
    if name == "input":
        return 0
    m = re.match(r"block(\d)_", name)
    return int(m.group(1)) if m else 4


def plot_architecture(model: keras.Model, out_path: Path) -> None:
    """Diagrama orizontala (16:9): coloane Intrare | Bloc 1-3 | Clasificator."""
    heads = ["Intrare", "Bloc 1", "Bloc 2", "Bloc 3", "Clasificator"]
    cols: list[list] = [[] for _ in heads]
    for layer in model.layers:
        kind = _kind(layer)
        act = getattr(layer, "activation", None)
        act_name = act.__name__ if act is not None else "linear"
        act_text = (
            f"activare: {_ACT_NAMES.get(act_name, act_name)}" if act_name != "linear" else None
        )
        cols[_column(layer.name)].append(
            (kind, _title(layer, kind), f"ieșire: {_shape_text(layer)}", act_text)
        )

    col_step, box_w, box_h, row_step = 3.2, 2.85, 1.2, 1.5
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(-box_w / 2 - 0.1, 4 * col_step + box_w / 2 + 0.1)
    ax.set_ylim(-3 * row_step - 2.4, 1.9)
    ax.axis("off")

    boxes = []  # (x, y) centrul fiecarei casete, pe coloane
    for c, items in enumerate(cols):
        x = c * col_step
        ax.text(x, 0.95, heads[c], ha="center", va="center", fontsize=15, fontweight="bold")
        pos = []
        for r, (kind, title, shape, act_text) in enumerate(items):
            y = -r * row_step
            ax.add_patch(
                FancyBboxPatch(
                    (x - box_w / 2, y - box_h / 2), box_w, box_h,
                    boxstyle="round,pad=0.02,rounding_size=0.12",
                    facecolor=_KINDS[kind][1], edgecolor="#333333", linewidth=1.2,
                )
            )
            lines = [(title, 13, "bold", 0.33), (shape, 12, "normal", 0.0)]
            if act_text:
                lines.append((act_text, 12, "normal", -0.33))
            else:
                lines = [(title, 13, "bold", 0.2), (shape, 12, "normal", -0.2)]
            for text, size, weight, dy in lines:
                ax.text(x, y + dy, text, ha="center", va="center", fontsize=size, fontweight=weight)
            if r > 0:
                ax.add_patch(
                    FancyArrowPatch(
                        (x, y + row_step - box_h / 2 - 0.02), (x, y + box_h / 2 + 0.02),
                        arrowstyle="-|>", mutation_scale=16, color="#333333", linewidth=1.5,
                    )
                )
            pos.append((x, y))
        boxes.append(pos)
    for c in range(len(cols) - 1):  # sageti intre coloane
        (x0, y0), (x1, y1) = boxes[c][-1], boxes[c + 1][0]
        ax.add_patch(
            FancyArrowPatch(
                (x0 + box_w / 2 + 0.02, y0), (x1 - box_w / 2 - 0.02, y1),
                arrowstyle="-|>", mutation_scale=16, color="#333333", linewidth=1.5,
            )
        )

    mid = 2 * col_step
    ax.text(mid, 1.75, "Arhitectura CNN propriu", ha="center", fontsize=20, fontweight="bold")
    ax.text(
        mid, 1.35, f"Total parametri: {model.count_params():,}".replace(",", "."),
        ha="center", fontsize=14, color="#444444",
    )
    used = [k for k in _KINDS if any(k == it[0] for col in cols for it in col)]
    handles = [
        Patch(facecolor=_KINDS[k][1], edgecolor="#333333", label=_KINDS[k][0]) for k in used
    ]
    ax.legend(
        handles=handles, loc="lower center", ncol=len(handles), fontsize=12, frameon=False,
        bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=1.4,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
