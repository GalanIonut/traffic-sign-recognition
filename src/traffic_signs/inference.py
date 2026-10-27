"""Logica de inferenta pentru aplicatia Streamlit: registru de modele, citire imagine, top-k."""
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from traffic_signs import config
from traffic_signs.classes import CLASS_NAMES
from traffic_signs.preprocessing import load_image, preprocess

LOW_CONFIDENCE: float = 0.5


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    filename: str
    image_size: int


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("cnn", "CNN propriu", "cnn_no_crop.keras", config.IMG_SIZE_CNN),
    ModelSpec("resnet50", "ResNet50 (conv4+conv5)", "resnet50_conv4.keras", config.IMG_SIZE_RESNET),
)


@dataclass(frozen=True)
class Prediction:
    class_id: int
    name: str
    confidence: float


class InvalidImageError(ValueError):
    """Fisierul incarcat nu poate fi citit ca imagine (mesaj prietenos, in romana)."""


def available_models(models_dir: Path = config.MODELS_DIR) -> list[ModelSpec]:
    """Specificatiile modelelor al caror fisier exista, in ordinea din MODEL_SPECS."""
    return [s for s in MODEL_SPECS if (Path(models_dir) / s.filename).is_file()]


def read_image(source: str | Path | BinaryIO) -> np.ndarray:
    """Citeste o imagine (cale sau file-like) ca uint8 (H, W, 3); eroare prietenoasa daca nu e valida."""
    try:
        return load_image(source)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as err:
        raise InvalidImageError(
            "Fișierul nu pare să fie o imagine validă. Încearcă un PNG sau JPG."
        ) from err


def classify(model, image: np.ndarray, image_size: int, k: int = 3) -> list[Prediction]:
    """Top-k predictii (descrescator dupa incredere) pentru o imagine RGB uint8."""
    batch = preprocess(image, image_size)[None, ...]
    probs = np.asarray(model(batch, training=False))[0]
    top = np.argsort(-probs, kind="stable")[:k]
    return [Prediction(int(i), CLASS_NAMES[int(i)], float(probs[i])) for i in top]


def is_uncertain(prediction: Prediction) -> bool:
    return prediction.confidence < LOW_CONFIDENCE


def format_percent(x: float) -> str:
    """0.9734 -> '97,3%' (o zecimala, virgula romaneasca)."""
    return f"{x * 100:.1f}".replace(".", ",") + "%"


def meta_image_path(class_id: int, assets_dir: Path = config.APP_ASSETS_DIR) -> Path | None:
    path = Path(assets_dir) / "meta" / f"{class_id}.png"
    return path if path.is_file() else None


def sample_images(assets_dir: Path = config.APP_ASSETS_DIR) -> list[Path]:
    return sorted((Path(assets_dir) / "samples").glob("*.png"), key=lambda p: p.name)


def sample_class_id(path: Path) -> int:
    """class_14.png -> 14."""
    return int(Path(path).stem.split("_")[1])
