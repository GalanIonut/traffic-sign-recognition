"""Preprocesarea unei imagini: incarcare, decupare ROI, redimensionare, normalizare."""
import os
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path | bytes) -> np.ndarray:
    """Incarca o imagine ca uint8 (H, W, 3) RGB; transparenta se compune pe alb."""
    if isinstance(path, bytes):  # tf.numpy_function trimite caile ca bytes
        path = os.fsdecode(path)
    with Image.open(path) as im:
        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        if has_alpha:
            rgba = im.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(background, rgba)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def crop_roi(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Decupeaza ROI inclusiv, limitat la imagine; daca ramane gol, intoarce imaginea."""
    h, w = img.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w - 1), min(y2, h - 1)
    if x2 < x1 or y2 < y1:
        return img
    return img[y1:y2 + 1, x1:x2 + 1]


def preprocess(img: np.ndarray, size: int) -> np.ndarray:
    """Resize biliniar la (size, size) si normalizare in [0, 1] (float32)."""
    resized = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0
