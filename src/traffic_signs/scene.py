"""Lantul scena -> YOLO (detectie) -> decupaj -> CNN (clasificare), potrivire cu GT si metrici."""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from traffic_signs import config
from traffic_signs.classes import CLASS_NAMES
from traffic_signs.inference import Prediction
from traffic_signs.preprocessing import preprocess

Box = tuple[float, float, float, float]
DETECT_FLOOR: float = 0.05
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float


@dataclass(frozen=True)
class SceneSign:
    number: int
    box: Box
    det_score: float
    top: tuple[Prediction, ...]

    @property
    def best(self) -> Prediction:
        return self.top[0]


@dataclass(frozen=True)
class GroundTruth:
    box: Box
    class_id: int


@dataclass(frozen=True)
class PipelineConfig:
    classifier: str  # numele fisierului .keras
    image_size: int
    margin: float
    conf: float


def load_pipeline_config(path: Path = config.PIPELINE_CONFIG) -> PipelineConfig:
    d = json.loads(Path(path).read_text())
    return PipelineConfig(str(d["classifier"]), int(d["image_size"]),
                          float(d["margin"]), float(d["conf"]))


def save_pipeline_config(cfg: PipelineConfig, path: Path = config.PIPELINE_CONFIG) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n")


def crop_with_margin(image: np.ndarray, box: Box, margin: float) -> np.ndarray:
    """Decupeaza box-ul extins cu `margin` pe fiecare parte, taiat la imagine; minim 1x1 px."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = margin * (x2 - x1), margin * (y2 - y1)
    left = min(max(math.floor(x1 - dx), 0), w - 1)
    top = min(max(math.floor(y1 - dy), 0), h - 1)
    right = max(min(math.ceil(x2 + dx), w), left + 1)
    bottom = max(min(math.ceil(y2 + dy), h), top + 1)
    return image[top:bottom, left:right]


def detect(yolo, image: np.ndarray, conf: float = DETECT_FLOOR, device: str = "cpu",
           imgsz: int = 1024) -> list[Detection]:
    """Detectii YOLO (imagine RGB -> BGR), descrescator dupa scor."""
    result = yolo.predict(np.ascontiguousarray(image[..., ::-1]), conf=conf, device=device,
                          imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes.xyxy.tolist()
    scores = result.boxes.conf.tolist()
    dets = [Detection(tuple(float(v) for v in b), float(s)) for b, s in zip(boxes, scores)]
    return sorted(dets, key=lambda d: -d.score)


def warm_up_yolo(yolo, device: str = "cpu", imgsz: int = 1024) -> None:
    """O predictie pe o poza goala, ca prima poza reala sa nu includa initializarea."""
    detect(yolo, np.zeros((800, 1360, 3), np.uint8), device=device, imgsz=imgsz)


def classify_crops(model, crops: list[np.ndarray], image_size: int,
                   k: int = 3) -> list[tuple[Prediction, ...]]:
    """Top-k pentru toate decupajele, intr-un singur lot."""
    if not crops:
        return []
    batch = np.stack([preprocess(c, image_size) for c in crops])
    probs = np.asarray(model(batch, training=False))
    tops = []
    for p in probs:
        idx = np.argsort(-p, kind="stable")[:k]
        tops.append(tuple(Prediction(int(i), CLASS_NAMES[int(i)], float(p[i])) for i in idx))
    return tops


def build_signs(detections: list[Detection],
                tops: list[tuple[Prediction, ...]]) -> list[SceneSign]:
    return [SceneSign(i, d.box, d.score, t)
            for i, (d, t) in enumerate(zip(detections, tops), start=1)]


def run_pipeline(yolo, model, image: np.ndarray, cfg: PipelineConfig,
                 device: str = "cpu") -> tuple[list[SceneSign], dict[str, float]]:
    """Detecteaza la DETECT_FLOOR (filtrarea dupa cfg.conf se face separat) si clasifica."""
    t0 = time.perf_counter()
    dets = detect(yolo, image, conf=DETECT_FLOOR, device=device)
    t1 = time.perf_counter()
    crops = [crop_with_margin(image, d.box, cfg.margin) for d in dets]
    tops = classify_crops(model, crops, cfg.image_size)
    t2 = time.perf_counter()
    timings = {"detect_ms": (t1 - t0) * 1000, "classify_ms": (t2 - t1) * 1000}
    return build_signs(dets, tops), timings


def filter_by_conf(signs: list[SceneSign], conf: float) -> list[SceneSign]:
    return [s for s in signs if s.det_score >= conf]


def iou(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[tuple[int, int], ...]  # (indice semn, indice GT)
    false_signs: tuple[int, ...]
    missed: tuple[int, ...]


def match(signs: list[SceneSign], gts: list[GroundTruth], iou_thr: float = 0.5) -> MatchResult:
    """Potrivire greedy dupa scor descrescator; fiecare GT cel mult o data."""
    used: set[int] = set()
    pairs, false_signs = [], []
    for si in sorted(range(len(signs)), key=lambda i: -signs[i].det_score):
        best_gi, best_iou = -1, iou_thr
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            v = iou(signs[si].box, g.box)
            if v >= best_iou and (best_gi < 0 or v > best_iou):
                best_gi, best_iou = gi, v
        if best_gi >= 0:
            used.add(best_gi)
            pairs.append((si, best_gi))
        else:
            false_signs.append(si)
    missed = [gi for gi in range(len(gts)) if gi not in used]
    return MatchResult(tuple(sorted(pairs)), tuple(sorted(false_signs)), tuple(missed))


def sign_statuses(signs: list[SceneSign], gts: list[GroundTruth],
                  match_result: MatchResult) -> dict[int, str]:
    """number -> 'correct' | 'wrong' | 'false'."""
    out = {signs[i].number: "false" for i in match_result.false_signs}
    for si, gi in match_result.pairs:
        ok = signs[si].best.class_id == gts[gi].class_id
        out[signs[si].number] = "correct" if ok else "wrong"
    return out


@dataclass(frozen=True)
class Counts:
    n_gt: int
    n_det: int
    found: int
    found_correct: int

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.n_gt + other.n_gt, self.n_det + other.n_det,
                      self.found + other.found, self.found_correct + other.found_correct)


def count(signs: list[SceneSign], gts: list[GroundTruth], iou_thr: float = 0.5) -> Counts:
    m = match(signs, gts, iou_thr)
    correct = sum(signs[si].best.class_id == gts[gi].class_id for si, gi in m.pairs)
    return Counts(len(gts), len(signs), len(m.pairs), correct)


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def summarize(counts: Counts) -> dict[str, float]:
    recall = _div(counts.found_correct, counts.n_gt)
    precision = _div(counts.found_correct, counts.n_det)
    return {
        "detection_recall": _div(counts.found, counts.n_gt),
        "classification_accuracy": _div(counts.found_correct, counts.found),
        "e2e_recall": recall,
        "e2e_precision": precision,
        "e2e_f1": _div(2 * precision * recall, precision + recall),
    }


def gt_for_image(gt: pd.DataFrame, image_id: str) -> list[GroundTruth]:
    """Randurile GT ale unei poze; colturile incluse devin exclusive (+1)."""
    rows = gt[gt["image_id"] == image_id]
    return [GroundTruth((float(r.x1), float(r.y1), float(r.x2 + 1), float(r.y2 + 1)),
                        int(r.class_id)) for r in rows.itertuples()]


def choose_config(rows: list[dict]) -> dict:
    """Acuratete maxima (toleranta 0,1 pp), apoi marja minima, apoi cnn_no_crop.keras."""
    top = max(r["accuracy"] for r in rows)
    candidates = [r for r in rows if r["accuracy"] >= top - 0.001]
    return min(candidates, key=lambda r: (r["margin"], r["classifier"] != "cnn_no_crop.keras"))


def best_threshold(per_image: list[tuple[list[SceneSign], list[GroundTruth]]],
                   grid: Sequence[float] = THRESHOLD_GRID) -> float:
    """Pragul cu e2e_f1 maxim; la egalitate cel mai mic."""
    best_t, best_f1 = None, -1.0
    for t in sorted(grid):
        total = Counts(0, 0, 0, 0)
        for signs, gts in per_image:
            total = total + count(filter_by_conf(signs, t), gts)
        f1 = summarize(total)["e2e_f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


COLOR_NEUTRAL = (0x2A, 0x78, 0xD6)
COLOR_CORRECT = (0x1B, 0xAF, 0x7A)
COLOR_WRONG = (0xEB, 0x68, 0x34)
COLOR_MISSED_BG = (0x13, 0x23, 0x3F)
STATUS_COLORS = {"correct": COLOR_CORRECT, "wrong": COLOR_WRONG, "false": COLOR_WRONG}
STATUS_SUFFIX = {"correct": " (corect)", "wrong": " (greșit)", "false": " (nu e semn)"}


def _font(size: int):
    """DejaVu Sans (livrat cu matplotlib) are ă, î, ș, ț; fontul implicit PIL nu le are."""
    try:
        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default(size=size)


def label_text(sign: SceneSign, status: str | None = None) -> str:
    """Eticheta de pe imagine: numar, clasa, incredere CNN; statusul ca sufix (fara emoji)."""
    best = sign.best
    return f"{sign.number} · {best.name} {round(best.confidence * 100)}%{STATUS_SUFFIX.get(status, '')}"


def missed_label(gt: GroundTruth) -> str:
    return f"ratat: {CLASS_NAMES[gt.class_id]}"


def _dotted_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                 color: tuple[int, int, int], width: int, dash: int, gap: int) -> None:
    x1, y1, x2, y2 = box
    for pos in range(x1, x2 + 1, dash + gap):
        end = min(pos + dash, x2)
        draw.line([(pos, y1), (end, y1)], fill=color, width=width)
        draw.line([(pos, y2), (end, y2)], fill=color, width=width)
    for pos in range(y1, y2 + 1, dash + gap):
        end = min(pos + dash, y2)
        draw.line([(x1, pos), (x1, end)], fill=color, width=width)
        draw.line([(x2, pos), (x2, end)], fill=color, width=width)


def _draw_label(draw: ImageDraw.ImageDraw, font, text: str, box: tuple[int, int, int, int],
                bg: tuple[int, int, int], size: tuple[int, int]) -> None:
    """Eticheta deasupra patratului (dedesubt daca nu incape), taiata in interiorul imaginii."""
    w, h = size
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th, pad = right - left, bottom - top, 3
    bw, bh = tw + 2 * pad, th + 2 * pad
    x = min(max(box[0], 0), max(w - bw, 0))
    y = box[1] - bh
    if y < 0:
        y = min(box[3], max(h - bh, 0))
    draw.rectangle([x, y, x + bw - 1, y + bh - 1], fill=bg)
    draw.text((x + pad - left, y + pad - top), text, fill=(255, 255, 255), font=font)


def annotate(image: np.ndarray, signs: list[SceneSign], statuses: dict[int, str] | None = None,
             missed: list[GroundTruth] | None = None) -> np.ndarray:
    """Copie a imaginii RGB cu patrate colorate dupa status si etichete; intrarea ramane intacta."""
    h, w = image.shape[:2]
    pil = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    width = max(2, round(min(h, w) / 250))
    font = _font(max(12, min(h, w) // 40))

    def to_px(b: Box) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = (round(v) for v in b)
        x1, x2 = min(max(x1, 0), w - 1), min(max(x2 - 1, 0), w - 1)
        y1, y2 = min(max(y1, 0), h - 1), min(max(y2 - 1, 0), h - 1)
        return x1, y1, max(x2, x1), max(y2, y1)

    for g in missed or []:
        box = to_px(g.box)
        _dotted_rect(draw, box, (255, 255, 255), width, dash=3 * width + 4, gap=2 * width + 2)
        _draw_label(draw, font, missed_label(g), box, COLOR_MISSED_BG, (w, h))
    for s in signs:
        status = statuses.get(s.number) if statuses else None
        color = STATUS_COLORS.get(status, COLOR_NEUTRAL)
        box = to_px(s.box)
        draw.rectangle(box, outline=color, width=width)
        _draw_label(draw, font, label_text(s, status), box, color, (w, h))
    return np.array(pil)


SCENES_GT_NAME = "scenes_gt.json"


def scene_samples(assets_dir: Path = config.APP_ASSETS_DIR) -> list[Path]:
    """Pozele de scena din `scenes/*.jpg`, sortate."""
    return sorted((Path(assets_dir) / "scenes").glob("*.jpg"))


def write_scene_gt(assets_dir: Path, mapping: dict[str, list[GroundTruth]]) -> None:
    """Scrie scenes_gt.json: {stem: [[x1, y1, x2, y2, class_id], ...]} cu colturi exclusive."""
    d = Path(assets_dir) / "scenes"
    d.mkdir(parents=True, exist_ok=True)
    data = {k: [[*g.box, g.class_id] for g in v] for k, v in mapping.items()}
    (d / SCENES_GT_NAME).write_text(json.dumps(data, indent=2) + "\n")


def load_scene_gt(assets_dir: Path = config.APP_ASSETS_DIR) -> dict[str, list[GroundTruth]]:
    path = Path(assets_dir) / "scenes" / SCENES_GT_NAME
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: [GroundTruth((float(r[0]), float(r[1]), float(r[2]), float(r[3])), int(r[4]))
                for r in v] for k, v in raw.items()}


def _min_area(gts: list[GroundTruth]) -> float:
    return min((g.box[2] - g.box[0]) * (g.box[3] - g.box[1]) for g in gts)


def pick_examples(per_image: dict[str, tuple[list[SceneSign], list[GroundTruth]]],
                  k: int = 6) -> list[str]:
    """Alege k poze: cele mai multe semne, cel mai mic semn, o greseala, apoi cele mai mici ID-uri.

    Deterministic; ignora pozele fara semne GT. `signs` sunt cele deja filtrate dupa prag.
    """
    ids = sorted(n for n, (_, gts) in per_image.items() if gts)
    chosen: list[str] = []

    def add(name: str | None) -> None:
        if name is not None and name not in chosen and len(chosen) < k:
            chosen.append(name)

    add(max(ids, key=lambda n: (len(per_image[n][1]), [-ord(c) for c in n]), default=None))
    add(min(ids, key=lambda n: (_min_area(per_image[n][1]), n), default=None))

    def first_with(kind: str) -> str | None:
        for n in ids:
            signs, gts = per_image[n]
            m = match(signs, gts)
            st = sign_statuses(signs, gts, m)
            if (kind == "wrong" and "wrong" in st.values()) \
                    or (kind == "false" and "false" in st.values()) \
                    or (kind == "missed" and m.missed):
                return n
        return None

    add(next((x for x in (first_with("wrong"), first_with("false"), first_with("missed"))
              if x is not None), None))
    for n in ids:
        add(n)
    return chosen
