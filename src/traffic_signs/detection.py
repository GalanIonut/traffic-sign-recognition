"""Conversia GTSDB (detectie) in format YOLO cu o singura clasa."""
import shutil
from pathlib import Path

import pandas as pd
import yaml

SPLITS: dict[str, range] = {
    "train": range(0, 500),
    "val": range(500, 600),   # bloc contiguu: cadre vecine nu ajung in ambele seturi
    "test": range(600, 900),  # testul oficial GTSDB
}
CLASS_NAME = "semn"


def load_gt(path: Path) -> pd.DataFrame:
    """Citeste gt.txt (`NNNNN.ppm;x1;y1;x2;y2;clasa`); tolereaza CRLF, spatii, linii goale."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name, x1, y1, x2, y2, class_id = [f.strip() for f in line.split(";")]
        rows.append((name.split(".")[0], int(x1), int(y1), int(x2), int(y2), int(class_id)))
    df = pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "class_id"])
    df["image_id"] = df["image_id"].astype(str)
    return df


def to_yolo_box(x1: int, y1: int, x2: int, y2: int,
                width: int, height: int) -> tuple[float, float, float, float]:
    """Colturi in pixeli (incluse) -> (cx, cy, w, h) normalizate."""
    cx = (x1 + x2 + 1) / 2 / width
    cy = (y1 + y2 + 1) / 2 / height
    w = (x2 - x1 + 1) / width
    h = (y2 - y1 + 1) / height
    return cx, cy, w, h


def yolo_label_lines(boxes: pd.DataFrame, width: int, height: int) -> list[str]:
    """O linie `0 cx cy w h` per patrat; clasa GTSRB nu intra in etichete."""
    lines = []
    for r in boxes.itertuples():
        cx, cy, w, h = to_yolo_box(r.x1, r.y1, r.x2, r.y2, width, height)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def split_of(image_number: int) -> str:
    """Numele setului (train/val/test) pentru numarul pozei."""
    for name, rng in SPLITS.items():
        if image_number in rng:
            return name
    raise ValueError(f"Poza {image_number} nu apartine niciunui set")


def _find_image(images_dir: Path, number: int) -> Path:
    name = f"{number:05d}.jpg"
    for sub in ("Train", "Test"):
        candidate = Path(images_dir) / sub / "images" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Poza {name} nu exista in {images_dir}/Train|Test/images")


def build_yolo_dataset(gt: pd.DataFrame, images_dir: Path,
                       out_dir: Path) -> dict[str, dict[str, int]]:
    """Recreeaza out_dir cu cele 900 de poze + etichete (goale pentru poze fara semne)."""
    from traffic_signs.config import GTSDB_IMAGE_SIZE
    width, height = GTSDB_IMAGE_SIZE
    out_dir = Path(out_dir)
    bad_ids = sorted({i for i in gt["image_id"] if not 0 <= int(i) <= 899})
    if bad_ids:
        raise ValueError(f"image_id in afara intervalului 0-899: {bad_ids}")
    if out_dir.exists() and any(out_dir.iterdir()) and not (out_dir / "dataset.yaml").exists():
        raise ValueError(f"{out_dir} nu e gol si nu contine dataset.yaml: refuz sa il sterg")
    # Verificam toate pozele inainte sa stergem ceva: eroare clara, nu set incomplet
    sources = {n: _find_image(images_dir, n) for rng in SPLITS.values() for n in rng}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    by_image = {k: g for k, g in gt.groupby("image_id")}
    counts = {s: {"images": 0, "boxes": 0, "empty": 0} for s in SPLITS}
    for split, rng in SPLITS.items():
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)
        for n in rng:
            key = f"{n:05d}"
            shutil.copy2(sources[n], out_dir / "images" / split / f"{key}.jpg")
            lines = yolo_label_lines(by_image[key], width, height) if key in by_image else []
            text = "\n".join(lines) + "\n" if lines else ""
            (out_dir / "labels" / split / f"{key}.txt").write_text(text)
            counts[split]["images"] += 1
            counts[split]["boxes"] += len(lines)
            counts[split]["empty"] += 0 if lines else 1

    cfg = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": {0: CLASS_NAME}}
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return counts


def train_args(data_yaml: Path, run_name: str, epochs: int = 100, imgsz: int = 1024,
               batch: int = 8, device: str = "mps", patience: int = 20) -> dict:
    """Argumentele exacte trimise la YOLO.train."""
    from traffic_signs.config import PROJECT_ROOT, SEED
    return {
        "data": str(data_yaml), "epochs": epochs, "imgsz": imgsz, "batch": batch,
        "device": device, "patience": patience, "seed": SEED, "deterministic": True,
        "project": str(PROJECT_ROOT / "runs" / "yolo"), "name": run_name,
        "exist_ok": False, "plots": True,
    }


def summarize_metrics(results) -> dict:
    """Cele 4 metrici de detectie din obiectul de validare Ultralytics."""
    b = results.box
    return {"precision": float(b.mp), "recall": float(b.mr),
            "map50": float(b.map50), "map50_95": float(b.map)}
