"""Constante globale ale proiectului."""
from pathlib import Path

# Radacina proiectului (src/traffic_signs/config.py -> sus 3 niveluri)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "archive"
SPLITS_DIR = PROJECT_ROOT / "splits"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

SEED = 42
NUM_CLASSES = 43
IMG_SIZE_CNN = 48
IMG_SIZE_RESNET = 96
BATCH_SIZE = 64
VAL_FRACTION = 0.15
MODELS_DIR = PROJECT_ROOT / "models"
APP_ASSETS_DIR = PROJECT_ROOT / "app" / "assets"
GTSDB_DIR = PROJECT_ROOT / "data" / "gtsdb"
GTSDB_YOLO_DIR = PROJECT_ROOT / "data" / "gtsdb_yolo"
GTSDB_IMAGE_SIZE = (1360, 800)  # (latime, inaltime)
