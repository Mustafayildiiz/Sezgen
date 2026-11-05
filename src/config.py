"""Sezgen sistem yapılandırma ayarları."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
UPLOADS_DIR = BASE_DIR / "uploads"
CONFIGS_DIR = BASE_DIR / "configs"

for _d in (MODELS_DIR, REPORTS_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "best.pt")))
RESULTS_CSV_PATH = Path(os.getenv("RESULTS_CSV_PATH", str(MODELS_DIR / "results.csv")))

# Figma tasarımındaki varsayılan eşik değerleri
DEFAULT_CONF_THRESHOLD = float(os.getenv("DEFAULT_CONF", "0.44"))    # Eşik 1: 44%
DEFAULT_IOU_THRESHOLD = float(os.getenv("DEFAULT_IOU", "0.65"))      # Eşik 2: 65%
DEFAULT_ALARM_THRESHOLD = float(os.getenv("DEFAULT_ALARM_THRESH", "0.85"))  # Eşik 3: 85%

DEFAULT_CAMERA_LABEL = "CAM 04 - RUNWAY 27R"

