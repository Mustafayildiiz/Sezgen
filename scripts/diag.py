"""Tanılama: test görsellerinde standart / derin tarama tespit sayılarını ölçer."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detector import FODDetector  # noqa: E402

d = FODDetector()
print("model:", d.model_loaded, d.model_path.name, d.imgsz, d.model_error)
for p in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "test").glob("*.jpg")):
    img = cv2.imread(str(p))
    for deep in (False, True):
        d.update_settings(deep_scan=deep)
        _, dets, _ = d.detect_frame(img)
        print(p.name, img.shape[:2], "deep" if deep else "live", len(dets),
              [(x["class_name"], round(x["confidence"], 2)) for x in dets][:8])

