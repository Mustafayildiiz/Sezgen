"""Sezgen YOLO + SAHI (Derin Tarama) FOD Tespit ve Simülasyon Motoru."""
import base64
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    DEFAULT_ALARM_THRESHOLD,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    MODEL_PATH,
)
from src.risk import (
    RISK_COLOR_BGR,
    RISK_YUKSEK,
    risk_of,
    runway_sector_of,
    tr_label_of,
)


def encode_Crop_base64(crop_bgr: np.ndarray, size: Tuple[int, int] = (64, 64)) -> str:
    """Tespit edilen cismin kırpılmış görüntüsünü base64 data URI formatına çevirir."""
    if crop_bgr is None or crop_bgr.size == 0:
        crop_bgr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    resized = cv2.resize(crop_bgr, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def generate_default_thumbnails() -> Dict[str, str]:
    """Figma tasarımındaki 'Metal Parça' ve 'Plastik Parça' için gerçekçi asfalt dokulu küçük görseller üretir."""
    rng = np.random.default_rng(42)

    # 1. Metal Parça (Asfalt üzerinde metalik cıvata/parça dokusu)
    metal = rng.integers(85, 145, size=(64, 64, 3), dtype=np.uint8)
    metal = cv2.GaussianBlur(metal, (3, 3), 0)
    cv2.line(metal, (14, 42), (48, 22), (210, 215, 220), 6, cv2.LINE_AA)
    cv2.circle(metal, (16, 41), 7, (180, 185, 195), -1, cv2.LINE_AA)
    cv2.line(metal, (14, 42), (48, 22), (90, 95, 100), 1, cv2.LINE_AA)

    # 2. Plastik Parça (Asfalt/çakıl üzerinde renkli plastik atık dokusu)
    plastic = rng.integers(75, 140, size=(64, 64, 3), dtype=np.uint8)
    plastic = cv2.GaussianBlur(plastic, (3, 3), 0)
    pts = np.array([[18, 20], [44, 16], [50, 38], [26, 48], [14, 36]], np.int32)
    cv2.fillPoly(plastic, [pts], (65, 155, 215))
    cv2.polylines(plastic, [pts], True, (180, 220, 245), 2, cv2.LINE_AA)

    return {
        "MetalPart": encode_Crop_base64(metal),
        "PlasticPart": encode_Crop_base64(plastic),
    }


class FODDetector:
    """Ultralytics YOLO ve SAHI tabanlı yabancı cisim (FOD) tespit sınıfı."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = Path(model_path or MODEL_PATH)
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.iou_threshold = DEFAULT_IOU_THRESHOLD
        self.alarm_threshold = DEFAULT_ALARM_THRESHOLD
        self.deep_scan = False  # False: Canlı Akış, True: Derin Tarama (SAHI)

        self.model = None
        self.sahi_model = None
        self.model_loaded = False
        self.model_status = "Simülasyon Modu (models/best.pt bekleniyor)"
        self._frame_idx = 0
        self.default_thumbs = generate_default_thumbnails()
        self.try_load_model()

    def try_load_model(self, custom_path: Optional[Path] = None) -> bool:
        """Eğitilen YOLO ağırlık dosyası mevcutsa yükler."""
        if custom_path is not None:
            self.model_path = Path(custom_path)

        if not self.model_path.exists():
            self.model_loaded = False
            self.model_status = f"Simülasyon Modu ({self.model_path.name} bulunamadı)"
            return False

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.model_loaded = True
            self.model_status = f"Aktif Model: {self.model_path.name}"
            return True
        except Exception as exc:
            self.model_loaded = False
            self.model_status = f"Model Yükleme Hatası: {exc}"
            return False

    def update_settings(
        self,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        alarm: Optional[float] = None,
        deep_scan: Optional[bool] = None,
    ) -> None:
        if conf is not None:
            self.conf_threshold = max(0.01, min(0.99, float(conf)))
        if iou is not None:
            self.iou_threshold = max(0.01, min(0.99, float(iou)))
        if alarm is not None:
            self.alarm_threshold = max(0.01, min(0.99, float(alarm)))
        if deep_scan is not None:
            self.deep_scan = bool(deep_scan)

    def render_synthetic_runway(self, width: int = 960, height: int = 720) -> np.ndarray:
        """Figma tasarımındaki gece pist görünümüne uygun gerçekçi pist simülasyon karesi üretir."""
        self._frame_idx += 1
        t = self._frame_idx * 0.05

        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Gökyüzü ve ufuk gradyanı
        horizon_y = int(height * 0.22)
        for y in range(horizon_y):
            v = int(8 + (y / max(1, horizon_y)) * 22)
            img[y, :] = (v + 4, v + 2, v)

        # Zemin / Pist asfalt gradyanı
        for y in range(horizon_y, height):
            ratio = (y - horizon_y) / max(1, height - horizon_y)
            v = int(20 + ratio * 18)
            img[y, :] = (v, v, v)

        # Pist perspektif geometrisi
        vp_x, vp_y = width // 2 + 18, horizon_y
        left_bottom = int(width * 0.26)
        right_bottom = int(width * 0.78)

        runway_poly = np.array(
            [[vp_x - 16, vp_y], [vp_x + 16, vp_y], [right_bottom, height], [left_bottom, height]],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [runway_poly], (32, 34, 36))

        # Yan taksi yolları
        taxi_poly = np.array(
            [[vp_x - 18, vp_y + 20], [vp_x - 8, vp_y + 20], [int(width * 0.05), int(height * 0.62)], [0, int(height * 0.50)]],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [taxi_poly], (38, 40, 42))

        # Pist kenar ve merkez çizgileri
        cv2.line(img, (vp_x - 14, vp_y), (left_bottom + 24, height), (85, 90, 95), 2, cv2.LINE_AA)
        cv2.line(img, (vp_x + 14, vp_y), (right_bottom - 24, height), (85, 90, 95), 2, cv2.LINE_AA)

        # Perspektif pist ışıkları (Merkez yeşil/beyaz, kenar kehribar/beyaz)
        for i in range(1, 18):
            p = ((i + (t % 1.0)) / 18.0) ** 1.85
            y = int(vp_y + p * (height - vp_y))
            spread = int(14 + p * ((right_bottom - left_bottom) // 2 - 24))
            cx = int(vp_x + p * ((left_bottom + right_bottom) // 2 - vp_x))

            radius = max(1, int(1 + p * 4))
            glow_r = radius * 3

            # Merkez hattı ışığı (Yeşilimsi beyaz)
            cv2.circle(img, (cx, y), glow_r, (90, 145, 110), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, y), radius, (220, 255, 235), -1, cv2.LINE_AA)

            # Sol & sağ kenar ışıkları (Sıcak kehribar)
            lx, rx = cx - spread, cx + spread
            cv2.circle(img, (lx, y), glow_r, (60, 115, 165), -1, cv2.LINE_AA)
            cv2.circle(img, (lx, y), radius, (190, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (rx, y), glow_r, (60, 115, 165), -1, cv2.LINE_AA)
            cv2.circle(img, (rx, y), radius, (190, 230, 255), -1, cv2.LINE_AA)

        # Ufukta terminal/apronda şehir ışıkları
        rng = np.random.default_rng(101)
        for _ in range(95):
            lx = int(rng.integers(20, width - 20))
            ly = int(rng.integers(horizon_y - 12, horizon_y + 28))
            lr = int(rng.integers(1, 4))
            cv2.circle(img, (lx, ly), lr, (200, 225, 245), -1, cv2.LINE_AA)

        # Pist üzerindeki simüle edilmiş FOD nesneleri (MetalPart ve PlasticPart)
        # 1. Sol sektörde Metal Parça (Pist 12L)
        m_x = int(width * 0.42 + math.sin(t * 0.4) * 6)
        m_y = int(height * 0.62 + math.cos(t * 0.3) * 4)
        cv2.rectangle(img, (m_x - 14, m_y - 8), (m_x + 14, m_y + 8), (185, 195, 205), -1)
        cv2.circle(img, (m_x - 10, m_y), 5, (220, 225, 230), -1)

        # 2. Sağ sektörde Plastik Parça (Pist 12R)
        p_x = int(width * 0.58 + math.cos(t * 0.35) * 6)
        p_y = int(height * 0.74 + math.sin(t * 0.25) * 4)
        cv2.rectangle(img, (p_x - 12, p_y - 10), (p_x + 12, p_y + 10), (80, 165, 230), -1)

        # 3. Derin Tarama (SAHI) aktifse küçük cıvata (Bolt) tespiti de ekle
        if self.deep_scan:
            b_x = int(width * 0.49)
            b_y = int(height * 0.46)
            cv2.circle(img, (b_x, b_y), 5, (200, 210, 225), -1)

        return img

    def detect_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """Tek bir video karesi üzerinde FOD tespiti yapar ve çizimli kare + tespit listesi + FPS döndürür."""
        t0 = time.perf_counter()
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        detections: List[Dict[str, Any]] = []
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        mode_label = "Derin Tarama" if self.deep_scan else "Canlı Akış"

        if self.model_loaded and self.model is not None:
            if self.deep_scan:
                detections = self._run_sahi_or_sliced(frame_bgr, w, h, now_str, mode_label)
            else:
                results = self.model.predict(
                    source=frame_bgr,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )
                for r in results:
                    names = r.names
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = names.get(cls_id, str(cls_id))
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        norm_x = ((x1 + x2) / 2.0) / max(1, w)
                        detections.append({
                            "timestamp": now_str,
                            "class_name": cls_name,
                            "tr_label": tr_label_of(cls_name),
                            "confidence": round(conf, 3),
                            "risk": risk_of(cls_name),
                            "sector": runway_sector_of(norm_x),
                            "bbox": [x1, y1, x2, y2],
                            "mode": mode_label,
                            "is_alarm": conf >= self.alarm_threshold or risk_of(cls_name) == RISK_YUKSEK,
                            "thumbnail": encode_Crop_base64(crop),
                        })
        else:
            # Simülasyon modu: Eşik değerlerine göre dinamik FOD tespiti
            sim_candidates = [
                {
                    "class_name": "MetalPart",
                    "conf": 0.91,
                    "bbox": [int(w * 0.42) - 22, int(h * 0.62) - 16, int(w * 0.42) + 22, int(h * 0.62) + 16],
                },
                {
                    "class_name": "PlasticPart",
                    "conf": 0.78,
                    "bbox": [int(w * 0.58) - 20, int(h * 0.74) - 18, int(w * 0.58) + 20, int(h * 0.74) + 18],
                },
            ]
            if self.deep_scan:
                sim_candidates.append({
                    "class_name": "Bolt",
                    "conf": 0.68,
                    "bbox": [int(w * 0.49) - 14, int(h * 0.46) - 14, int(w * 0.49) + 14, int(h * 0.46) + 14],
                })

            for item in sim_candidates:
                if item["conf"] >= self.conf_threshold:
                    x1, y1, x2, y2 = item["bbox"]
                    cls_name = item["class_name"]
                    conf = item["conf"]
                    norm_x = ((x1 + x2) / 2.0) / max(1, w)
                    crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    thumb = self.default_thumbs.get(cls_name) or encode_Crop_base64(crop)
                    detections.append({
                        "timestamp": now_str,
                        "class_name": cls_name,
                        "tr_label": tr_label_of(cls_name),
                        "confidence": round(conf, 3),
                        "risk": risk_of(cls_name),
                        "sector": runway_sector_of(norm_x),
                        "bbox": [x1, y1, x2, y2],
                        "mode": mode_label,
                        "is_alarm": conf >= self.alarm_threshold or risk_of(cls_name) == RISK_YUKSEK,
                        "thumbnail": thumb,
                    })

        # Bounding box ve etiketleri kare üzerine çiz
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            risk = det["risk"]
            color = RISK_COLOR_BGR.get(risk, (0, 165, 255))
            label = f"{det['tr_label']} %{int(det['confidence'] * 100)} [{risk}]"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            # Köşe vurguları
            corner_len = min(10, (x2 - x1) // 3, (y2 - y1) // 3)
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 3)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 3)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            bg_y1 = max(0, y1 - th - 10)
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 5, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        elapsed = max(1e-4, time.perf_counter() - t0)
        fps = round(min(60.0, 1.0 / elapsed), 1)
        return annotated, detections, fps

    def _run_sahi_or_sliced(
        self, frame_bgr: np.ndarray, w: int, h: int, now_str: str, mode_label: str
    ) -> List[Dict[str, Any]]:
        """SAHI kütüphanesi ile yüksek çözünürlüklü kareyi dilimleyerek küçük FOD tespiti yapar."""
        detections: List[Dict[str, Any]] = []
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            if self.sahi_model is None:
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=str(self.model_path),
                    confidence_threshold=self.conf_threshold,
                )
            self.sahi_model.confidence_threshold = self.conf_threshold
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = get_sliced_prediction(
                frame_rgb,
                self.sahi_model,
                slice_height=320,
                slice_width=320,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
            )
            for pred in res.object_prediction_list:
                bbox = pred.bbox.to_xyxy()
                x1, y1, x2, y2 = [int(v) for v in bbox]
                conf = float(pred.score.value)
                cls_name = str(pred.category.name)
                crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                norm_x = ((x1 + x2) / 2.0) / max(1, w)
                detections.append({
                    "timestamp": now_str,
                    "class_name": cls_name,
                    "tr_label": tr_label_of(cls_name),
                    "confidence": round(conf, 3),
                    "risk": risk_of(cls_name),
                    "sector": runway_sector_of(norm_x),
                    "bbox": [x1, y1, x2, y2],
                    "mode": mode_label,
                    "is_alarm": conf >= self.alarm_threshold or risk_of(cls_name) == RISK_YUKSEK,
                    "thumbnail": encode_Crop_base64(crop),
                })
        except Exception:
            # SAHI kullanılamazsa yüksek çözünürlüklü YOLO (imgsz=960) ile geri dönüş yap
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=960,
                verbose=False,
            )
            for r in results:
                names = r.names
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    norm_x = ((x1 + x2) / 2.0) / max(1, w)
                    detections.append({
                        "timestamp": now_str,
                        "class_name": cls_name,
                        "tr_label": tr_label_of(cls_name),
                        "confidence": round(conf, 3),
                        "risk": risk_of(cls_name),
                        "sector": runway_sector_of(norm_x),
                        "bbox": [x1, y1, x2, y2],
                        "mode": mode_label,
                        "is_alarm": conf >= self.alarm_threshold or risk_of(cls_name) == RISK_YUKSEK,
                        "thumbnail": encode_Crop_base64(crop),
                    })
        return detections
