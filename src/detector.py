"""Sezgen Gerçek YOLO (YOLOv26 / Ultralytics) + SAHI Yabancı Cisim (FOD) Tespit Motoru.

Bu modül hiçbir sahte/simülasyon verisi içermez. Yalnızca yüklenen gerçek YOLO (.pt / .onnx)
model ağırlıklarını kullanarak gerçek video ve görsel kareleri üzerinde çıkarım yapar.
"""
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    DEFAULT_ALARM_THRESHOLD,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    MODEL_PATH,
    MODELS_DIR,
)
from src.risk import (
    RISK_COLOR_BGR,
    RISK_YUKSEK,
    risk_of,
    runway_sector_of,
    tr_label_of,
)

_ASCII_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def _safe_cv_text(text: str) -> str:
    """OpenCV Hershey fontlarında bozuk karakter oluşmaması için güvenli etiket metni üretir."""
    return text.translate(_ASCII_MAP)


def encode_crop_base64(
    frame_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    out_size: Tuple[int, int] = (96, 96),
) -> str:
    """Tespit edilen nesneyi gerçek video karesinden hafif bağlam payıyla kırpar ve base64 JPEG üretir."""
    h, w = frame_bgr.shape[:2]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x = max(8, int(bw * 0.25))
    pad_y = max(8, int(bh * 0.25))

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop is None or crop.size == 0:
        return ""
    resized = cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


def detect_compute_device() -> Tuple[str, str]:
    """Kullanılabilir hesaplama birimini (CUDA GPU veya CPU) gerçek zamanlı tespit eder."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "0", f"CUDA GPU ({gpu_name})"
    except Exception:
        pass
    return "cpu", "CPU"


def list_available_models() -> List[Dict[str, Any]]:
    """models/ klasöründeki tüm gerçek .pt ve .onnx ağırlık dosyalarını listeler."""
    items: List[Dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return items
    for p in sorted(MODELS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in {".pt", ".onnx"}:
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            items.append({
                "name": p.name,
                "path": str(p),
                "size_mb": size_mb,
                "format": p.suffix.lower().lstrip(".").upper(),
            })
    return items


class FODDetector:
    """Gerçek YOLO (.pt / .onnx) ve SAHI dilimlemeli FOD tespit motoru."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path: Path = Path(model_path or MODEL_PATH)
        self.conf_threshold: float = DEFAULT_CONF_THRESHOLD
        self.iou_threshold: float = DEFAULT_IOU_THRESHOLD
        self.alarm_threshold: float = DEFAULT_ALARM_THRESHOLD
        self.imgsz: int = 640
        self.deep_scan: bool = False  # False: Canlı Akış, True: Derin Tarama (Dilimlemeli / SAHI)

        self.device_id, self.device_label = detect_compute_device()
        self.model = None
        self.sahi_model = None
        self.model_loaded: bool = False
        self.model_error: str = ""
        self.class_names: Dict[int, str] = {}
        self.checkpoint_meta: Dict[str, Any] = {}
        self.embedded_train_results: Optional[Dict[str, List[float]]] = None

        # Başlangıçta models/ altında best.pt varsa veya başka bir .pt/.onnx varsa otomatik yükle
        if self.model_path.exists():
            self.load_model(self.model_path)
        else:
            available = list_available_models()
            if available:
                self.load_model(Path(available[0]["path"]))

    def load_model(self, target_path: Path) -> Tuple[bool, str]:
        """Belirtilen .pt veya .onnx model dosyasını Ultralytics YOLO ile yükler."""
        target_path = Path(target_path)
        self.model_path = target_path
        self.sahi_model = None
        self.embedded_train_results = None
        self.checkpoint_meta = {}

        if not target_path.exists():
            self.model = None
            self.model_loaded = False
            self.model_error = f"{target_path.name} bulunamadı"
            return False, f"Model dosyası bulunamadı: {target_path.name}"

        try:
            from ultralytics import YOLO

            # Eğer PyTorch .pt dosyası ise içindeki eğitim metriklerini ve argümanları oku
            if target_path.suffix.lower() == ".pt":
                self._inspect_pt_checkpoint(target_path)

            loaded_model = YOLO(str(target_path), task="detect")
            self.model = loaded_model
            self.model_loaded = True
            self.model_error = ""

            raw_names = getattr(loaded_model, "names", None) or {}
            if isinstance(raw_names, list):
                self.class_names = {i: str(n) for i, n in enumerate(raw_names)}
            elif isinstance(raw_names, dict):
                self.class_names = {int(k): str(v) for k, v in raw_names.items()}
            else:
                self.class_names = {}

            trained_imgsz = self.checkpoint_meta.get("trained_imgsz")
            if isinstance(trained_imgsz, int) and 160 <= trained_imgsz <= 1920:
                self.imgsz = trained_imgsz

            msg = (
                f"Model yüklendi: {target_path.name} "
                f"({len(self.class_names)} sınıf, imgsz={self.imgsz}, {self.device_label})"
            )
            return True, msg
        except Exception as exc:
            self.model = None
            self.model_loaded = False
            self.model_error = str(exc)
            return False, f"Model yüklenemedi ({target_path.name}): {exc}"

    def _inspect_pt_checkpoint(self, pt_path: Path) -> None:
        """Ultralytics .pt ağırlık dosyasının içindeki gerçek eğitim geçmişini ve parametreleri çıkarır."""
        try:
            import torch

            ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict):
                return

            train_args = ckpt.get("train_args") or {}
            train_metrics = ckpt.get("train_metrics") or {}
            train_results = ckpt.get("train_results")

            raw_imgsz = train_args.get("imgsz")
            if isinstance(raw_imgsz, (list, tuple)) and raw_imgsz:
                raw_imgsz = raw_imgsz[0]

            self.checkpoint_meta = {
                "arch": str(train_args.get("model", pt_path.name)),
                "epochs": train_args.get("epochs"),
                "trained_imgsz": int(raw_imgsz) if isinstance(raw_imgsz, (int, float)) else None,
                "ultralytics_version": ckpt.get("version", ""),
                "date": ckpt.get("date", ""),
                "best_metrics": {
                    k: round(float(v), 4)
                    for k, v in train_metrics.items()
                    if isinstance(v, (int, float))
                },
            }

            if isinstance(train_results, dict) and train_results:
                cleaned: Dict[str, List[float]] = {}
                for k, vals in train_results.items():
                    if isinstance(vals, (list, tuple, np.ndarray)):
                        cleaned[str(k).strip()] = [float(x) for x in vals]
                if cleaned:
                    self.embedded_train_results = cleaned
        except Exception:
            pass

    def update_settings(
        self,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        alarm: Optional[float] = None,
        deep_scan: Optional[bool] = None,
        imgsz: Optional[int] = None,
    ) -> None:
        if conf is not None:
            self.conf_threshold = max(0.01, min(0.99, float(conf)))
        if iou is not None:
            self.iou_threshold = max(0.01, min(0.99, float(iou)))
        if alarm is not None:
            self.alarm_threshold = max(0.01, min(0.99, float(alarm)))
        if deep_scan is not None:
            self.deep_scan = bool(deep_scan)
        if imgsz is not None and int(imgsz) in {320, 480, 640, 960, 1280}:
            self.imgsz = int(imgsz)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "loaded": self.model_loaded,
            "model_name": self.model_path.name if self.model_loaded else None,
            "model_path": str(self.model_path) if self.model_loaded else None,
            "error": self.model_error,
            "device": self.device_label,
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names.values()),
            "imgsz": self.imgsz,
            "meta": self.checkpoint_meta,
            "available_models": list_available_models(),
        }

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
        video_time_str: str = "00:00.0",
        source_name: str = "",
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """Gerçek video/görsel karesi üzerinde YOLO / SAHI tespiti çalıştırır.
        Model yüklü değilse sahte kutu ÜRETMEZ; orijinal kareyi ve boş tespit listesini döndürür.
        """
        t0 = time.perf_counter()
        if not self.model_loaded or self.model is None:
            elapsed = max(1e-4, time.perf_counter() - t0)
            return frame_bgr, [], round(1.0 / elapsed, 1)

        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        now_clock = datetime.now().strftime("%H:%M:%S")
        mode_label = "Derin Tarama (SAHI)" if self.deep_scan else "Canlı Akış"

        if self.deep_scan:
            raw_boxes = self._infer_sliced(frame_bgr)
        else:
            raw_boxes = self._infer_standard(frame_bgr)

        # Tüm yöntemlerin ortak son adımı: eşik filtresi + sınıflar arası NMS + kapsama bastırma
        raw_boxes = self.postprocess_boxes(raw_boxes)

        detections: List[Dict[str, Any]] = []
        for x1, y1, x2, y2, conf, cls_id, cls_name in raw_boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            tr_name = tr_label_of(cls_name)
            risk = risk_of(cls_name)
            norm_x = ((x1 + x2) / 2.0) / max(1, w)
            sector = runway_sector_of(norm_x)
            is_alarm = conf >= self.alarm_threshold

            detections.append({
                "id": f"{frame_idx}_{x1}_{y1}_{cls_id}",
                "timestamp": now_clock,
                "source_name": source_name,
                "frame_idx": frame_idx,
                "video_time": video_time_str,
                "class_id": cls_id,
                "class_name": cls_name,
                "tr_label": tr_name,
                "confidence": round(float(conf), 4),
                "risk": risk,
                "sector": sector,
                "bbox": [x1, y1, x2, y2],
                "mode": mode_label,
                "model_name": self.model_path.name,
                "is_alarm": is_alarm,
                "thumbnail": encode_crop_base64(frame_bgr, x1, y1, x2, y2),
            })

        # Tespit kutularını kare üzerine çiz
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            risk = det["risk"]
            color = RISK_COLOR_BGR.get(risk, (0, 165, 255))
            label_txt = _safe_cv_text(
                f"{det['tr_label']} %{det['confidence'] * 100:.1f} [{risk}]"
            )

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
            bg_y1 = max(0, y1 - th - 10)
            cv2.rectangle(annotated, (x1, bg_y1), (min(w, x1 + tw + 10), y1), color, -1)
            cv2.putText(
                annotated,
                label_txt,
                (x1 + 5, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        elapsed = max(1e-4, time.perf_counter() - t0)
        fps = round(1.0 / elapsed, 1)
        return annotated, detections, fps

    def postprocess_boxes(
        self, boxes: List[Tuple[int, int, int, int, float, int, str]]
    ) -> List[Tuple[int, int, int, int, float, int, str]]:
        """Ham kutuları temizler: (1) Conf eşiği, (2) sınıflar arası (agnostic) NMS (IoU eşiği),
        (3) büyük bir kutunun içine gömülü düşük güvenli kutuları bastırma.
        Aynı fiziksel nesne için tek kutu kalmasını sağlar."""
        valid = [b for b in boxes if b[4] >= self.conf_threshold and b[2] > b[0] and b[3] > b[1]]
        if len(valid) <= 1:
            return valid

        order = sorted(range(len(valid)), key=lambda i: valid[i][4], reverse=True)
        kept: List[Tuple[int, int, int, int, float, int, str]] = []
        for i in order:
            x1, y1, x2, y2 = valid[i][:4]
            area = (x2 - x1) * (y2 - y1)
            suppressed = False
            for k in kept:
                kx1, ky1, kx2, ky2 = k[:4]
                iw = min(x2, kx2) - max(x1, kx1)
                ih = min(y2, ky2) - max(y1, ky1)
                if iw <= 0 or ih <= 0:
                    continue
                inter = iw * ih
                karea = (kx2 - kx1) * (ky2 - ky1)
                iou = inter / float(area + karea - inter)
                containment = inter / float(min(area, karea))
                if iou >= self.iou_threshold or containment >= 0.85:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(valid[i])
        return kept

    def _infer_standard(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int, str]]:
        """Standart tam kare YOLO çıkarımı."""
        out: List[Tuple[int, int, int, int, float, int, str]] = []
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device_id,
            agnostic_nms=True,
            verbose=False,
        )
        for r in results:
            names = r.names or self.class_names
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(names[cls_id])
                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                out.append((x1, y1, x2, y2, conf, cls_id, cls_name))
        return out

    def _infer_sliced(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int, str]]:
        """Küçük FOD nesneleri için SAHI veya örtüşmeli karo (tile) dilimlemeli çıkarım."""
        # 1. Önce SAHI ile dene
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            if self.sahi_model is None:
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=str(self.model_path),
                    confidence_threshold=self.conf_threshold,
                    device=self.device_id,
                )
            self.sahi_model.confidence_threshold = self.conf_threshold
            h, w = frame_bgr.shape[:2]
            slice_sz = max(320, min(640, min(w, h) // 2 or 320))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = get_sliced_prediction(
                frame_rgb,
                self.sahi_model,
                slice_height=slice_sz,
                slice_width=slice_sz,
                overlap_height_ratio=0.25,
                overlap_width_ratio=0.25,
                postprocess_match_threshold=self.iou_threshold,
                verbose=0,
            )
            out: List[Tuple[int, int, int, int, float, int, str]] = []
            for pred in res.object_prediction_list:
                x1, y1, x2, y2 = [int(round(v)) for v in pred.bbox.to_xyxy()]
                conf = float(pred.score.value)
                cls_id = int(pred.category.id)
                cls_name = str(pred.category.name)
                out.append((x1, y1, x2, y2, conf, cls_id, cls_name))
            return out
        except Exception:
            # 2. YOLOv26 özel mimari / ONNX durumunda yerleşik örtüşmeli dilimleme + NMS
            return self._infer_manual_tiles(frame_bgr)

    def _infer_manual_tiles(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int, str]]:
        """Yüksek çözünürlüklü kareyi örtüşmeli karolara bölüp YOLO ile tarar ve NMS ile birleştirir."""
        h, w = frame_bgr.shape[:2]
        tile_w = max(320, min(640, w // 2 if w >= 640 else w))
        tile_h = max(320, min(640, h // 2 if h >= 640 else h))
        step_x = max(160, int(tile_w * 0.75))
        step_y = max(160, int(tile_h * 0.75))

        candidates: List[Tuple[int, int, int, int, float, int, str]] = list(self._infer_standard(frame_bgr))

        for y0 in range(0, max(1, h - tile_h + step_y), step_y):
            y0 = min(y0, max(0, h - tile_h))
            for x0 in range(0, max(1, w - tile_w + step_x), step_x):
                x0 = min(x0, max(0, w - tile_w))
                tile = frame_bgr[y0:y0 + tile_h, x0:x0 + tile_w]
                if tile.size == 0:
                    continue
                tile_preds = self._infer_standard(tile)
                for tx1, ty1, tx2, ty2, conf, cid, cname in tile_preds:
                    candidates.append((tx1 + x0, ty1 + y0, tx2 + x0, ty2 + y0, conf, cid, cname))
                if x0 + tile_w >= w:
                    break
            if y0 + tile_h >= h:
                break

        if not candidates:
            return []

        boxes_xywh = [[c[0], c[1], c[2] - c[0], c[3] - c[1]] for c in candidates]
        scores = [c[4] for c in candidates]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.conf_threshold, self.iou_threshold)
        if len(indices) == 0:
            return []
        flat_idx = [int(i) for i in np.array(indices).flatten()]
        return [candidates[i] for i in flat_idx]
