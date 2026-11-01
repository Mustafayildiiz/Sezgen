"""Sezgen FastAPI Web Sunucusu ve Gerçek Zamanlı Video İşleme API'si."""
import base64
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.config import (
    DEFAULT_ALARM_THRESHOLD,
    DEFAULT_CAMERA_LABEL,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    MODELS_DIR,
    RESULTS_CSV_PATH,
    UPLOADS_DIR,
)
from src.detector import FODDetector
from src.exporter import export_detections_csv, export_detections_pdf

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(
    title="Sezgen - Havalimanı Pist FOD Tespit Sistemi",
    description="Derin öğrenme (YOLO + SAHI) tabanlı gerçek zamanlı yabancı cisim (FOD) tespit ve izleme paneli.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class SettingsPayload(BaseModel):
    conf: Optional[float] = None
    iou: Optional[float] = None
    alarm: Optional[float] = None
    deep_scan: Optional[bool] = None
    running: Optional[bool] = None
    source_type: Optional[str] = None  # "simulation", "file", "webcam", "rtsp"
    rtsp_url: Optional[str] = None


class SystemState:
    """Uygulama çalışma zamanı durumu, loglar ve tespit geçmişi."""

    def __init__(self) -> None:
        self.detector = FODDetector()
        self.running: bool = False
        self.source_type: str = "simulation"
        self.uploaded_file: Optional[Path] = None
        self.rtsp_url: str = ""
        self.camera_label: str = DEFAULT_CAMERA_LABEL
        self.current_fps: float = 28.5
        self.alarm_active: bool = False

        # Figma tasarımındaki başlangıç Sistem Günlükleri
        self.logs: Deque[Dict[str, str]] = deque(maxlen=150)
        self.logs.append({"time": "12:00:00", "message": "Sistem başlatıldı.", "level": "info"})
        self.logs.append({"time": "12:00:05", "message": "Video akışı alındı.", "level": "info"})
        self.logs.append({"time": "12:00:10", "message": "FOD tespiti yapıldı.", "level": "alarm"})

        # Figma tasarımındaki başlangıç Bulunan Eşyalar listesi
        thumbs = self.detector.default_thumbs
        self.found_items: List[Dict[str, Any]] = [
            {
                "timestamp": "12:00:10",
                "class_name": "MetalPart",
                "tr_label": "Metal Parça",
                "sector": "Pist 12L",
                "confidence": 0.91,
                "risk": "Yüksek",
                "bbox": [380, 430, 424, 462],
                "mode": "Canlı Akış",
                "is_alarm": True,
                "thumbnail": thumbs.get("MetalPart", ""),
            },
            {
                "timestamp": "12:00:10",
                "class_name": "PlasticPart",
                "tr_label": "Plastik Parça",
                "sector": "Pist 12R",
                "confidence": 0.78,
                "risk": "Orta",
                "bbox": [536, 514, 576, 550],
                "mode": "Canlı Akış",
                "is_alarm": False,
                "thumbnail": thumbs.get("PlasticPart", ""),
            },
        ]
        self.detection_history: List[Dict[str, Any]] = list(self.found_items)
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_log_ts: float = 0.0

    def add_log(self, message: str, level: str = "info") -> None:
        now_t = datetime.now().strftime("%H:%M:%S")
        self.logs.append({"time": now_t, "message": message, "level": level})

    def release_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def get_next_frame(self) -> np.ndarray:
        """Seçili kaynaktan (dosya, kamera/RTSP veya simülasyon) sıradaki kareyi okur."""
        if self.source_type == "file" and self.uploaded_file and self.uploaded_file.exists():
            suffix = self.uploaded_file.suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                data = np.fromfile(str(self.uploaded_file), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            else:
                if self._cap is None or not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(str(self.uploaded_file))
                ok, frame = self._cap.read()
                if not ok:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                if ok and frame is not None:
                    return frame

        elif self.source_type in {"webcam", "rtsp"}:
            target: Any = 0 if self.source_type == "webcam" else self.rtsp_url
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(target)
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame

        return self.detector.render_synthetic_runway(width=960, height=720)

    def step_frame(self) -> np.ndarray:
        """Bir kare alır, çalışıyorsa YOLO/SAHI tespitini uygular ve durum tablolarını günceller."""
        raw_frame = self.get_next_frame()
        if not self.running:
            return raw_frame

        annotated, detections, fps = self.detector.detect_frame(raw_frame)
        self.current_fps = fps
        self.alarm_active = any(d.get("is_alarm", False) for d in detections)

        if detections:
            self.found_items = detections[:12]
            now = time.time()
            if now - self._last_log_ts >= 2.5:
                self._last_log_ts = now
                top = detections[0]
                self.add_log(
                    f"FOD tespiti yapıldı: {top['tr_label']} ({top['sector']}) - Güven: %{int(top['confidence'] * 100)}",
                    level="alarm" if top.get("is_alarm") else "warn",
                )
                for d in detections:
                    if len(self.detection_history) < 1000:
                        self.detection_history.append(d)

        return annotated


state = SystemState()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "sezgen-fod-detector",
        "model_loaded": state.detector.model_loaded,
        "model_status": state.detector.model_status,
    }


@app.get("/api/state")
async def get_state():
    return {
        "running": state.running,
        "source_type": state.source_type,
        "uploaded_file": state.uploaded_file.name if state.uploaded_file else None,
        "camera_label": state.camera_label,
        "conf": int(round(state.detector.conf_threshold * 100)),
        "iou": int(round(state.detector.iou_threshold * 100)),
        "alarm": int(round(state.detector.alarm_threshold * 100)),
        "deep_scan": state.detector.deep_scan,
        "fps": state.current_fps,
        "alarm_active": state.alarm_active,
        "model_loaded": state.detector.model_loaded,
        "model_status": state.detector.model_status,
        "logs": list(state.logs),
        "found_items": state.found_items,
        "total_detections": len(state.detection_history),
    }


@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    if payload.conf is not None or payload.iou is not None or payload.alarm is not None or payload.deep_scan is not None:
        prev_mode = state.detector.deep_scan
        state.detector.update_settings(
            conf=payload.conf / 100.0 if payload.conf is not None and payload.conf > 1.0 else payload.conf,
            iou=payload.iou / 100.0 if payload.iou is not None and payload.iou > 1.0 else payload.iou,
            alarm=payload.alarm / 100.0 if payload.alarm is not None and payload.alarm > 1.0 else payload.alarm,
            deep_scan=payload.deep_scan,
        )
        if payload.deep_scan is not None and payload.deep_scan != prev_mode:
            mode_str = "Derin Tarama (SAHI)" if payload.deep_scan else "Canlı Akış"
            state.add_log(f"Tarama modu değiştirildi: {mode_str}", level="info")

    if payload.source_type is not None:
        state.source_type = payload.source_type
        state.release_capture()
        state.add_log(f"Giriş kaynağı seçildi: {payload.source_type}", level="info")

    if payload.rtsp_url is not None:
        state.rtsp_url = payload.rtsp_url.strip()
        if state.rtsp_url:
            state.source_type = "rtsp"
            state.release_capture()
            state.add_log(f"RTSP akışı bağlandı: {state.rtsp_url}", level="info")

    if payload.running is not None:
        state.running = bool(payload.running)
        if state.running:
            state.detector.try_load_model()
            state.add_log("Tarama başlatıldı. Video akışı işleniyor.", level="info")
        else:
            state.alarm_active = False
            state.add_log("Tarama duraklatıldı.", level="info")

    return await get_state()


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya adı geçersiz.")

    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix.lower()
    content = await file.read()

    # Kullanıcı model ağırlığı (.pt / .onnx) yüklerse models/ altına kaydet ve yükle
    if suffix in {".pt", ".onnx"}:
        dst = MODELS_DIR / safe_name
        dst.write_bytes(content)
        loaded = state.detector.try_load_model(dst)
        state.add_log(
            f"Model ağırlığı yüklendi: {safe_name} ({'Aktif' if loaded else 'Hata'})",
            level="info" if loaded else "warn",
        )
        return {"status": "ok", "type": "model", "filename": safe_name, "loaded": loaded}

    # Kullanıcı YOLO eğitim sonuçları (results.csv) yüklerse
    if suffix == ".csv":
        RESULTS_CSV_PATH.write_bytes(content)
        state.add_log(f"Eğitim metrik dosyası güncellendi: {safe_name}", level="info")
        return {"status": "ok", "type": "metrics", "filename": safe_name}

    # Video veya görsel dosyası
    dst = UPLOADS_DIR / safe_name
    dst.write_bytes(content)
    state.release_capture()
    state.uploaded_file = dst
    state.source_type = "file"
    state.running = True
    state.add_log(f"Medya dosyası yüklendi: {safe_name}", level="info")
    state.add_log("Video akışı alındı.", level="info")

    return {"status": "ok", "type": "media", "filename": safe_name}


@app.get("/api/frame")
async def get_single_frame():
    """Tekil işlenmiş kareyi base64 JPEG ve güncel durumla birlikte döndürür."""
    frame = state.step_frame()
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
    st = await get_state()
    st["frame"] = f"data:image/jpeg;base64,{b64}"
    return JSONResponse(st)


def _mjpeg_generator():
    while True:
        frame = state.step_frame()
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        time.sleep(0.04)  # ~25 FPS


@app.get("/api/stream")
async def video_stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/training-metrics")
async def get_training_metrics():
    """Eğitim Grafikleri paneli için results.csv varsa gerçek veriyi, yoksa Figma referans serisini döndürür."""
    if RESULTS_CSV_PATH.exists():
        try:
            with RESULTS_CSV_PATH.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                epochs, train_series, val_series, map_series, recall_series = [], [], [], [], []
                for idx, row in enumerate(reader):
                    clean = {k.strip(): v.strip() for k, v in row.items() if k}
                    epochs.append(clean.get("epoch", str(idx + 1)))
                    t_loss = float(clean.get("train/box_loss", 0.0)) * 500
                    v_loss = float(clean.get("val/box_loss", 0.0)) * 500
                    m50 = float(clean.get("metrics/mAP50(B)", 0.0)) * 1000
                    rec = float(clean.get("metrics/recall(B)", 0.0)) * 1000
                    train_series.append(round(t_loss, 1))
                    val_series.append(round(v_loss, 1))
                    map_series.append(round(m50, 1))
                    recall_series.append(round(rec, 1))
                if epochs:
                    return {
                        "source": "results.csv",
                        "labels": epochs,
                        "chart1": {"blue": train_series, "grey": val_series, "title": "Box Loss (Train vs Val)"},
                        "chart2": {"blue": map_series, "grey": recall_series, "title": "mAP@0.5 vs Recall"},
                    }
        except Exception:
            pass

    # Figma tasarımındaki ikili grafik eğrisiyle birebir örtüşen varsayılan veri serisi (0 - 1200 ölçekli)
    blue_line = [
        925, 925, 845, 765, 920, 810, 625, 620, 815, 730, 520, 535, 860, 740, 845, 780, 645, 595,
        705, 895, 855, 890, 795, 975, 880, 935, 850, 875, 920, 875, 955, 860, 780, 890, 800, 725,
        825, 790, 795, 730, 740, 685, 710, 795, 745, 860, 810, 600, 775, 745, 845, 750, 665, 575,
        620, 840, 705, 735, 675, 620, 550, 560, 635, 480, 555, 485, 675, 385, 225, 305, 510, 445, 415
    ]
    grey_line = [
        1100, 1065, 815, 770, 1020, 890, 695, 995, 860, 715, 515, 790, 980, 675, 915, 700, 620, 635,
        1005, 875, 960, 845, 1035, 925, 970, 885, 995, 845, 975, 850, 985, 905, 825, 915, 1050, 815,
        925, 760, 890, 770, 935, 685, 695, 950, 915, 760, 750, 610, 920, 835, 745, 710, 590, 395,
        940, 725, 885, 880, 665, 615, 540, 565, 715, 550, 755, 535, 805, 260, 155, 290, 600, 415, 475
    ]
    labels = [str(i + 1) for i in range(len(blue_line))]
    return {
        "source": "default",
        "labels": labels,
        "chart1": {"blue": blue_line, "grey": grey_line, "title": "Eğitim & Doğrulama Kayıp (Loss) Eğrisi"},
        "chart2": {"blue": blue_line, "grey": grey_line, "title": "Yakınsama & mAP@0.5 / Recall Performansı"},
    }


@app.get("/api/export/csv")
async def download_csv():
    data = export_detections_csv(state.detection_history)
    filename = f"sezgen_fod_rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/pdf")
async def download_pdf():
    data = export_detections_pdf(state.detection_history)
    filename = f"sezgen_fod_rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

