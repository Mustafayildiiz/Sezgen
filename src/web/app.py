"""Sezgen FastAPI Gerçek Zamanlı Video Analiz, Model Yönetimi ve Raporlama Sunucusu.

Hiçbir sahte/prototip veri içermez:
- Yalnızca kullanıcının yüklediği/seçtiği gerçek video, görsel veya RTSP kaynağını oynatır.
- Yalnızca models/ altındaki veya arayüzden yüklenen gerçek YOLO (.pt / .onnx) modeli ile tespit yapar.
- Bulunan Eşyalar, Sistem Günlükleri ve Eğitim Grafikleri tamamen gerçek çalışma verilerinden beslenir.
"""
import csv
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import (
    MODELS_DIR,
    RESULTS_CSV_PATH,
    UPLOADS_DIR,
)
from src.detector import FODDetector, list_available_models
from src.exporter import export_detections_csv, export_detections_pdf

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_EXTS = {".pt", ".onnx"}

app = FastAPI(
    title="Sezgen — Havalimanı Pist FOD Tespit Sistemi",
    description="YOLOv26 + SAHI tabanlı gerçek zamanlı yabancı cisim (FOD) analiz sistemi.",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _format_seconds(sec: float) -> str:
    if sec < 0 or not np.isfinite(sec):
        return "00:00.0"
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins:02d}:{rem:04.1f}"


def list_uploaded_media() -> List[Dict[str, Any]]:
    """uploads/ klasöründeki gerçek video ve görsel dosyalarını listeler."""
    items: List[Dict[str, Any]] = []
    if not UPLOADS_DIR.exists():
        return items
    for p in sorted(UPLOADS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file() and p.suffix.lower() in (VIDEO_EXTS | IMAGE_EXTS):
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            mtype = "video" if p.suffix.lower() in VIDEO_EXTS else "image"
            items.append({
                "name": p.name,
                "type": mtype,
                "size_mb": size_mb,
            })
    return items


class SettingsPayload(BaseModel):
    conf: Optional[float] = None
    iou: Optional[float] = None
    alarm: Optional[float] = None
    deep_scan: Optional[bool] = None
    imgsz: Optional[int] = None
    loop_video: Optional[bool] = None


class SelectModelPayload(BaseModel):
    model_name: str


class SelectMediaPayload(BaseModel):
    filename: str


class RtspPayload(BaseModel):
    url: str


class SeekPayload(BaseModel):
    frame_idx: int


class SystemState:
    """Gerçek zamanlı video akışı, YOLO çıkarımı, tespit geçmişi ve sistem günlükleri yöneticisi."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.frame_cond = threading.Condition()

        self.detector = FODDetector()

        # Medya / Oynatıcı durumu
        self.media_loaded: bool = False
        self.media_type: str = "none"  # "none" | "video" | "image" | "stream"
        self.source_name: str = ""
        self.media_path: Optional[Path] = None
        self.rtsp_url: str = ""

        self.video_width: int = 0
        self.video_height: int = 0
        self.native_fps: float = 25.0
        self.total_frames: int = 0
        self.duration_sec: float = 0.0
        self.current_frame_idx: int = 0
        self.current_video_time: str = "00:00.0"

        self.running: bool = False
        self.loop_video: bool = False
        self.processing_fps: float = 0.0
        self.alarm_active: bool = False
        self.current_frame_detections: int = 0

        self._cap: Optional[cv2.VideoCapture] = None
        self._raw_image: Optional[np.ndarray] = None
        self._last_jpeg: bytes = b""
        self._frame_seq: int = 0

        # Gerçek Sistem Günlükleri ve Bulunan Eşyalar (Başlangıçta temiz!)
        self.logs: Deque[Dict[str, str]] = deque(maxlen=300)
        self.found_items: List[Dict[str, Any]] = []
        self.detection_history: List[Dict[str, Any]] = []
        self._recent_det_cache: Dict[str, float] = {}
        self._tracks: List[Dict[str, Any]] = []
        self._deep_forced: bool = False
        self._deep_prior: bool = False

        # Gerçek başlangıç günlük kaydı
        self.add_log(f"Sistem başlatıldı. Hesaplama birimi: {self.detector.device_label}", "info")
        if self.detector.model_loaded:
            self.add_log(
                f"Model hazır: {self.detector.model_path.name} ({len(self.detector.class_names)} sınıf)",
                "info",
            )
        else:
            self.add_log(
                "Henüz aktif model yok. Model Ayarları bölümünden best.pt (.pt / .onnx) yükleyin.",
                "warn",
            )

        # Arka plan video işleme iş parçacığı
        self._stop_worker = False
        self._worker = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker.start()

    def add_log(self, message: str, level: str = "info") -> None:
        now_t = datetime.now().strftime("%H:%M:%S")
        self.logs.append({"time": now_t, "message": message, "level": level})

    def enforce_mode_for_media(self, media_type: str) -> None:
        """Görsel girdide Canlı Akış kapalıdır: yalnızca Derin Tarama kullanılabilir.
        Video / akış girdisinde kullanıcının önceki tercihi geri yüklenir."""
        det = self.detector
        if media_type == "image":
            if not self._deep_forced:
                self._deep_prior = det.deep_scan
                self._deep_forced = True
            det.deep_scan = True
        elif self._deep_forced:
            det.deep_scan = self._deep_prior
            self._deep_forced = False

    def release_capture(self) -> None:
        with self.lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def _publish_frame(self, bgr_frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok:
            with self.frame_cond:
                self._last_jpeg = buf.tobytes()
                self._frame_seq += 1
                self.frame_cond.notify_all()

    def _record_detections(self, detections: List[Dict[str, Any]]) -> None:
        """Aynı kare/saniyedeki mükerrer kayıtları akıllıca süzerek gerçek tespitleri kaydeder."""
        self.current_frame_detections = len(detections)
        self.alarm_active = any(d.get("is_alarm", False) for d in detections)
        if not detections:
            return

        # Aynı fiziksel nesneyi (aynı sınıf + yüksek IoU) kareler boyunca TEK kayıt olarak izle.
        # Böylece tek bir cisim 70 kare görünse bile sayaç 1 kalır.
        for d in detections:
            track = self._match_track(d)
            if track is not None:
                track["bbox"] = d["bbox"]
                track["last_frame"] = d["frame_idx"]
                # Daha yüksek güvenli kareyi temsilci kayıt yap
                if d["confidence"] > track["item"]["confidence"]:
                    idx = next((i for i, it in enumerate(self.found_items) if it is track["item"]), None)
                    hidx = next((i for i, it in enumerate(self.detection_history) if it is track["item"]), None)
                    track["item"] = d
                    if idx is not None:
                        self.found_items[idx] = d
                    if hidx is not None:
                        self.detection_history[hidx] = d
                continue

            self._tracks.append({"class": d["class_name"], "bbox": d["bbox"],
                                 "last_frame": d["frame_idx"], "item": d})
            self.found_items.insert(0, d)
            if len(self.found_items) > 60:
                self.found_items = self.found_items[:60]
            if len(self.detection_history) < 5000:
                self.detection_history.append(d)
            self.add_log(
                f"[Kare #{d['frame_idx']} | {d['video_time']}] Yeni FOD: "
                f"{d['tr_label']} (%{d['confidence'] * 100:.1f} - Risk: {d['risk']}) "
                f"Konum: {d['bbox']}",
                level="alarm" if d.get("is_alarm") else "warn",
            )

    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        iw = min(a[2], b[2]) - max(a[0], b[0])
        ih = min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _match_track(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        max_gap = max(30, int(self.native_fps * 2))
        best, best_iou = None, 0.3
        for t in self._tracks:
            if t["class"] != d["class_name"]:
                continue
            if abs(d["frame_idx"] - t["last_frame"]) > max_gap:
                continue
            v = self._iou(t["bbox"], d["bbox"])
            if v >= best_iou:
                best, best_iou = t, v
        return best

    def reset_detections(self) -> None:
        self.found_items.clear()
        self.detection_history.clear()
        self._tracks.clear()
        self._recent_det_cache.clear()
        self.current_frame_detections = 0
        self.alarm_active = False

    def load_media_file(self, file_path: Path) -> Dict[str, Any]:
        """Yüklenen gerçek video veya görsel dosyasını açar, meta verilerini okur ve ilk kareyi hazırlar."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path.name}")

        suffix = file_path.suffix.lower()
        with self.lock:
            self.running = False
            self.release_capture()
            self._raw_image = None
            self.alarm_active = False
            self.reset_detections()

            self.enforce_mode_for_media("image" if suffix in IMAGE_EXTS else "video")

            if suffix in IMAGE_EXTS:
                raw_bytes = np.fromfile(str(file_path), dtype=np.uint8)
                img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Görsel dosyası okunamadı veya bozuk.")

                h, w = img.shape[:2]
                self.media_loaded = True
                self.media_type = "image"
                self.source_name = file_path.name
                self.media_path = file_path
                self.video_width, self.video_height = w, h
                self.native_fps = 0.0
                self.total_frames = 1
                self.duration_sec = 0.0
                self.current_frame_idx = 1
                self.current_video_time = "Görsel"
                self._raw_image = img

                self.add_log(f"Görsel yüklendi: {file_path.name} ({w}x{h})", "info")
                if self.detector.model_loaded:
                    self.analyze_current_image()
                else:
                    self._publish_frame(img)
                    self.add_log(
                        "Görsel önizlemede. FOD tespiti yapmak için Model Ayarları'ndan best.pt yükleyin.",
                        "warn",
                    )

            elif suffix in VIDEO_EXTS:
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    raise ValueError("Video dosyası açılamadı (desteklenmeyen kodek veya bozuk dosya).")

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
                if fps <= 0 or fps > 240:
                    fps = 25.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                dur = total / fps if total > 0 else 0.0

                ok, first_frame = cap.read()
                if not ok or first_frame is None:
                    cap.release()
                    raise ValueError("Videodan ilk kare okunamadı.")

                # Başa geri sar ki Başla'ya basıldığında 0. kareden başlasın
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                self._cap = cap
                self.media_loaded = True
                self.media_type = "video"
                self.source_name = file_path.name
                self.media_path = file_path
                self.video_width, self.video_height = w, h
                self.native_fps = round(fps, 2)
                self.total_frames = total
                self.duration_sec = round(dur, 2)
                self.current_frame_idx = 0
                self.current_video_time = "00:00.0"

                self._publish_frame(first_frame)
                self.add_log(
                    f"Video yüklendi: {file_path.name} ({w}x{h}, {self.native_fps} FPS, "
                    f"{total} kare, {_format_seconds(dur)})",
                    "info",
                )
            else:
                raise ValueError(f"Desteklenmeyen medya uzantısı: {suffix}")

        return self.to_dict()

    def connect_stream(self, url: str) -> Dict[str, Any]:
        """Gerçek RTSP IP kamera veya yerel USB kamera (0, 1...) akışına bağlanır."""
        clean_url = url.strip()
        target: Any = int(clean_url) if clean_url.isdigit() else clean_url

        with self.lock:
            self.running = False
            self.release_capture()
            self.reset_detections()
            self.enforce_mode_for_media("stream")
            cap = cv2.VideoCapture(target)
            if not cap.isOpened():
                raise ValueError(f"Kamera / RTSP akışına bağlanılamadı: {clean_url}")

            ok, first_frame = cap.read()
            if not ok or first_frame is None:
                cap.release()
                raise ValueError(f"Akıştan görüntü alınamadı: {clean_url}")

            h, w = first_frame.shape[:2]
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            if fps <= 0 or fps > 120:
                fps = 25.0

            self._cap = cap
            self.media_loaded = True
            self.media_type = "stream"
            self.rtsp_url = clean_url
            self.source_name = f"Kamera #{clean_url}" if clean_url.isdigit() else clean_url
            self.video_width, self.video_height = w, h
            self.native_fps = round(fps, 1)
            self.total_frames = 0
            self.duration_sec = 0.0
            self.current_frame_idx = 1
            self.current_video_time = "CANLI"
            self._publish_frame(first_frame)
            self.add_log(f"Canlı akış bağlandı: {self.source_name} ({w}x{h})", "info")

        return self.to_dict()

    def analyze_current_image(self) -> None:
        """Yüklü tekil görsel üzerinde gerçek YOLO/SAHI analizini çalıştırır."""
        with self.lock:
            if self.media_type != "image" or self._raw_image is None:
                return
            if not self.detector.model_loaded:
                self._publish_frame(self._raw_image)
                self.add_log("Analiz yapılamadı: Önce best.pt model dosyasını yükleyin.", "warn")
                return

            annotated, dets, fps = self.detector.detect_frame(
                self._raw_image,
                frame_idx=1,
                video_time_str="Görsel",
                source_name=self.source_name,
            )
            self.processing_fps = fps
            # Aynı görselin yeniden analizinde (eşik değişimi vb.) sonuçları BİRİKTİRME, değiştir.
            self.reset_detections()
            self.found_items = list(dets)
            self.detection_history = list(dets)
            self._tracks = [{"class": d["class_name"], "bbox": d["bbox"],
                             "last_frame": 1, "item": d} for d in dets]
            self.current_frame_detections = len(dets)
            self.alarm_active = any(d.get("is_alarm", False) for d in dets)
            self._publish_frame(annotated)
            self.add_log(
                f"Görsel analizi tamamlandı ({self.source_name}): {len(dets)} yabancı cisim tespit edildi.",
                "alarm" if dets else "info",
            )

    def process_single_video_frame(self) -> bool:
        """Videodan sıradaki tek kareyi okur, model yüklüyse tespit yapar ve yayınlar."""
        with self.lock:
            if self._cap is None or not self._cap.isOpened():
                self.running = False
                return False

            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self.media_type == "video" and self.loop_video:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_idx = 0
                    ok, frame = self._cap.read()
                if not ok or frame is None:
                    self.running = False
                    self.alarm_active = False
                    self.add_log(
                        f"Video sonuna ulaşıldı ({self.current_frame_idx} kare işlendi, "
                        f"toplam {len(self.detection_history)} FOD kaydı).",
                        "info",
                    )
                    return False

            pos_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or (self.current_frame_idx + 1))
            self.current_frame_idx = pos_frame
            if self.media_type == "video":
                t_sec = pos_frame / max(1.0, self.native_fps)
                self.current_video_time = _format_seconds(t_sec)
            else:
                self.current_video_time = datetime.now().strftime("%H:%M:%S")

            if self.detector.model_loaded:
                annotated, dets, fps = self.detector.detect_frame(
                    frame,
                    frame_idx=self.current_frame_idx,
                    video_time_str=self.current_video_time,
                    source_name=self.source_name,
                )
                self.processing_fps = fps
                self._record_detections(dets)
                self._publish_frame(annotated)
            else:
                self.processing_fps = self.native_fps
                self.current_frame_detections = 0
                self.alarm_active = False
                self._publish_frame(frame)

            return True

    def seek_video(self, target_frame: int) -> None:
        """Videoyu belirtilen kare numarasına sarar ve o kareyi analiz edip ekranda gösterir."""
        with self.lock:
            if self.media_type != "video" or self._cap is None or not self._cap.isOpened():
                return
            clamped = max(0, min(max(0, self.total_frames - 1), int(target_frame)))
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
            self.current_frame_idx = clamped
            self.process_single_video_frame()

    def _playback_worker(self) -> None:
        """Arka planda video oynatma ve gerçek zamanlı çıkarım döngüsü."""
        while not self._stop_worker:
            if not self.running or self.media_type not in {"video", "stream"}:
                time.sleep(0.05)
                continue

            t_start = time.perf_counter()
            ok = self.process_single_video_frame()
            if not ok:
                time.sleep(0.05)
                continue

            # Videonun doğal hızından daha hızlı akmaması için kare hızı dengelemesi
            target_dt = 1.0 / max(1.0, min(60.0, self.native_fps or 25.0))
            elapsed = time.perf_counter() - t_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "loop_video": self.loop_video,
            "media": {
                "loaded": self.media_loaded,
                "type": self.media_type,
                "source_name": self.source_name,
                "width": self.video_width,
                "height": self.video_height,
                "native_fps": self.native_fps,
                "total_frames": self.total_frames,
                "duration_sec": self.duration_sec,
                "duration_str": _format_seconds(self.duration_sec),
                "current_frame": self.current_frame_idx,
                "current_time": self.current_video_time,
                "frame_seq": self._frame_seq,
                "uploaded_files": list_uploaded_media(),
            },
            "model": self.detector.get_model_info(),
            "settings": {
                "conf": int(round(self.detector.conf_threshold * 100)),
                "iou": int(round(self.detector.iou_threshold * 100)),
                "alarm": int(round(self.detector.alarm_threshold * 100)),
                "deep_scan": self.detector.deep_scan,
                "live_allowed": self.media_type != "image",
                "imgsz": self.detector.imgsz,
            },
            "telemetry": {
                "processing_fps": self.processing_fps,
                "alarm_active": self.alarm_active,
                "current_frame_detections": self.current_frame_detections,
                "total_detections": len(self.detection_history),
            },
            "logs": list(self.logs),
            "found_items": self.found_items,
        }


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
        "media_loaded": state.media_loaded,
    }


@app.get("/api/state")
async def get_state():
    return state.to_dict()


@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    prev_deep = state.detector.deep_scan
    prev_imgsz = state.detector.imgsz

    # Görsel girdide Canlı Akış kullanılamaz: Derin Tarama zorunlu.
    if state.media_type == "image" and payload.deep_scan is False:
        payload.deep_scan = True
        state.add_log("Görsel girdide Canlı Akış kapalı; Derin Tarama kullanılıyor.", "warn")

    state.detector.update_settings(
        conf=payload.conf / 100.0 if payload.conf is not None and payload.conf > 1.0 else payload.conf,
        iou=payload.iou / 100.0 if payload.iou is not None and payload.iou > 1.0 else payload.iou,
        alarm=payload.alarm / 100.0 if payload.alarm is not None and payload.alarm > 1.0 else payload.alarm,
        deep_scan=payload.deep_scan,
        imgsz=payload.imgsz,
    )

    if payload.loop_video is not None:
        state.loop_video = bool(payload.loop_video)

    if payload.deep_scan is not None and payload.deep_scan != prev_deep:
        m_txt = "Derin Tarama (SAHI)" if state.detector.deep_scan else "Canlı Akış"
        state.add_log(f"Tarama modu güncellendi: {m_txt}", "info")

    if payload.imgsz is not None and state.detector.imgsz != prev_imgsz:
        state.add_log(f"Çıkarım çözünürlüğü (imgsz) güncellendi: {state.detector.imgsz}px", "info")

    # Eğer ekranda tekil bir görsel açıksa veya video duraklatılmışsa anında mevcut kareyi yeni eşiklerle güncelle
    if state.media_loaded and state.media_type == "image" and state.detector.model_loaded:
        state.analyze_current_image()

    return state.to_dict()


@app.post("/api/upload/media")
async def upload_media(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya adı geçersiz.")

    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in (VIDEO_EXTS | IMAGE_EXTS):
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya türü ({suffix}). Lütfen video (.mp4, .avi, .mov, .mkv) veya görsel (.jpg, .png) yükleyin.",
        )

    dst = UPLOADS_DIR / safe_name
    with dst.open("wb") as out_f:
        shutil.copyfileobj(file.file, out_f)

    try:
        return state.load_media_file(dst)
    except Exception as exc:
        state.add_log(f"Medya yükleme hatası ({safe_name}): {exc}", "alarm")
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/media/select")
async def select_existing_media(payload: SelectMediaPayload):
    target = UPLOADS_DIR / Path(payload.filename).name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Seçilen medya dosyası bulunamadı.")
    try:
        return state.load_media_file(target)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/media/rtsp")
async def connect_rtsp_or_camera(payload: RtspPayload):
    if not payload.url.strip():
        raise HTTPException(status_code=400, detail="RTSP adresi veya kamera numarası boş olamaz.")
    try:
        return state.connect_stream(payload.url)
    except Exception as exc:
        state.add_log(f"Akış bağlantı hatası: {exc}", "alarm")
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/upload/model")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Model dosya adı geçersiz.")

    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in MODEL_EXTS:
        raise HTTPException(
            status_code=400,
            detail="Yalnızca Ultralytics YOLO .pt (örn. best.pt) veya .onnx model dosyaları yüklenebilir.",
        )

    dst = MODELS_DIR / safe_name
    with dst.open("wb") as out_f:
        shutil.copyfileobj(file.file, out_f)

    ok, msg = state.detector.load_model(dst)
    state.add_log(msg, "info" if ok else "alarm")
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    # Eğer bir görsel yüklüyse yeni modelle hemen analiz et
    if state.media_loaded and state.media_type == "image":
        state.analyze_current_image()

    return state.to_dict()


@app.post("/api/model/select")
async def select_model(payload: SelectModelPayload):
    target = MODELS_DIR / Path(payload.model_name).name
    ok, msg = state.detector.load_model(target)
    state.add_log(msg, "info" if ok else "alarm")
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    if state.media_loaded and state.media_type == "image":
        state.analyze_current_image()
    return state.to_dict()


@app.post("/api/model/refresh")
async def refresh_models():
    """models/ klasörüne dışarıdan kopyalanan best.pt dosyasını tarar ve henüz model yüklü değilse yükler."""
    models = list_available_models()
    if not state.detector.model_loaded and models:
        best_candidate = next((m for m in models if m["name"] == "best.pt"), models[0])
        ok, msg = state.detector.load_model(Path(best_candidate["path"]))
        state.add_log(msg, "info" if ok else "warn")
    else:
        state.add_log(f"Model klasörü tarandı ({len(models)} model dosyası bulundu).", "info")
    return state.to_dict()


@app.post("/api/upload/metrics")
async def upload_training_metrics(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Lütfen YOLO eğitim çıktısı olan results.csv dosyasını seçin.")

    with RESULTS_CSV_PATH.open("wb") as out_f:
        shutil.copyfileobj(file.file, out_f)

    state.add_log(f"Gerçek eğitim metrikleri yüklendi: {file.filename}", "info")
    return await get_training_metrics()


@app.post("/api/control/toggle")
async def control_toggle():
    if not state.media_loaded:
        state.add_log("Başlatılamadı: Önce analiz edilecek video veya görsel yükleyin.", "warn")
        raise HTTPException(status_code=400, detail="Lütfen önce bir video veya görsel yükleyin.")

    if state.media_type == "image":
        state.analyze_current_image()
        return state.to_dict()

    with state.lock:
        # Video sonuna gelmişse ve tekrar Başla'ya basıldıysa otomatik başa sar
        if not state.running and state.media_type == "video" and state.total_frames > 0:
            if state.current_frame_idx >= state.total_frames - 1 and state._cap is not None:
                state._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state.current_frame_idx = 0

        state.running = not state.running
        if state.running:
            if state.detector.model_loaded:
                state.add_log(
                    f"Analiz başlatıldı: {state.source_name} | Model: {state.detector.model_path.name}",
                    "info",
                )
            else:
                state.add_log(
                    f"Video oynatılıyor ({state.source_name}) — Uyarı: Model yüklü olmadığı için FOD tespiti yapılmıyor.",
                    "warn",
                )
        else:
            state.alarm_active = False
            state.add_log(f"Video duraklatıldı (Kare #{state.current_frame_idx}).", "info")

    return state.to_dict()


@app.post("/api/control/restart")
async def control_restart():
    if not state.media_loaded:
        raise HTTPException(status_code=400, detail="Yüklü medya yok.")
    with state.lock:
        state.running = False
        state.alarm_active = False
        if state.media_type == "video":
            state.seek_video(0)
            state.add_log(f"Video başa sarıldı: {state.source_name}", "info")
        elif state.media_type == "image":
            state.analyze_current_image()
    return state.to_dict()


@app.post("/api/control/step")
async def control_step_frame():
    """Duraklatılmış videoda tek kare ileri gider ve o kareyi analiz eder."""
    if not state.media_loaded or state.media_type != "video":
        raise HTTPException(status_code=400, detail="Kare adımlama yalnızca video dosyalarında kullanılabilir.")
    with state.lock:
        state.running = False
        state.process_single_video_frame()
    return state.to_dict()


@app.post("/api/control/seek")
async def control_seek(payload: SeekPayload):
    if not state.media_loaded or state.media_type != "video":
        raise HTTPException(status_code=400, detail="Konumlandırma yalnızca video dosyalarında kullanılabilir.")
    with state.lock:
        state.running = False
        state.seek_video(payload.frame_idx)
    return state.to_dict()


@app.post("/api/detections/clear")
async def clear_detections():
    with state.lock:
        state.reset_detections()
        state.add_log("Bulunan eşyalar listesi temizlendi.", "info")
    return state.to_dict()


@app.post("/api/logs/clear")
async def clear_logs():
    state.logs.clear()
    state.add_log("Sistem günlükleri temizlendi.", "info")
    return state.to_dict()


@app.get("/api/snapshot")
async def get_snapshot():
    """En son işlenen gerçek video/görsel karesini JPEG olarak döndürür."""
    if not state._last_jpeg:
        return Response(status_code=204)
    return Response(
        content=state._last_jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


def _mjpeg_stream_generator():
    last_sent_seq = -1
    while True:
        with state.frame_cond:
            if state._frame_seq == last_sent_seq:
                state.frame_cond.wait(timeout=0.5)
            jpeg_bytes = state._last_jpeg
            curr_seq = state._frame_seq

        if jpeg_bytes and curr_seq != last_sent_seq:
            last_sent_seq = curr_seq
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        elif not state.running:
            time.sleep(0.08)


@app.get("/api/stream")
async def video_stream():
    return StreamingResponse(
        _mjpeg_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


def _parse_metrics_rows(columns_dict: Dict[str, List[float]], source_label: str) -> Dict[str, Any]:
    """Ultralytics results.csv veya best.pt train_results sözlüğünü gerçek grafik serilerine dönüştürür."""
    def _pick(*candidates: str) -> List[float]:
        for c in candidates:
            if c in columns_dict and columns_dict[c]:
                return [round(float(v), 5) for v in columns_dict[c]]
        return []

    t_box = _pick("train/box_loss", "box_loss")
    v_box = _pick("val/box_loss")
    t_cls = _pick("train/cls_loss", "cls_loss")
    v_cls = _pick("val/cls_loss")
    t_dfl = _pick("train/dfl_loss", "dfl_loss")

    map50 = _pick("metrics/mAP50(B)", "metrics/mAP_0.5", "mAP50")
    map50_95 = _pick("metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95", "mAP50-95")
    precision = _pick("metrics/precision(B)", "metrics/precision", "precision")
    recall = _pick("metrics/recall(B)", "metrics/recall", "recall")

    n = max(
        len(t_box), len(v_box), len(t_cls), len(map50), len(recall), 0
    )
    if n == 0:
        return {"available": False}

    raw_epochs = columns_dict.get("epoch", [])
    if len(raw_epochs) == n:
        epochs = [int(round(e)) for e in raw_epochs]
    else:
        epochs = list(range(1, n + 1))

    loss_series = []
    if t_box:
        loss_series.append({"name": "Train Box Loss", "color": "#1565c0", "data": t_box})
    if v_box:
        loss_series.append({"name": "Val Box Loss", "color": "#ef4444", "data": v_box})
    if t_cls:
        loss_series.append({"name": "Train Cls Loss", "color": "#0ea5e9", "data": t_cls})
    if v_cls:
        loss_series.append({"name": "Val Cls Loss", "color": "#f59e0b", "data": v_cls})
    if not loss_series and t_dfl:
        loss_series.append({"name": "Train DFL Loss", "color": "#1565c0", "data": t_dfl})

    metric_series = []
    if map50:
        metric_series.append({"name": "mAP@0.5", "color": "#1565c0", "data": [round(v * 100, 2) for v in map50]})
    if recall:
        metric_series.append({"name": "Recall", "color": "#10b981", "data": [round(v * 100, 2) for v in recall]})
    if precision:
        metric_series.append({"name": "Precision", "color": "#f59e0b", "data": [round(v * 100, 2) for v in precision]})
    if map50_95:
        metric_series.append({"name": "mAP@0.5:0.95", "color": "#78879b", "data": [round(v * 100, 2) for v in map50_95]})

    return {
        "available": True,
        "source": source_label,
        "epochs": epochs,
        "summary": {
            "total_epochs": len(epochs),
            "best_map50": round(max(map50) * 100, 2) if map50 else None,
            "best_map50_95": round(max(map50_95) * 100, 2) if map50_95 else None,
            "best_recall": round(max(recall) * 100, 2) if recall else None,
            "best_precision": round(max(precision) * 100, 2) if precision else None,
            "min_val_box_loss": round(min(v_box), 4) if v_box else (round(min(t_box), 4) if t_box else None),
        },
        "loss_chart": {
            "title": "Kayıp Eğrileri (Box & Cls Loss)",
            "unit": "",
            "series": loss_series,
        },
        "metrics_chart": {
            "title": "Doğruluk Metrikleri (mAP@0.5, Recall, Precision %)",
            "unit": "%",
            "y_min": 0,
            "y_max": 100,
            "series": metric_series,
        },
    }


@app.get("/api/training-metrics")
async def get_training_metrics():
    """Yalnızca gerçek YOLO eğitim verilerini (models/results.csv veya best.pt içi kayıt) döndürür."""
    # 1. Öncelik: models/results.csv dosyası
    if RESULTS_CSV_PATH.exists():
        try:
            with RESULTS_CSV_PATH.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols: Dict[str, List[float]] = {}
                for row in reader:
                    for k, v in row.items():
                        if not k or v is None:
                            continue
                        ck = k.strip()
                        try:
                            val = float(v.strip())
                            cols.setdefault(ck, []).append(val)
                        except ValueError:
                            continue
                parsed = _parse_metrics_rows(cols, "models/results.csv")
                if parsed.get("available"):
                    return parsed
        except Exception:
            pass

    # 2. Öncelik: Yüklenen best.pt dosyasının içinde gömülü train_results geçmişi varsa
    if state.detector.embedded_train_results:
        parsed = _parse_metrics_rows(
            state.detector.embedded_train_results,
            f"{state.detector.model_path.name} (Model İçi Eğitim Kaydı)",
        )
        if parsed.get("available"):
            return parsed

    # Gerçek veri henüz yüklenmemişse sahte grafik çizme!
    return {
        "available": False,
        "best_metrics": state.detector.checkpoint_meta.get("best_metrics", {}),
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
