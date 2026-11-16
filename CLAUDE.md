# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Sezgen is a FastAPI + Ultralytics YOLO (+ SAHI) web app that detects Foreign Object Debris (FOD) on airport runways from uploaded video/images or RTSP/USB camera streams. UI strings, logs and code comments are in Turkish.

## Commands

- Run locally: `python main.py` → http://localhost:8000. Env vars: `SEZGEN_HOST`, `SEZGEN_PORT`, `SEZGEN_RELOAD=true` (uvicorn reload).
- Install: `pip install -r requirements.txt` (a `.venv` exists in the repo dir).
- Docker (CPU): `docker compose up --build -d`; GPU: `docker compose --profile gpu up --build -d` (build arg `TORCH_INDEX_URL` selects the torch wheel index).
- Health check: `GET /api/health`.
- No test framework or linter is configured. `scripts/verify.py [test_dir]` is an ad-hoc assertion script that drives `src.web.app.state` directly (requires a model in `models/`; it references `test/IMG_5572_270.jpg`, which is not in the repo — only `test/DSC05397.JPG` is). `scripts/diag.py` is a diagnostic helper.
- Dataset/training: `src/voc_to_yolo.py` converts FOD-A Pascal VOC (`dataset/FODPascalVOCFormat-V.2.1`) to YOLO format under `data/yolo`; `configs/data.yaml` defines the 31 classes; training happens in `notebooks/FOD_YOLO_Training.ipynb`.

## Architecture

- `main.py` just launches uvicorn on `src.web.app:app`.
- `src/web/app.py` holds a single global `SystemState` (`state`) containing the detector, the OpenCV capture, logs, `found_items`, `detection_history` and tracker state. A daemon thread (`_playback_worker`) reads frames, runs detection and publishes annotated JPEGs, which `/api/stream` serves as MJPEG (synchronised via `frame_cond`). The frontend (`templates/index.html`, `static/js/app.js`) polls `/api/state` and calls the `/api/*` control endpoints. Shared state is guarded by `state.lock` (RLock).
- Detection dedup: `_record_detections` / `_match_track` treat detections of the same class with IoU ≥ 0.3 within a frame gap as one physical object (one record, highest-confidence frame kept). Single-image re-analysis (`analyze_current_image`) replaces results instead of accumulating.
- `src/detector.py` `FODDetector`: loads `.pt`/`.onnx` via Ultralytics and, for `.pt`, reads the checkpoint (`train_args`, `train_metrics`, `train_results`) to set imgsz and provide embedded training curves. Two modes: standard full-frame inference, or "deep scan" (SAHI slicing, falling back to `_infer_manual_tiles` on any SAHI error). Both go through `postprocess_boxes` (conf filter, agnostic NMS, containment suppression). Without a loaded model it returns the raw frame and no detections.
- Image inputs force deep scan (`enforce_mode_for_media`); the user's previous mode is restored for video/stream.
- `src/risk.py` maps FOD-A class names → Turkish labels, risk level (Yüksek/Orta/Düşük), box colours and runway sector. Its class list must stay consistent with `configs/data.yaml` and the trained model.
- `src/exporter.py` builds CSV/PDF reports from `detection_history`.
- `/api/training-metrics` prefers `models/results.csv` (`RESULTS_CSV_PATH`) and falls back to training history embedded in the loaded `.pt`.
- `src/config.py` reads env vars (`MODEL_PATH`, `RESULTS_CSV_PATH`, `DEFAULT_CONF`, `DEFAULT_IOU`, `DEFAULT_ALARM_THRESH`) and creates `models/`, `uploads/`, `reports/` (mounted as volumes in docker-compose). Weights (`.pt`) are git-ignored/uploaded at runtime.

## Conventions

- The project deliberately contains no simulated/fake data: detections, logs and charts must come from a real model and real inputs.
- `/api/settings` accepts thresholds as either 0–1 fractions or percentages (values > 1 are divided by 100).
