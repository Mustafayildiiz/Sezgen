"""Sezgen - Streamlit Alternatif Operatör Paneli (Figma Tasarım Uyumlu)."""
from datetime import datetime
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import UPLOADS_DIR
from src.detector import FODDetector
from src.exporter import export_detections_csv, export_detections_pdf

st.set_page_config(
    page_title="Sezgen — Havalimanı FOD Tespit Sistemi",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "detector" not in st.session_state:
    st.session_state.detector = FODDetector()
if "logs" not in st.session_state:
    st.session_state.logs = [
        "[12:00:00] Sistem başlatıldı.",
        "[12:00:05] Video akışı alındı.",
        "[12:00:10] FOD tespiti yapıldı.",
    ]
if "history" not in st.session_state:
    st.session_state.history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

top_l, top_r = st.columns([8, 2])
with top_l:
    st.markdown("## ✈️ **Sezgen**")
with top_r:
    st.session_state.dark_mode = st.toggle(
        "☀️ Karanlık tema" if st.session_state.dark_mode else "☀️ Aydınlık tema",
        value=st.session_state.dark_mode,
    )

col_left, col_center, col_right = st.columns([2.4, 5.8, 1.8], gap="medium")

with col_left:
    st.markdown("### Video Yükleyiniz")
    uploaded = st.file_uploader(
        "Dosya sürükleyip bırakın veya yükleyin",
        type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
    )
    deep_scan = st.toggle("Derin Tarama (SAHI)", value=False, help="Açık: Derin Tarama | Kapalı: Canlı Akış")
    esik_1 = st.slider("Eşik 1 (Güven %)", 0, 100, 44)
    esik_2 = st.slider("Eşik 2 (IoU %)", 0, 100, 65)
    esik_3 = st.slider("Eşik 3 (Alarm %)", 0, 100, 85)
    run_btn = st.button("Başla", type="primary", use_container_width=True)

detector: FODDetector = st.session_state.detector
detector.update_settings(
    conf=esik_1 / 100.0,
    iou=esik_2 / 100.0,
    alarm=esik_3 / 100.0,
    deep_scan=deep_scan,
)

frame_bgr = detector.render_synthetic_runway(width=960, height=720)
if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    raw_bytes = uploaded.read()
    if suffix in {".jpg", ".jpeg", ".png"}:
        arr = np.frombuffer(raw_bytes, np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is not None:
            frame_bgr = decoded
    else:
        tmp_path = UPLOADS_DIR / uploaded.name
        tmp_path.write_bytes(raw_bytes)
        cap = cv2.VideoCapture(str(tmp_path))
        ok, f = cap.read()
        cap.release()
        if ok and f is not None:
            frame_bgr = f

annotated_bgr, detections, fps = detector.detect_frame(frame_bgr)
if run_btn and detections:
    now_t = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{now_t}] FOD tespiti yapıldı ({len(detections)} nesne).")
    st.session_state.history.extend(detections)

with col_center:
    st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.markdown("#### Sistem Günlükleri")
    st.code("\n".join(st.session_state.logs[-10:]), language="text")
    c_csv, c_pdf = st.columns(2)
    with c_csv:
        st.download_button(
            "CSV Rapor İndir",
            data=export_detections_csv(detections),
            file_name="sezgen_fod_rapor.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c_pdf:
        st.download_button(
            "PDF Rapor İndir",
            data=export_detections_pdf(detections),
            file_name="sezgen_fod_rapor.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

with col_right:
    st.markdown("#### Bulunan Eşyalar")
    for d in detections:
        st.markdown(f"**{d['tr_label']}**  \n`{d['sector']}` — `%{int(d['confidence']*100)}` ({d['risk']})")

st.markdown("---")
st.markdown("#### Eğitim Grafikleri")
g1, g2 = st.columns(2)
chart_df = pd.DataFrame(
    {
        "Mavi Seri": [925, 845, 765, 920, 810, 625, 815, 730, 520, 860, 740, 845, 645, 895, 855, 795, 685, 550, 385, 415],
        "Gri Seri": [1100, 815, 770, 1020, 890, 695, 860, 715, 515, 980, 675, 915, 620, 875, 960, 1035, 685, 540, 260, 475],
    }
)
with g1:
    st.line_chart(chart_df)
with g2:
    st.line_chart(chart_df)
