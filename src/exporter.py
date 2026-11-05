"""Sezgen gerçek FOD tespit kayıtlarını CSV ve PDF formatında dışa aktarma modülü."""
import csv
import io
from datetime import datetime
from typing import Any, Dict, List


def export_detections_csv(detections: List[Dict[str, Any]]) -> bytes:
    """Gerçek tespit geçmişini UTF-8 BOM destekli CSV byte dizisi olarak döndürür."""
    buf = io.StringIO()
    buf.write("\ufeff")  # Excel Türkçe karakter uyumu
    writer = csv.writer(buf)
    writer.writerow([
        "Saat",
        "Kaynak Dosya",
        "Video Süresi",
        "Kare No",
        "Sınıf (Model)",
        "Cisim Türü (TR)",
        "Pist Bölgesi",
        "Güven Skoru (%)",
        "Risk Seviyesi",
        "Koordinatlar [x1, y1, x2, y2]",
        "Tarama Modu",
        "Model",
    ])
    for d in detections:
        writer.writerow([
            d.get("timestamp", ""),
            d.get("source_name", "-"),
            d.get("video_time", "-"),
            d.get("frame_idx", 0),
            d.get("class_name", ""),
            d.get("tr_label", ""),
            d.get("sector", ""),
            f"{float(d.get('confidence', 0.0)) * 100:.1f}",
            d.get("risk", ""),
            str(d.get("bbox", [])),
            d.get("mode", ""),
            d.get("model_name", ""),
        ])
    return buf.getvalue().encode("utf-8")


def export_detections_pdf(detections: List[Dict[str, Any]]) -> bytes:
    """Gerçek tespit geçmişini PDF raporu olarak oluşturur."""
    try:
        from fpdf import FPDF
    except ImportError:
        return export_detections_csv(detections)

    def _ascii_tr(text: Any) -> str:
        tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        return str(text).translate(tr_map)

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, _ascii_tr("SEZGEN - Havalimani Pist FOD Gercek Tespit Raporu"), ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(
        0,
        7,
        _ascii_tr(f"Rapor Tarihi: {now_str} | Toplam Tespit Sayisi: {len(detections)}"),
        ln=True,
        align="C",
    )
    pdf.ln(3)

    headers = ["Saat", "Kaynak", "Video Zamani", "Kare", "Cisim Turu", "Guven", "Risk", "Koordinat", "Mod"]
    widths = [22, 42, 24, 18, 44, 20, 20, 50, 35]

    pdf.set_fill_color(11, 101, 198)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 8, _ascii_tr(h), border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_text_color(20, 25, 35)
    pdf.set_font("Helvetica", "", 8.5)
    for idx, d in enumerate(detections):
        fill = idx % 2 == 1
        if fill:
            pdf.set_fill_color(238, 243, 248)
        row = [
            d.get("timestamp", "-"),
            d.get("source_name", "-"),
            d.get("video_time", "-"),
            str(d.get("frame_idx", "-")),
            d.get("tr_label", d.get("class_name", "-")),
            f"%{float(d.get('confidence', 0.0)) * 100:.1f}",
            d.get("risk", "-"),
            str(d.get("bbox", "-")),
            d.get("mode", "-"),
        ]
        for val, w in zip(row, widths):
            pdf.cell(w, 7, _ascii_tr(val)[:26], border=1, align="C", fill=fill)
        pdf.ln()

    return bytes(pdf.output())
