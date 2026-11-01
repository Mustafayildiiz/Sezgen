"""Sezgen FOD tespit kayıtlarını CSV ve PDF formatında dışa aktarma modülü."""
import csv
import io
from datetime import datetime
from typing import Any, Dict, List


def export_detections_csv(detections: List[Dict[str, Any]]) -> bytes:
    """Tespit geçmişini UTF-8 BOM destekli CSV byte dizisi olarak döndürür."""
    buf = io.StringIO()
    buf.write("\ufeff")  # Excel Türkçe karakter uyumu için UTF-8 BOM
    writer = csv.writer(buf)
    writer.writerow([
        "Tespit Saati",
        "Sınıf (EN)",
        "Cisim Türü (TR)",
        "Pist Bölgesi",
        "Güven Skoru (%)",
        "Risk Seviyesi",
        "Koordinatlar (x1,y1,x2,y2)",
        "Tarama Modu",
    ])
    for d in detections:
        writer.writerow([
            d.get("timestamp", ""),
            d.get("class_name", ""),
            d.get("tr_label", ""),
            d.get("sector", "Pist 12L"),
            f"{float(d.get('confidence', 0.0)) * 100:.1f}",
            d.get("risk", "Orta"),
            str(d.get("bbox", [])),
            d.get("mode", "Canlı Akış"),
        ])
    return buf.getvalue().encode("utf-8")


def export_detections_pdf(detections: List[Dict[str, Any]]) -> bytes:
    """Tespit geçmişini kurumsal PDF raporu olarak oluşturur."""
    try:
        from fpdf import FPDF
    except ImportError:
        # fpdf2 kurulu değilse düz metin tabanlı minimal PDF veya bilgi döndür
        return export_detections_csv(detections)

    def _ascii_tr(text: str) -> str:
        """Standart PDF fontları için Türkçe karakterleri güvenli eşdeğerlerine dönüştürür."""
        tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        return str(text).translate(tr_map)

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _ascii_tr("SEZGEN - Havalimani Pist FOD Tespit Raporu"), ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 7, _ascii_tr(f"Rapor Olusturma Tarihi: {now_str} | Toplam Tespit: {len(detections)}"), ln=True, align="C")
    pdf.ln(4)

    headers = ["Saat", "Cisim Turu", "Sinif", "Pist", "Guven (%)", "Risk", "Koordinat", "Mod"]
    widths = [25, 48, 42, 26, 25, 25, 52, 32]

    pdf.set_fill_color(11, 101, 198)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 8, _ascii_tr(h), border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_text_color(20, 25, 35)
    pdf.set_font("Helvetica", "", 9)
    for idx, d in enumerate(detections):
        fill = idx % 2 == 1
        if fill:
            pdf.set_fill_color(238, 243, 248)
        row = [
            d.get("timestamp", "-"),
            d.get("tr_label", "-"),
            d.get("class_name", "-"),
            d.get("sector", "Pist 12L"),
            f"%{float(d.get('confidence', 0.0)) * 100:.1f}",
            d.get("risk", "Orta"),
            str(d.get("bbox", "-")),
            d.get("mode", "Canli Akis"),
        ]
        for val, w in zip(row, widths):
            pdf.cell(w, 7, _ascii_tr(val)[:26], border=1, align="C", fill=fill)
        pdf.ln()

    out = pdf.output()
    return bytes(out)
