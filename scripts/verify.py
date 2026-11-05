"""Doğrulama: conf / IoU / alarm eşikleri, tekilleştirme, sayaç ve görsel-mod kuralı."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.web.app import state  # noqa: E402

TEST = Path(sys.argv[1] if len(sys.argv) > 1 else "test")
img = TEST / "IMG_5572_270.jpg"
fails = []


def check(name, cond, info=""):
    print(("PASS" if cond else "FAIL"), name, info)
    if not cond:
        fails.append(name)


d = state.detector
d.update_settings(conf=0.25, iou=0.65, alarm=0.85, deep_scan=False)
st = state.load_media_file(img)
check("görselde deep_scan zorunlu", st["settings"]["deep_scan"] and not st["settings"]["live_allowed"])
n1 = st["telemetry"]["total_detections"]
check("tek nesne sayısı makul (<=3)", 1 <= n1 <= 3, f"count={n1} items={[i['class_name'] for i in st['found_items']]}")

for _ in range(5):
    state.analyze_current_image()
n2 = len(state.detection_history)
check("tekrar analizde sayaç birikmiyor", n2 == n1, f"{n1}->{n2}")
check("found_items == history", len(state.found_items) == n2)

d.update_settings(conf=0.95)
state.analyze_current_image()
check("conf=0.95 -> 0 tespit", len(state.found_items) == 0 and not state.alarm_active)

d.update_settings(conf=0.3, alarm=0.99)
state.analyze_current_image()
check("alarm=0.99 -> alarm yok", len(state.found_items) > 0 and not state.alarm_active)

d.update_settings(alarm=0.3)
state.analyze_current_image()
check("alarm=0.3 -> alarm var", state.alarm_active)

# IoU: çok düşük IoU daha agresif birleştirir -> sayı artmamalı
d.update_settings(conf=0.25, iou=0.65, alarm=0.85)
state.analyze_current_image()
hi = len(state.found_items)
d.update_settings(iou=0.05)
state.analyze_current_image()
lo = len(state.found_items)
check("düşük IoU eşiği sayıyı artırmaz", lo <= hi, f"iou0.65={hi} iou0.05={lo}")

# Her kutu conf eşiğinin üstünde ve çiftler arası IoU < eşik
d.update_settings(conf=0.25, iou=0.5)
state.analyze_current_image()
b = [i["bbox"] for i in state.found_items]
ok = all(state._iou(b[i], b[j]) < 0.5 for i in range(len(b)) for j in range(i + 1, len(b)))
check("kalan kutular arası IoU < eşik", ok)
check("tüm conf >= eşik", all(i["confidence"] >= 0.25 for i in state.found_items))

# Video modu: önceki tercih geri gelir (test klasöründe video yoksa atla)
state.enforce_mode_for_media("video")
check("video girişinde canlı akış serbest", not d.deep_scan)

print("\nSONUÇ:", "TÜMÜ GEÇTİ" if not fails else f"BAŞARISIZ: {fails}")
sys.exit(1 if fails else 0)

