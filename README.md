# ✈️ Sezgen — Havalimanı Pistlerinde Yabancı Cisim (FOD) Tespiti Sistemi

**Sezgen**, havalimanı pist ve taksi yollarındaki yabancı cisimleri (**Foreign Object Debris - FOD**: cıvata, somun, metal parça, plastik atık, taş, kablo vb.) **YOLOv26** ve **SAHI** (*Slicing Aided Hyper Inference*) kullanarak gerçek video/kamera akışları üzerinden tespit eden operasyonel izleme ve raporlama sistemidir.

---

## 🖼️ Arayüz Tasarımı

Operatör paneli **Aydınlık Tema** ve **Karanlık Tema** seçeneklerine sahiptir:

| Aydınlık Tema | Karanlık Tema |
| :---: | :---: |
| ![Aydınlık Tema](assets/figma/light_theme.png) | ![Karanlık Tema](assets/figma/dark_theme.png) |

---

## 🚀 Sistem Bileşenleri

1. **Gerçek Video & Kamera Oynatıcı:**
   - Yüklenen pist videolarını (`.mp4, .avi, .mov, .mkv, .webm`), görselleri (`.jpg, .png`) veya canlı **RTSP / USB Kamera** akışlarını kare kare işler.
   - İnteraktif zaman çubuğu (**Seek**), **Başla / Duraklat**, **Başa Sar**, **+1 Kare Adımlama** ve **Tam Ekran** kontrolleri içerir.
2. **YOLOv26 Model Ayarları (`best.pt` / `.onnx`):**
   - Eğittiğiniz modelin `best.pt` veya `best.onnx` ağırlık dosyasını doğrudan arayüzdeki **Model Ayarları** panelinden yükleyebilir ya da `models/best.pt` dizinine kopyalayabilirsiniz.
   - Çıkarım çözünürlüğü (`imgsz`: `320` – `1280`), **Canlı Akış** / **Derin Tarama (SAHI)** geçişi, **Güven Eşiği (Conf)**, **Çakışma Eşiği (IoU)** ve **Alarm Eşiği** gerçek zamanlı değiştirilebilir.
3. **Bulunan Eşyalar (Gerçek Kırpılmış Görseller & Kareye Atlama):**
   - Videodan tespit edilen her yabancı cismin gerçek kare kırpımı (*crop thumbnail*), sınıfı, güven skoru, risk seviyesi (`Yüksek` / `Orta` / `Düşük`), video zamanı ve kare numarası listelenir.
   - Listeden herhangi bir cisme tıklandığında video doğrudan o kareye sarılır.
4. **Sistem Günlükleri & CSV / PDF Raporlama:**
   - Tüm sistem ve tespit olayları gerçek zaman damgalarıyla kaydedilir; tek tıkla **CSV** veya **PDF** raporu olarak indirilir.
5. **Gerçek Eğitim Grafikleri:**
   - Eğitim çıktısı olan `results.csv` dosyasından veya `best.pt` içerisindeki gömülü eğitim geçmişinden (`train_results`) gerçek **Box/Cls Loss** ve **mAP@0.5 / Recall / Precision** grafiklerini interaktif olarak çizer.

---

## 📂 Proje Dizin Yapısı

```text
Sezgen/
├── assets/figma/            # Figma Aydınlık ve Karanlık tema tasarım referansları
├── configs/
│   └── data.yaml            # FOD-A v2.1 (31 sınıf) veri seti yapılandırması
├── models/                  # Eğitilen YOLOv26 ağırlıkları (best.pt / best.onnx / results.csv)
├── notebooks/
│   └── train_colab.ipynb    # YOLO model eğitim Jupyter Notebook dosyası
├── reports/                 # Dışa aktarılan CSV ve PDF FOD tespit raporları
├── src/
│   ├── config.py            # Sistem sabitleri ve varsayılan eşik ayarları
│   ├── detector.py          # YOLOv26 + SAHI gerçek çıkarım ve model inceleme motoru
│   ├── exporter.py          # Gerçek tespit kayıtları CSV ve PDF rapor üretici
│   ├── risk.py              # 31 FOD sınıfı -> Türkçe etiket, risk seviyesi ve pist eşlemesi
│   ├── voc_to_yolo.py       # FOD-A Pascal VOC (XML) -> YOLO (.txt) dönüştürücü
│   └── web/                 # Operasyonel Web Arayüzü (FastAPI + HTML5/CSS3/JS)
├── Dockerfile               # Konteyner imaj tanımı
├── docker-compose.yml       # CPU ve GPU (NVIDIA CUDA) destekli servis orkestrasyonu
├── main.py                  # Ana web sunucusu başlatıcı (Port 8000)
└── requirements.txt         # Python bağımlılıkları
```

---

## 🛠️ Kurulum ve Çalıştırma

### Docker ile Çalıştırma

```bash
# Standart (CPU) modunda başlat
docker compose up --build -d

# NVIDIA GPU (CUDA) hızlandırmalı modda başlatmak için:
docker compose --profile gpu up --build -d
```

Tarayıcıdan **`http://localhost:8000`** adresine giderek sistemi kullanabilirsiniz.
