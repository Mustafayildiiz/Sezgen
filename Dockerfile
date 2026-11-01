# Sezgen - Havalimanı Pistlerinde Yabancı Cisim (FOD) Tespiti Sistemi
FROM python:3.11-slim

# Sistem bağımlılıkları (OpenCV ve görüntü işleme için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python bağımlılıklarını kur
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Gerekli klasörleri oluştur
RUN mkdir -p models reports uploads

# Web Arayüzü (FastAPI: 8000, Streamlit: 8501)
EXPOSE 8000 8501

ENV PYTHONUNBUFFERED=1 \
    SEZGEN_HOST=0.0.0.0 \
    SEZGEN_PORT=8000 \
    MODEL_PATH=/app/models/best.pt

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "main.py"]
