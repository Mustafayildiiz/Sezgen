# Sezgen - Havalimanı Pistlerinde Yabancı Cisim (FOD) Tespiti Sistemi
# Çok aşamalı derleme: bağımlılıklar izole bir sanal ortamda (/opt/venv) kurulur,
# çalışma imajına yalnızca bu venv kopyalanır.

ARG PYTHON_VERSION=3.11

# ---------- 1) Builder: sanal ortam + bağımlılıklar ----------
FROM python:${PYTHON_VERSION}-slim AS builder

# CPU için varsayılan; GPU için: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Önce yalnızca requirements.txt kopyalanır: kod değişince bu katman yeniden kurulmaz (cache)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --extra-index-url ${TORCH_INDEX_URL} -r requirements.txt

# ---------- 2) Runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SEZGEN_HOST=0.0.0.0 \
    SEZGEN_PORT=8000 \
    MODEL_PATH=/app/models/best.pt

WORKDIR /app

# Uygulama kodu en sonda kopyalanır
COPY . .
RUN mkdir -p models reports uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "main.py"]
