FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config.yaml ./config.yaml
COPY scripts ./scripts
COPY tests ./tests
COPY pytest.ini ./pytest.ini

# match the host user so the bind-mounted ./data volume is writable
ARG APP_UID=1000
RUN useradd -m -u ${APP_UID} appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /srv/app /data
USER appuser

# model caches live under /data so they persist in the shared compose volume
ENV HF_HOME=/data/models/hf \
    XDG_DATA_HOME=/data/models/xdg \
    XDG_CACHE_HOME=/data/models/cache \
    SPEECHBRAIN_CACHE=/data/models/speechbrain \
    CONFIG_FILE=config.yaml

# Bound per-process math threads. torch/CTranslate2/BLAS each default to "all
# cores"; with N celery children that is N*cores threads fighting over cores, and
# concurrent cold model loads degrade badly (measured: 137s vs 1.3s for the same
# translate stage). Rule of thumb: cores / worker-concurrency.
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
