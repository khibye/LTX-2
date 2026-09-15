FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models/.cache \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-dev python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir uv \
    && ln -s /opt/venv/bin/uv /usr/local/bin/uv

COPY pyproject.toml README.md LICENSE* ./
COPY packages ./packages

RUN uv sync --no-cache --group api --no-dev

COPY api.py ./api.py

VOLUME ["/models", "/app/output"]

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]