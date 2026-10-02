FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd --create-home --uid 10001 app
WORKDIR /app

# Dependencies first for layer caching; dev and train extras are not installed.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# No data or weights are baked in: mount the model read-only at /models.
ENV PATH="/app/.venv/bin:$PATH" \
    LIVENESS_API_MODEL_PATH=/models/model.onnx
VOLUME ["/models"]
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"]

# Thresholds (LIVENESS_API_DECISION_THRESHOLD, ...) are required env vars; see settings.py.
CMD ["uvicorn", "--factory", "liveness.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
