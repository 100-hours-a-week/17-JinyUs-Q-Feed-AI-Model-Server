# syntax=docker/dockerfile:1.6

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 && \
    uv sync --frozen --no-dev --python 3.12

COPY . .

EXPOSE 8002

CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8002"]
