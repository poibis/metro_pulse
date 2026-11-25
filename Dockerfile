# syntax=docker/dockerfile:1.7

FROM python:3.14-slim-bookworm AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ==========================================
# Stage 1: Builder (uv로 의존성 및 패키지 컴파일)
# ==========================================
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1) 서드파티 패키지 캐시 레이어
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# 2) 애플리케이션 패키지 설치 (--no-editable)
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ==========================================
# Stage 2: Runtime (경량 프로덕션 실행 환경)
# ==========================================
FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Builder의 가상환경과 앱 실행에 필요한 정적/데이터 자산 복사
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --chown=appuser:appgroup data ./data
COPY --chown=appuser:appgroup artifacts ./artifacts
COPY --chown=appuser:appgroup frontend ./frontend

ENV PATH="/app/.venv/bin:$PATH" \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

USER appuser
EXPOSE 8000

CMD ["uvicorn", "metro_pulse.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]