# syntax=docker/dockerfile:1.7
# Voice MCP Server image. Build with --build-arg VARIANT=cuda (default) or VARIANT=cpu.
# EXTRAS selects optional features (denoise = DeepFilterNet3 ONNX on onnxruntime).
ARG VARIANT=cuda
ARG EXTRAS="--extra tts --extra denoise --extra s3"

FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM debian:bookworm-slim AS builder
ARG VARIANT
ARG EXTRAS
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/voice-mcp/.venv
WORKDIR /opt/voice-mcp
RUN uv python install 3.14
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS} --extra ${VARIANT}
COPY alembic.ini ./
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${EXTRAS} --extra ${VARIANT}

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime-cuda
FROM debian:bookworm-slim AS runtime-cpu

FROM runtime-${VARIANT} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates libsndfile1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 voice \
    && useradd --system --uid 10001 --gid voice --home-dir /srv/voice-mcp --shell /usr/sbin/nologin voice \
    && mkdir -p /srv/voice-mcp && chown voice:voice /srv/voice-mcp
COPY --from=builder /opt/python /opt/python
COPY --from=builder --chown=voice:voice /opt/voice-mcp /opt/voice-mcp
COPY --chmod=0755 deploy/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENV PATH=/opt/voice-mcp/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf \
    XDG_CACHE_HOME=/tmp/cache \
    NUMBA_CACHE_DIR=/tmp/numba \
    DATA_ROOT=/srv/voice-mcp \
    VOICE_MCP_SERVER__HOST=0.0.0.0 \
    VOICE_MCP_OBSERVABILITY__LOG_FORMAT=json
WORKDIR /opt/voice-mcp
USER voice
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8000), 3).close()" || exit 1
ENTRYPOINT ["tini", "--", "entrypoint.sh"]
CMD ["serve"]
