# syntax=docker/dockerfile:1.9
FROM python:3.12-slim

# Inherit build arguments for labels
ARG GRAPHITI_VERSION
ARG BUILD_DATE
ARG VCS_REF

# OCI image annotations
LABEL org.opencontainers.image.title="Graphiti FastAPI Server"
LABEL org.opencontainers.image.description="FastAPI server for Graphiti temporal knowledge graphs"
LABEL org.opencontainers.image.version="${GRAPHITI_VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="Zep AI"
LABEL org.opencontainers.image.source="https://github.com/getzep/graphiti"
LABEL org.opencontainers.image.documentation="https://github.com/getzep/graphiti/tree/main/server"
LABEL io.graphiti.core.version="${GRAPHITI_VERSION}"

# Install uv using the installer script
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

# Configure uv for runtime
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Create non-root user
RUN groupadd -r app && useradd -r -d /app -g app app

# Set up the server application first
WORKDIR /app

# Copy the local graphiti_core package first
COPY ./graphiti_core ./graphiti_core
COPY ./pyproject.toml ./py.typed ./README.md ./

# Copy server files
COPY ./server/pyproject.toml ./server/README.md ./server/uv.lock ./server/
COPY ./server/graph_service ./server/graph_service

# Install server dependencies first, then override with local graphiti_core
# This ensures we use the local code with gpt-5 support instead of PyPI version
ARG INSTALL_FALKORDB=false
RUN cd /app/server && \
    uv sync --frozen --no-dev && \
    rm -rf /app/server/.venv/lib/python3.12/site-packages/graphiti_core && \
    rm -rf /app/server/.venv/lib/python3.12/site-packages/graphiti_core-*.dist-info && \
    cd /app && \
    find graphiti_core -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    cp -r /app/graphiti_core /app/server/.venv/lib/python3.12/site-packages/ && \
    grep "is_reasoning_model" /app/server/.venv/lib/python3.12/site-packages/graphiti_core/llm_client/openai_client.py && \
    python -m compileall /app/server/.venv/lib/python3.12/site-packages/graphiti_core

# Set the working directory to server for runtime
WORKDIR /app/server

# Change ownership to app user
RUN chown -R app:app /app

# Set environment variables - use the server's venv, not /app/.venv
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/server/.venv/bin:$PATH"

# Switch to non-root user
USER app

# Set port
ENV PORT=8000
EXPOSE $PORT

# Run uvicorn directly from the venv without uv (which would resync and overwrite our local graphiti_core)
# Use multiple workers so long-running community builds do not starve health checks and searches.
ENV UVICORN_WORKERS=2
CMD ["sh", "-c", "python -m uvicorn graph_service.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]
