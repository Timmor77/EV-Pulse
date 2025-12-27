# Lightweight and recent Python image
FROM python:3.10-slim

# Install curl to download uv
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (ultra-fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Working directory
WORKDIR /app

COPY pyproject.toml uv.lock ./

# --- FIX HERE ---
# Remove '--system'. uv will create a .venv folder in /app
# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Run sync (creates .venv)
RUN uv sync --frozen --no-install-project

# CRITICAL: Add .venv to system PATH
# This way, when running "python" or "uvicorn", it will use the venv version automatically
ENV PATH="/app/.venv/bin:$PATH"
# ----------------------

# 3. Copy code and models
COPY src/ ./src/

# Python environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command (will be overridden by docker-compose)
CMD ["python"]