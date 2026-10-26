# syntax=docker/dockerfile:1
# AeroGuardian :: Autonomous Multi-Spectral Search-and-Rescue Aerial Triad
# Targeted for AWS Graviton (linux/arm64) and standard x86_64 edge platforms

FROM python:3.12-slim-bookworm

LABEL maintainer="Abhishek Khanra <ishekofficial@gmail.com>"
LABEL project="AeroGuardian OpenCV AI Competition 2026"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install essential system runtime libraries for OpenCV Headless and network sockets
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code and benchmarks
COPY src/ ./src/
COPY models/ ./models/
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY data/samples/ ./data/samples/

# Expose Web Dashboard HUD (HTTP) and MAVLink 2.0 Telemetry Socket (UDP)
EXPOSE 8000
EXPOSE 14550/udp

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Launch production ASGI server
CMD ["uvicorn", "src.dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
