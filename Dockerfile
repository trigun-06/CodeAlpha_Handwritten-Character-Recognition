# Multi-stage production container for TextReader AI
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=/app/emnist_alphanumeric_cnn.keras

WORKDIR /app

# Install minimal OS dependencies for OpenCV headless and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files and model
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY emnist_alphanumeric_cnn.keras .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

EXPOSE 8000

# Start server respecting dynamic $PORT from cloud providers (Render, Railway, Cloud Run)
CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}
