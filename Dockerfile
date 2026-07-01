FROM python:3.11-slim as builder

# Set env vars to suppress bytecode and force unbuffered output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Final image ---
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create non-root user
RUN useradd --create-home appuser

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ /app/src/
COPY data/ /app/data/
COPY requirements.txt /app/

# We intentionally do not copy .env; secrets config must be provided by the runtime environment (e.g. Railway config)

# Change ownership
RUN chown -R appuser:appuser /app
USER appuser

# Expose standard port
EXPOSE 8000

# Start Uvicorn
CMD sh -c "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"
