# === Stage 1: Builder (Fix #23 — multi-stage build) ===
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# === Stage 2: Runtime ===
FROM python:3.11-slim AS runtime

# Fix #23: non-root user
RUN useradd -m -u 1000 appuser

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code (Security M5: no stale data files)
COPY openenv.yaml .
COPY environment/ ./environment/
COPY inference.py .
# inference_advanced.py intentionally excluded: judge uses inference.py only
COPY README.md .

# Initialize empty leaderboard (M5: don't copy stale test data)
RUN echo "[]" > leaderboard.json

# Fix ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 7860

# L1: disable server header to prevent tech fingerprinting
# M6: document TLS — use reverse proxy (nginx/Caddy) or set --ssl-keyfile/--ssl-certfile
CMD ["uvicorn", "environment.server:app", "--host", "0.0.0.0", "--port", "7860", "--server-header", "--no-server-header"]
