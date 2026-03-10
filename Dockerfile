# ---------------------------------------------------------------------------
# Stage 1 – build / install dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools required for some C-extension wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch to keep image size manageable; override in
# production builds that target a GPU node with:
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url "${TORCH_INDEX}" \
        -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 – runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd -r beamctl && useradd -r -g beamctl beamctl

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ /app/src/
COPY setup.py /app/
COPY README.md /app/

# Install the package in editable mode so imports resolve correctly
RUN pip install --no-cache-dir -e /app

# Environment defaults (override at runtime via -e flags or k8s ConfigMap)
ENV PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL="INFO" \
    DEVICE="cpu" \
    SNR_THRESHOLD_DB="5.0" \
    STEP_INTERVAL_MS="500"

USER beamctl

# Expose metrics port (Prometheus scrape)
EXPOSE 9090

# Health-check endpoint (simple Python process liveness)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.inference.online_controller" || exit 1

# Default entrypoint – override in Kubernetes with the actual inference script
CMD ["python", "-m", "src.inference"]
