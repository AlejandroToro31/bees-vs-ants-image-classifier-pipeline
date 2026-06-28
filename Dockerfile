# ==============================================================
# Bees vs Ants Classification API — Production Dockerfile
# ==============================================================
# ResNet18 binary classifier microservice.
# CPU inference, non-root execution.
#
# Build:
#   docker build -t bees-ants-api:v1 .
#
# Run:
#   docker run -p 8000:8000 -e API_SECRET_KEY=your_secret_key bees-ants-api:v1
# ==============================================================

# ── Base Image
FROM mirror.gcr.io/library/python:3.10-slim

# ── Image Metadata
LABEL version="1.1.0"
LABEL description="Bees vs Ants Classification API — ResNet18 transfer learning"

# ── Python Runtime Configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Application Configuration
# API_SECRET_KEY intentionally has NO default — server refuses to boot
# without it (see main.py). Must be passed via docker run -e.
ENV MODEL_PATH=models/best_model.pth

# ── OS-Level Dependencies
# Note: No libgl1/libglib2.0-0 needed here — this service uses PIL
# (Pillow) for image decoding, not OpenCV, so the GUI system libraries
# required by cv2 on slim Debian images are not applicable.
# curl: required for Docker HEALTHCHECK instruction.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Security: Non-Root User
RUN groupadd -r api_user && useradd -m -r -g api_user api_user

# ── Working Directory
WORKDIR /workspace
RUN chown api_user:api_user /workspace

# ── Layer Caching Strategy
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code
COPY --chown=api_user:api_user app/ app/
COPY --chown=api_user:api_user models/ models/

# ── Drop Privileges
USER api_user

# ── Port Declaration
EXPOSE 8000

# ── Health Monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
