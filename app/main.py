"""
Hymenoptera Classification API — Bees vs Ants
================================================
Production FastAPI microservice for binary classification of bees
and ants. The foundational deployment establishing the core inference
patterns applied at scale in later projects.

Features:
    - API key authentication via FastAPI native Security
    - Double payload validation (header + actual bytes) — detects spoofed headers
    - asyncio.to_thread for non-blocking inference
    - Model warmup on startup

Endpoints:
    GET  /              → API metadata
    GET  /health        → Liveness check
    GET  /ready         → Readiness check (model loaded)
    POST /api/v1/predict → Bees vs ants classification

Environment Variables:
    MODEL_PATH      : Path to best_model.pth (default: models/best_model.pth)
    API_SECRET_KEY  : Authentication key — REQUIRED, no default
"""

# ── Standard Library
import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional

# ── Third Party
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Header, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models, transforms


# ════════════════════════════════════════════════════════
# 1. LOGGING INFRASTRUCTURE
# ════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [Bees-Ants-API] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BeesAntsAPI")


# ════════════════════════════════════════════════════════
# 2. GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best_model.pth")

# Class indices must match training ImageFolder.class_to_idx.
# ImageFolder assigns alphabetically: {'ants': 0, 'bees': 1}
CLASS_NAMES = ["ants", "bees"]

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

# Singleton store — model loaded once at startup
ml_state: Dict = {}

# ── API Key Authentication
# No default value — server refuses to boot if key is not configured
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
if not API_SECRET_KEY:
    raise EnvironmentError(
        "API_SECRET_KEY environment variable is not set. "
        "Provide it via: docker run -e API_SECRET_KEY=your_secret_key"
    )

API_KEY_NAME = "Authorization-API-Key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Validates the incoming API key against the configured secret.
    Returns the key on success. Raises 401 on mismatch.
    """
    if api_key != API_SECRET_KEY:
        logger.warning("SECURITY: Unauthorized access attempt intercepted.")
        raise HTTPException(status_code=401, detail="Invalid API Key. Access Denied.")
    return api_key


# ── Preprocessing Pipeline
# CRITICAL: must match val_transform in the training script exactly.
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ════════════════════════════════════════════════════════
# 3. SERVER LIFESPAN — MODEL SINGLETON PATTERN
# ════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager — controls model lifecycle.

    STARTUP: Reconstructs ResNet18 architecture, loads trained weights,
             runs warmup inference.
    SHUTDOWN: Clears model state, releases GPU memory.
    """
    logger.info("Booting Hymenoptera Vision API...")
    logger.info(f"Device: {DEVICE.type.upper()} | Model: {MODEL_PATH}")

    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        ml_state["model"] = model
        logger.info("Model artifact loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model artifact: {e}")
        raise RuntimeError(
            f"Server boot aborted — artifact not found at: {MODEL_PATH}"
        ) from e

    # ── Warmup inference — compiles CUDA kernels before first real request
    logger.info("Running warmup inference...")
    dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
    with torch.inference_mode():
        ml_state["model"](dummy)
    logger.info("Model warmed up. API ready to serve requests.")

    yield

    logger.info("Shutting down. Releasing resources...")
    ml_state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete.")


# ════════════════════════════════════════════════════════
# 4. API INSTANTIATION
# ════════════════════════════════════════════════════════

app = FastAPI(
    title="Hymenoptera Classification API",
    description="Real-time Computer Vision endpoint for Bees vs Ants classification.",
    version="1.1.0",
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════
# 5. RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    """Inference result payload."""
    filename    : str
    prediction  : str    # "ants" or "bees"
    confidence  : float  # softmax probability of predicted class [0, 1]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status      : str
    model_path  : str
    device      : str


# ════════════════════════════════════════════════════════
# 6. UTILITY ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/", tags=["Utility"])
async def root() -> dict:
    """API metadata — entry point for documentation discovery."""
    return {
        "api"    : "Hymenoptera Classification API",
        "version": "1.1.0",
        "docs"   : "/docs",
        "health" : "/health",
        "ready"  : "/ready",
        "predict": "/api/v1/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health() -> HealthResponse:
    """Liveness check — confirms the API process is running."""
    return HealthResponse(status="healthy")


@app.get("/ready", response_model=ReadyResponse, tags=["Utility"])
async def ready() -> ReadyResponse:
    """Readiness check — confirms the model is loaded and inference is possible."""
    if ml_state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be initializing."
        )
    return ReadyResponse(
        status    ="ready",
        model_path=MODEL_PATH,
        device    =DEVICE.type.upper(),
    )


# ════════════════════════════════════════════════════════
# 7. INFERENCE HELPERS
# ════════════════════════════════════════════════════════

def _run_inference(model: nn.Module, tensor: torch.Tensor) -> tuple:
    """
    Synchronous inference function — runs in thread pool via asyncio.to_thread.

    Separated from the async endpoint to enable proper thread offloading.
    PyTorch inference is CPU/GPU-bound — running it directly in async def
    blocks the event loop, preventing other requests from being served.

    Returns:
        (confidence: float, predicted_idx: int)
    """
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
    return float(confidence[0]), int(predicted_idx[0])


# ════════════════════════════════════════════════════════
# 8. INFERENCE ENDPOINT
# ════════════════════════════════════════════════════════

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_image(
    file: UploadFile = File(...),
    api_key: str = Security(verify_api_key),
    content_length: int = Header(default=0, alias="Content-Length"),
) -> PredictionResponse:
    """
    Bees vs ants classification endpoint.

    Pipeline:
        1. API key validation          → reject unauthorized requests
        2. Header size check           → early rejection of oversized payloads
        3. MIME type validation        → reject non-image content types
        4. Actual bytes size check     → detect spoofed Content-Length headers
        5. In-memory image decoding    → zero disk I/O via io.BytesIO
        6. Thread-offloaded inference  → asyncio.to_thread (non-blocking)
        7. Structured response         → Pydantic-validated JSON payload

    Raises:
        401 : Invalid or missing API key
        400 : Invalid content type (non-image upload)
        413 : Payload exceeds 10MB limit (checked twice — header + actual)
        422 : Valid image type but content cannot be decoded (corrupted)
        500 : Unexpected inference error
        503 : Model not loaded
    """

    # ── Model availability check
    model = ml_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Server may still be initializing."
        )

    # ── Step 1: Header-based size check (early rejection before reading bytes)
    if content_length > MAX_FILE_SIZE_BYTES:
        logger.warning(
            f"SECURITY: Payload header claims {content_length} bytes — "
            f"exceeds {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB limit."
        )
        raise HTTPException(status_code=413, detail="Payload Too Large. Maximum size is 10MB.")

    # ── Step 2: MIME type validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: '{file.content_type}'. Image required."
        )

    try:
        # ── Step 3: Read bytes + actual size validation (detects spoofed headers)
        image_bytes: bytes = await file.read()
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            logger.warning(
                f"SECURITY: Spoofed Content-Length header detected. "
                f"Actual payload: {len(image_bytes)} bytes."
            )
            raise HTTPException(status_code=413, detail="Payload Too Large. Maximum size is 10MB.")

        # ── Step 4: In-memory image decoding
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = image_transforms(img).unsqueeze(0).to(DEVICE)

        # ── Step 5: Thread-offloaded inference
        # asyncio.to_thread offloads CPU/GPU-bound inference to a thread pool,
        # keeping the event loop free to accept other requests during computation.
        confidence_score, predicted_idx = await asyncio.to_thread(
            _run_inference, model, img_tensor
        )

        predicted_class = CLASS_NAMES[predicted_idx]

        logger.info(
            f"Evaluated '{file.filename or 'unknown'}': "
            f"{predicted_class.upper()} ({confidence_score:.4f})"
        )

        return PredictionResponse(
            filename  =file.filename or "unknown",
            prediction=predicted_class,
            confidence=round(confidence_score, 4),
        )

    except HTTPException:
        # CRITICAL: re-raise intentional HTTP errors before the generic handler.
        # Without this, the 413 raised above for spoofed headers would be
        # caught by `except Exception` below and incorrectly returned as a
        # misleading 500 — HTTPException is itself a subclass of Exception.
        raise

    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Corrupted or unreadable image file.")

    except Exception as e:
        logger.error(f"Unexpected inference error on '{file.filename}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during inference. Check server logs."
        )

    finally:
        try:
            del image_bytes, img, img_tensor
        except NameError:
            pass  # Variables may not exist if error occurred before assignment