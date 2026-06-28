# Bees vs Ants Classification API

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

The foundational computer vision project — a production-deployed binary classifier distinguishing bees from ants, built on ResNet18 transfer learning. This project establishes the core supervised classification and deployment patterns applied at scale across the rest of this portfolio.

---

## System Architecture

| Component | Implementation | Details |
|-----------|---------------|---------|
| **Classification Engine** | ResNet18 | Transfer learning, ImageNet pretrained |
| **Web Framework** | FastAPI + Uvicorn | ASGI, async request handling |
| **Inference** | asyncio.to_thread | Non-blocking CPU inference |
| **Image Decoding** | io.BytesIO + PIL | Zero disk I/O — raw bytes decoded in RAM |
| **Authentication** | API Key header | Unauthorized requests rejected with 401 |
| **Container** | python:3.10-slim | Non-root user, layer-cached builds, HEALTHCHECK |

**Why this project matters beyond its simplicity:**
This is where the core engineering decisions — transfer learning, normalization consistency, device-agnostic code, optimized DataLoaders — were first validated before being applied at scale in the anomaly detection, segmentation, and real-time detection projects in this portfolio.

---

## Model Performance

Fine-tuned over 20 epochs with deep-copy state checkpointing on best validation accuracy:

| Metric | Value |
|--------|-------|
| Peak Validation Accuracy | 94.12% |
| Validation Loss | 0.2216 |

---

## Tech Stack

- **Deep Learning:** PyTorch 2.1, Torchvision
- **Web Server:** FastAPI 0.104, Uvicorn (with uvloop + httptools)
- **Image Processing:** Pillow
- **DevOps:** Docker, python:3.10-slim base image

---

## API Endpoints

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| `GET` | `/` | No | API metadata and endpoint discovery |
| `GET` | `/health` | No | Liveness check — is the process running? |
| `GET` | `/ready` | No | Readiness check — is the model loaded? |
| `POST` | `/api/v1/predict` | Yes | Bees vs ants classification |

---

## Project Structure

```
bees-ants-classifier/
├── app/
│   └── main.py                          # FastAPI inference endpoint
├── models/
│   └── best_model.pth                   # Model artifact (download separately)
├── research/
│   └── Bees_vs_Ants_Image_Classifier.ipynb  # Training pipeline
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Quick Start

### 1. Download the Model Artifact

Due to GitHub file size limits, trained weights are stored externally.

1. Download `best_model.pth` from: [Model Registry (Google Drive)](https://drive.google.com/file/d/1nXwCB6OXB4Kyk1qZoqpJc5QLdwT-gq3I/view?usp=drive_link)
2. Place it inside the `models/` directory:

```
models/
└── best_model.pth
```

### 2. Build the Container

```bash
docker build -t bees-ants-api:v1 .
```

### 3. Run the Container

The API requires an `API_SECRET_KEY` environment variable — the server will refuse to boot without it.

```bash
docker run -p 8000:8000 \
  -e API_SECRET_KEY=your_secret_key \
  bees-ants-api:v1
```

Verify the API is ready:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### 4. Run Inference

**Option A — Swagger UI (recommended):**

1. Navigate to **http://localhost:8000/docs**
2. Click the **Authorize** button (top right)
3. Enter the same key you used in `API_SECRET_KEY` into the `Authorization-API-Key` field
4. Open `POST /api/v1/predict`, click **Try it out**, upload an ant or bee image, click **Execute**

**Option B — Terminal:**

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict' \
  -H "Authorization-API-Key: your_secret_key" \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_test_image.jpg'
```

---

## Example Response

```json
{
  "filename": "ant_sample_042.jpg",
  "prediction": "ants",
  "confidence": 0.9734
}
```

---

## Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `API_SECRET_KEY` | None | Yes | Authentication key — server refuses to boot without it |
| `MODEL_PATH` | `models/best_model.pth` | No | Path to trained model artifact |

---

## Research & Development

To reproduce training metrics or experiment with the architecture, install the research dependencies:

```bash
pip install -r requirements-dev.txt
jupyter lab
```

Open `research/Bees_vs_Ants_Image_Classifier.ipynb`. The notebook handles:
1. Automated Hymenoptera dataset download and extraction
2. AMP-accelerated training (autocast + GradScaler) with AdamW
3. Per-class evaluation — confusion matrix and classification report
4. Production smoke testing before deployment

---

## Docker Notes

**No OpenCV system dependencies required** — this service uses Pillow for image decoding rather than OpenCV, so the `libgl1`/`libglib2.0-0` libraries needed in this portfolio's other projects don't apply here.

**Non-root execution:** Container runs as `api_user` — principle of least privilege.

**Health monitoring:** Docker's native `HEALTHCHECK` polls `/health` every 30 seconds with a 60-second startup grace period for model loading.