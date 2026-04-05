# Hymenoptera Vision API: Binary Classification Microservice

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker)
![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)

An enterprise-grade, containerized Computer Vision microservice designed to perform high-speed binary classification (Bees vs. Ants) on the edge.

This repository demonstrates the transition from a Data Science research environment (Jupyter) into a production-ready **MLOps pipeline**. The API is built with FastAPI, utilizing a deeply optimized PyTorch ResNet18 architecture for inference, and is fully encapsulated within a CPU-optimized Docker container for rapid cloud deployment.

## System Architecture

* **The Engine:** PyTorch (`torch`, `torchvision`) utilizing Transfer Learning on a ResNet18 backbone.
* **The Server:** Asynchronous FastAPI utilizing strict Pydantic schema validation.
* **The Infrastructure:** Dockerized with Layer Caching and a CPU-only PyTorch configuration to minimize image bloat and maximize deployment velocity.
* **VRAM Management:** The model weights are loaded globally into a state dictionary via FastAPI's `@asynccontextmanager` during server lifespan initialization, ensuring zero latency during HTTP inference requests.

## Model Performance

The neural network was fine-tuned over 20 epochs utilizing `Deepcopy` state-saving to prevent overfitting, achieving the following metrics on the validation set:
* **Peak Validation Accuracy:** 94.12%
* **Loss:** 0.2216

## Quick Start (Production Environment)

The easiest way to run this API is via Docker. The container is completely isolated and does not require a local Python environment.

### Download the Model Artifact
Due to GitHub's file size constraints, the trained neural network weights are hosted securely via Google Drive.
1. Download `best_model.pth` from this direct link: **[https://drive.google.com/file/d/1nXwCB6OXB4Kyk1qZoqpJc5QLdwT-gq3I/view?usp=drive_link]**
2. Place the downloaded `.pth` file directly inside the empty `models/` directory of this repository.

### Build the Microservice
```bash
docker build -t hymenoptera-api:latest .

```

### Ignite the Container

```bash
docker run -p 8000:8000 hymenoptera-api:latest
```

### Execute Inference
Navigate to http://127.0.0.1:8000/docs to access the interactive Swagger UI.

## Live API Testing (DevSecOps Secured)

This inference endpoint is strictly protected by memory-exhaustion payload limiters (10MB max) and an API Key cryptographic lock to prevent unauthorized VRAM consumption.

To test the model's predictions, please use the following temporary evaluation key:
**`antvsbee_dev_key`**

### Option A: The Visual Web Interface (Recommended)
FastAPI provides a built-in interactive testing environment.

1. Run the Docker container and navigate to `http://localhost:8000/docs`.
2. Click the green **Authorize** button in the top right corner.
3. Paste the evaluation key into the `X-API-Key` field and click Authorize.
4. Open the `POST /predict/` dropdown, click **Try it out**, upload any image of an ant or bee, and click **Execute** to see the real-time classification.

### Option B: The Terminal
If you prefer to bypass the UI and test the raw JSON response and header validation directly:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'X-API-Key: antvsbee_dev_key' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_test_image.jpg'
```

# Research & Development (Local Environment)
To reproduce the training metrics or experiment with the model architecture, you must install the heavier research dependencies.

1. Clone the repository and initialize a virtual environment (Conda recommended).

2. Install the isolated development requirements:

```bash
pip install -r requirements-dev.txt
```

3. Launch the Jupyter environment to access research/Bees_vs_Ants_Image_Classifier.ipynb.

Note: The Jupyter notebook contains an automated pipeline that will automatically download and extract the Hymenoptera dataset if it is not found locally.