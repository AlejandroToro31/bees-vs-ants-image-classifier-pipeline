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

### 1. Build the Microservice

```bash
docker build -t hymenoptera-api:latest . 
```

### 2. Ignite the Container

```bash
docker run -p 8000:8000 hymenoptera-api:latest
```

### 3. Execute Inference
Navigate to http://127.0.0.1:8000/docs to access the interactive Swagger UI.
You can upload any image to the /predict/ endpoint to receive a real-time JSON probability payload.

# Research & Development (Local Environment)
To reproduce the training metrics or experiment with the model architecture, you must install the heavier research dependencies.

1. Clone the repository and initialize a virtual environment (Conda recommended).

2. Install the isolated development requirements:

```bash
pip install -r requirements-dev.txt
```

3. Launch the Jupyter environment to access research/bees_vs_ants_training.ipynb.

Note: The Jupyter notebook contains an automated pipeline that will automatically download and extract the Hymenoptera dataset if it is not found locally.