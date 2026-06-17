# NeuralServe — Transformer Inference & Optimization Platform

Production-grade ML inference platform built on **DistilBERT** (transformer architecture) with **ONNX Runtime graph optimization**, **dynamic memory-aware batching**, **FastAPI REST API**, and **PostgreSQL inference logging**.

---

## Architecture

```
Training (Google Colab — Free T4 GPU / CUDA)
         │
         ▼
Fine-tune DistilBERT on SST-2 sentiment dataset (PyTorch + CUDA)
         │
         ▼
Export to ONNX (compiler-level graph optimization + memory scheduling)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│                 FastAPI Inference Server                │
│                                                        │
│  POST /api/v1/predict        — single inference        │
│  POST /api/v1/predict/batch  — dynamic batching        │
│  GET  /api/v1/model/info     — model metadata          │
│  GET  /api/v1/metrics        — latency & throughput    │
│  GET  /api/v1/logs           — inference history       │
│  GET  /health/ready          — liveness + DB check     │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Rust-backed HuggingFace tokenizers              │  │
│  │ ONNX Runtime (graph opt) / PyTorch fallback     │  │
│  │ Dynamic padding — memory-efficient batching     │  │
│  │ Softmax via linear algebra (numpy C++ backend)  │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
         │
         ▼
PostgreSQL — inference_logs (latency, confidence, label, runtime)
         │
         ▼
Docker → Railway / Render (free deployment)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Model** | DistilBERT (Transformer — 6-layer, 768-dim, 12-heads, 66M params) |
| **Training** | PyTorch + CUDA (Google Colab free T4 GPU) |
| **Tokenization** | HuggingFace Tokenizers (Rust-backed fast tokenizer) |
| **Inference** | ONNX Runtime (graph optimization) / PyTorch fallback |
| **API** | FastAPI + asyncpg + SQLAlchemy ORM |
| **Database** | PostgreSQL 16 |
| **Container** | Docker multi-stage build (non-root appuser UID 1001) |
| **CI/CD** | GitHub Actions (lint → test → docker build) |

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Mukunth1004/NeuralServe.git
cd NeuralServe
cp .env.example .env
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

API will be available at `http://localhost:8000`
Auto-interactive docs: `http://localhost:8000/docs`

> The DistilBERT model downloads automatically on first startup (~260 MB). Takes ~60s on first run.

### 3. Run locally (without Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start PostgreSQL then:
uvicorn app.main:app --reload --port 8000
```

---

## API Reference

### Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

```json
{
  "result": {
    "label": "POSITIVE",
    "confidence": 0.9987,
    "scores": {"POSITIVE": 0.9987, "NEGATIVE": 0.0013},
    "text_length": 36,
    "inference_ms": 14.2
  },
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "runtime": "pytorch",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Batch Prediction (Dynamic Batching)

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible experience.", "Okay, nothing special."]}'
```

### Metrics Dashboard

```bash
curl http://localhost:8000/api/v1/metrics
```

```json
{
  "total_inferences": 1247,
  "avg_inference_ms": 13.5,
  "avg_confidence": 0.9234,
  "label_distribution": {"POSITIVE": 832, "NEGATIVE": 415},
  "last_24h_count": 98
}
```

---

## ONNX Graph Optimization (Optional)

For faster inference with compiler-level graph optimizations:

```bash
pip install optimum[onnxruntime]
python scripts/export_onnx.py --output ./models/onnx
```

Then update `.env`:
```
USE_ONNX=true
MODEL_PATH=./models/onnx
```

Optimizations applied by ONNX Runtime:
- **Attention fusion** — fuses multi-head attention layers into single kernels
- **Constant folding** — pre-computes static graph nodes at compile time
- **Memory layout optimization** — reorders tensor memory for cache efficiency
- **Operator fusion** — combines adjacent elementwise ops into single CUDA kernels

---

## Fine-tune on Google Colab (Free GPU)

1. Go to [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Upload `scripts/train_colab.py` and run all cells
4. Downloads model → trains 3 epochs on SST-2 → exports ONNX
5. Download `./models/onnx` and place in project root

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
NeuralServe/
├── app/
│   ├── main.py              # FastAPI app, lifespan, middleware
│   ├── config.py            # Pydantic settings
│   ├── logger.py            # Structured JSON logging
│   ├── models/schemas.py    # Pydantic request/response models
│   ├── routers/
│   │   ├── health.py        # /health, /health/live, /health/ready
│   │   ├── inference.py     # /api/v1/predict, /predict/batch, /model/info
│   │   └── metrics.py       # /api/v1/metrics, /logs
│   ├── services/
│   │   ├── model_service.py # ONNX + PyTorch inference, dynamic batching
│   │   └── db_service.py    # Inference log read/write, metrics aggregation
│   └── db/database.py       # SQLAlchemy async engine, InferenceRecord model
├── scripts/
│   ├── export_onnx.py       # ONNX export with graph optimization
│   └── train_colab.py       # Colab fine-tuning script (CUDA)
├── tests/
│   ├── test_health.py
│   └── test_inference.py
├── Dockerfile               # Multi-stage build, non-root user
├── docker-compose.yml       # API + PostgreSQL
└── .github/workflows/ci.yml # Lint → Test → Docker build
```

---

## Resume Bullets (What This Project Demonstrates)

- Architected a **transformer-based inference platform** fine-tuning **DistilBERT** with **PyTorch** and **CUDA** (Google Colab T4 GPU), exported to **ONNX** for compiler-level **graph optimization** (attention fusion, constant folding, memory scheduling), achieving sub-100ms inference latency.

- Implemented **memory-aware dynamic batching** using **Rust-backed HuggingFace tokenizers**, applying **linear algebra** (numerically stable softmax over probability distributions) and **ONNX Runtime graph optimizations** to serve predictions via **FastAPI REST APIs** with async **PostgreSQL** inference logging.

- Built end-to-end **AI/ML inference pipeline** with **transformer architectures** (DistilBERT 6-layer, 768-dim, 12-head attention), **Docker** multi-stage containerization, and **GitHub Actions** CI/CD with ruff linting, pytest, and automated health verification.

---

## License

MIT
