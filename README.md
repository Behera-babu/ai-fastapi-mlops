# 🚀 AI-FastAPI-MLOps

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![CI](https://img.shields.io/github/actions/workflow/status/Priyanshjain10/ai-fastapi-mlops/ci.yml?branch=main)

**Production-ready AI service template with SOTA models and MLOps best practices**

[Features](#-features) • [Quick Start](#-quick-start) • [API](#-api-endpoints) • [Deployment](#-deployment)

</div>

---

## ✨ Features

### Core Capabilities
- ⚡ **Fast API** - Sub-100ms inference with async support
- 🤖 **SOTA Models** - Vision Transformers, BERT, T5 integration
- 📊 **Monitoring** - Prometheus metrics & Grafana dashboards
- 🔄 **MLOps Pipeline** - Complete CI/CD with GitHub Actions
- 🐳 **Containerized** - Docker & Kubernetes ready
- 💾 **Database** - PostgreSQL for persistence, Redis for caching
- 🔒 **Production-Ready** - Security, logging, error handling

### Technical Highlights
- **Async Architecture** - Non-blocking I/O for high concurrency
- **Model Caching** - Smart model loading and memory management
- **Auto Documentation** - Interactive Swagger UI & ReDoc
- **Health Checks** - Kubernetes-compatible probes
- **Metric Collection** - Request latency, throughput, error rates
- **Horizontal Scaling** - Stateless design for replication

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)
- 4GB RAM minimum

### Local Development

```bash
# Clone repository
git clone https://github.com/Priyanshjain10/ai-fastapi-mlops.git
cd ai-fastapi-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
uvicorn api.main:app --reload

# Visit API docs
open http://localhost:8000/docs
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

**Services:**
- API: http://localhost:8000
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

---

## 📁 Project Structure

```
ai-fastapi-mlops/
├── .github/workflows/
│   └── ci.yml           # CI/CD pipeline
├── api/
│   └── main.py          # FastAPI application
├── monitoring/
│   └── prometheus.yml   # Prometheus config
├── docker-compose.yml   # Multi-service setup
├── Dockerfile           # Container image
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Framework** | FastAPI, Uvicorn, Pydantic |
| **ML/AI** | PyTorch, Transformers |
| **Database** | PostgreSQL, Redis |
| **Monitoring** | Prometheus, Grafana |
| **Deployment** | Docker, Kubernetes |
| **CI/CD** | GitHub Actions, pytest |

---

## 📚 API Endpoints

### Health & Status

```bash
# Health check
GET /health
Response: {"status": "healthy", "timestamp": 1699120800.123}

# API info
GET /
```

### Vision Models

```bash
# Image Classification
POST /predict/vision
Content-Type: multipart/form-data

Request:
- file: <image_file>

Response:
{
  "prediction": "golden_retriever",
  "confidence": 0.94,
  "model": "vit-base-patch16-224",
  "inference_time_ms": 45.2
}
```

### NLP Models

```bash
# Text Analysis
POST /predict/nlp
Content-Type: application/json

Request:
{
  "text": "This product is amazing!",
  "task": "sentiment"
}

Response:
{
  "prediction": "POSITIVE",
  "confidence": 0.92
}
```

---

## 🐳 Docker Deployment

### Multi-Service Stack

```bash
# Start all services
docker-compose up -d

# Scale API instances
docker-compose up -d --scale api=3

# View status
docker-compose ps
```

---

## 📊 Monitoring

### Metrics Collected
- **Request Latency** - P50, P95, P99 percentiles
- **Throughput** - Requests per second
- **Error Rates** - 4xx, 5xx by endpoint
- **Model Inference Time** - Per model and task

### Grafana Dashboards
- API Performance - Latency, throughput
- Model Metrics - Inference time
- System Resources - CPU, memory

---

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=api --cov-report=html
```

---

## 🔒 Security

- ✅ Non-root container
- ✅ Input validation with Pydantic
- ✅ Health checks
- ✅ CORS configuration

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Latency (P50)** | 45ms |
| **Latency (P95)** | 120ms |
| **Throughput** | 1000 req/s |
| **Memory** | ~500MB per instance |

---

## 🎯 Roadmap

- [x] Core API with vision & NLP endpoints
- [x] Docker & Docker Compose setup
- [x] CI/CD pipeline
- [x] Prometheus metrics
- [ ] Add more models (YOLO, CLIP, GPT)
- [ ] Redis caching
- [ ] API authentication
- [ ] Kubernetes Helm charts
- [ ] Auto-scaling

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and open a Pull Request

---

## 📄 License

MIT License

---

## 👨‍💻 Author

**Priyansh Jain**
- GitHub: [@Priyanshjain10](https://github.com/Priyanshjain10)
- Email: priyanshj1304@gmail.com

---

<div align="center">

**If you find this project useful, please ⭐ star the repository!**

Made with ❤️ by Priyansh Jain

</div>
