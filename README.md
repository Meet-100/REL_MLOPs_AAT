# Adaptive Water Distribution Optimization using Reinforcement Learning

A professional, academic-grade Reinforcement Learning + MLOps project that dynamically allocates water from a central reservoir to multiple urban zones with changing demands.

## 🚀 MLOps Automation & Lifecycle (Excellent Category)
This project implements the full MLOps lifecycle:
- **Experiment Tracking (MLflow)**: Centralized logging of hyperparameters, training metrics, and evaluation results.
- **Model & Artifact Management**: Automated versioning of RL policies, plots, and markdown reports.
- **CI/CD Automation**: GitHub Actions pipeline for testing and Docker builds.
- **Production Inference API**: FastAPI service for real-time policy predictions.
- **Containerization**: Multi-service Docker orchestration for Dashboard, API, and MLflow.

## 📊 Experiment Tracking with MLflow
All training and evaluation runs are tracked using MLflow. This allows for rigorous comparison of hyperparameters and policy performance.

### Features
- **Parameter Logging**: Track `alpha`, `gamma`, `epsilon_decay`, `leakage_rate`, etc.
- **Metric Logging**: Monitor `avg_reward` and `shortage` over time.
- **Artifact Management**: Policy `.pkl` files, training plots, and evaluation reports are stored for every run.
- **Model Registry**: Light tagging of "best_policy" based on evaluation benchmarks.

### Accessing the MLflow UI
1. **Start the services**: `docker-compose up` or `mlflow ui`
2. **Access UI**: [http://localhost:5000](http://localhost:5000)

## 🐳 Docker Setup
The project runs three integrated services:
- **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Inference API**: [http://localhost:8000](http://localhost:8000)
- **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)

```bash
docker-compose up --build
```

## 📡 Inference API (FastAPI)
REST API for real-time water allocation predictions.
```bash
curl -X POST "http://localhost:8000/predict_action" \
     -H "Content-Type: application/json" \
     -d '{"reservoir": 45.0, "demand_a": 20.0, "demand_b": 15.0, "demand_c": 30.0}'
```

## ⚙️ Local Installation
```bash
pip install -r requirements.txt
python train.py
python evaluate.py
mlflow ui
```

## 🤝 Acknowledgements
Built for academic evaluation in advanced Reinforcement Learning and MLOps.
