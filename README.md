# Adaptive Water Distribution Optimization using Reinforcement Learning

A professional, academic-grade Reinforcement Learning + MLOps project that dynamically allocates water from a central reservoir to multiple urban zones with changing demands.

## 🚀 MLOps Automation & Scalability (Excellent Category)
This project implements the highest level of MLOps maturity:
- **CI/CD Automation**: Fully automated GitHub Actions pipeline for testing, training validation, and Docker builds.
- **Production Inference API**: Scalable FastAPI service for real-time policy predictions.
- **Containerization**: Multi-service Docker orchestration for both Dashboard and API.
- **Scientific Reproducibility**: Strict seeding and config-driven experimentation.
- **Experiment Tracking**: Automated logging of all metrics and model versions.

## 🧠 Reinforcement Learning Architecture
- **Environment**: Custom simulator (`sim/water_env.py`) with stochastic spikes.
- **Agent**: Tabular Q-Learning (`agents/qlearning_agent.py`).
- **State Space**: Discretized Reservoir (8 bins) + Demands (6 bins each).
- **Action Space**: 5 Discrete strategies (Equal, Priority A/B/C, Conservation).

## 🗂️ Repository Structure
```
REL_AAT/
├── .github/workflows/       # CI/CD Pipeline (GitHub Actions)
├── agents/                  # RL Agent logic
├── configs/                 # YAML Configuration
├── logs/                    # MLOps tracking (CSV/JSON)
├── policies/                # Saved model weights
├── results/                 # Evaluation plots and reports
├── sim/                     # Water Distribution Environment
├── train.py                 # Multi-experiment training script
├── evaluate.py              # Scientific stress-testing script
├── streamlit_app.py         # Interactive UI Dashboard
├── api.py                   # FastAPI Inference Service
├── Dockerfile               # Containerization definition
├── docker-compose.yml       # Multi-service orchestration
└── start.sh                 # Multi-process startup script
```

## 🐳 Docker Setup
The easiest way to run the full project (Dashboard + API) is using Docker.

### 1. Build and Run
```bash
docker-compose up --build
```
- **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Inference API**: [http://localhost:8000](http://localhost:8000)

## 📡 Inference API (FastAPI)
The project includes a production-ready REST API for real-time water allocation predictions.

### Endpoints
- `GET /`: Status and metadata.
- `GET /health`: API health check.
- `POST /predict_action`: Get recommended allocation strategy.

### Example Request (cURL)
```bash
curl -X POST "http://localhost:8000/predict_action" \
     -H "Content-Type: application/json" \
     -d '{"reservoir": 45.5, "demand_a": 20.0, "demand_b": 15.0, "demand_c": 35.0}'
```

### Example Response
```json
{
  "recommended_action_id": 4,
  "recommended_action": "Conservation Mode",
  "state_discretized": [3, 2, 1, 3]
}
```

## ⚙️ Local Installation
1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd REL_AAT
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage
### Training
```bash
python train.py
```
### Dashboard
```bash
streamlit run streamlit_app.py
```
### API
```bash
uvicorn api:app --reload
```

## 🔬 CI/CD Pipeline
Every push to `main` triggers the **GitHub Actions** workflow:
1. **Linting**: Basic code validation.
2. **Smoke Test**: Runs a 10-episode training run to verify the pipeline.
3. **Evaluation**: Verifies that `evaluate.py` generates reports successfully.
4. **Integration**: Checks that Streamlit and FastAPI apps import without errors.
5. **Docker Build**: Verifies the Docker image builds correctly.

## 🤝 Acknowledgements
Built for academic evaluation in advanced Reinforcement Learning and MLOps.
