# Adaptive Water Distribution Optimization using Reinforcement Learning

A professional, academic-grade Reinforcement Learning + MLOps project that dynamically allocates water from a central reservoir to multiple urban zones with changing demands.

## 🌍 Sustainable Development Goals (SDG) Impact
This project is built to directly address global sustainability challenges:
- **SDG 6 (Clean Water and Sanitation)**: By learning to penalize water wastage (overflow/leakage) and optimize shortage distribution, the RL agent promotes the efficient use of freshwater resources.
- **SDG 11 (Sustainable Cities and Communities)**: Smart, adaptive infrastructure makes urban centers more resilient to population growth and unpredictable climate-induced water scarcity.

## 🧠 Reinforcement Learning Architecture
- **Environment**: A custom simulator (`sim/water_env.py`) managing 1 reservoir and 3 stochastic demand zones.
- **Agent**: A Tabular Q-Learning agent (`agents/qlearning_agent.py`) utilizing $\epsilon$-greedy exploration.
- **State Space**: Discretized Reservoir (8 bins) + Demands (6 bins each).
- **Action Space**: 5 Discrete strategies (Equal, Priority A/B/C, Conservation).
- **Reward Function**: Multi-objective (Shortage, Wastage, Sustainability Bonus).

## 🗂️ Repository Structure
```
REL_AAT/
├── agents/                  # RL Agent logic
├── configs/                 # YAML Configuration
├── logs/                    # MLOps tracking (CSV/JSON)
├── policies/                # Saved model weights
├── results/                 # Evaluation plots and reports
├── sim/                     # Water Distribution Environment
├── train.py                 # Multi-experiment training script
├── evaluate.py              # Scientific stress-testing script
├── streamlit_app.py         # Interactive UI Dashboard
├── Dockerfile               # Containerization definition
├── docker-compose.yml       # Multi-container orchestration
└── requirements.txt
```

## 🐳 Docker Setup (Recommended)
The easiest way to run the project is using Docker. This ensures a consistent environment and isolated dependencies.

### 1. Build the Container
```bash
docker build -t water-rl .
```

### 2. Run the Dashboard
```bash
docker run -p 8501:8501 water-rl
```
*Access the dashboard at [http://localhost:8501](http://localhost:8501)*

### 3. Using Docker Compose
```bash
docker-compose up
```
*This maps the `results/` and `logs/` folders to your local machine, allowing you to see generated reports and metrics in real-time.*

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

### 1. Training the Agent
Training is fully config-driven. The script tracks MLOps metrics and serializes policies.
```bash
python train.py
```

### 2. Evaluation & Dashboard
Launch the Streamlit dashboard for a professional demonstration:
```bash
streamlit run streamlit_app.py
```

## 🔬 MLOps Deployment & Reproducibility
This project implements production-grade MLOps practices:
- **Containerization**: Docker isolated environments ensure that "it works on my machine" translates to "it works everywhere."
- **Dependency Isolation**: All versions are locked in `requirements.txt`.
- **Portable Deployment**: The containerized Streamlit app is ready for deployment on **Streamlit Cloud**, **Render**, **AWS ECS**, or **Google Cloud Run**.
- **Reproducible Environments**: Seeding is enforced across the entire stack (RNG, Environment, Agent) for verifiable scientific results.
- **Volume Mapping**: Docker Compose uses volume mapping to persist logs and model artifacts outside the container lifecycle.

## 📈 Sample Results
The RL agent demonstrates clear adaptive advantages in stressed environments. Refer to `results/evaluation_report.md` for scenario-wise performance ranking and analysis.

## 🤝 Acknowledgements
Built for academic evaluation in advanced Reinforcement Learning and MLOps.
