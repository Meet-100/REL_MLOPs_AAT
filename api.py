import os
import yaml
import logging
import pickle
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agents.qlearning_agent import QLearningAgent

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
def load_config():
    config_path = os.path.join("configs", "qlearning.yaml")
    if not os.path.exists(config_path):
        logger.error("Configuration file not found.")
        return None
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# --- MODEL HANDLING ---
# Load the trained policy (policy_v2.pkl is specified in requirements)
policy_path = os.path.join("policies", "policy_v2.pkl")
agent = QLearningAgent(n_actions=5)

if os.path.exists(policy_path):
    try:
        agent.load_policy(policy_path)
        agent.set_evaluation_mode()
        logger.info(f"Policy loaded successfully from {policy_path}")
    except Exception as e:
        logger.error(f"Error loading policy: {e}")
else:
    logger.warning(f"Policy file {policy_path} not found. API will use untrained agent.")

# --- REQUEST/RESPONSE MODELS ---
class InferenceRequest(BaseModel):
    reservoir: float = Field(..., ge=0, le=100, description="Current reservoir level (0-100)")
    demand_a: float = Field(..., ge=0, description="Current demand for Zone A")
    demand_b: float = Field(..., ge=0, description="Current demand for Zone B")
    demand_c: float = Field(..., ge=0, description="Current demand for Zone C")

class InferenceResponse(BaseModel):
    recommended_action_id: int
    recommended_action: str
    state_discretized: tuple

# --- API APP ---
app = FastAPI(
    title="Water Distribution RL Inference API",
    description="Production-style REST API for adaptive water allocation predictions.",
    version="1.0.0"
)

def discretize_state(reservoir, demands):
    """
    Mirror the environment's discretization logic:
    - 8 bins for reservoir
    - 6 bins for demands (scaled to max_d_ref=60)
    """
    res_bins = 8
    dem_bins = 6
    res_cap = 100.0 # From config default
    max_d_ref = 60.0
    
    res_bin = min(int(reservoir / (res_cap / res_bins)), res_bins - 1)
    dem_bins_list = [min(int(d / (max_d_ref / dem_bins)), dem_bins - 1) for d in demands]
    return (res_bin, *dem_bins_list)

@app.get("/")
async def root():
    """Return project status and metadata."""
    return {
        "project": "Adaptive Water Distribution Optimization",
        "status": "Online",
        "model_loaded": os.path.exists(policy_path),
        "algorithm": "Tabular Q-Learning",
        "sdg_alignment": ["SDG 6", "SDG 11"]
    }

@app.get("/health")
async def health():
    """Health check endpoint for monitoring systems."""
    return {"status": "healthy", "timestamp": logger.manager.loggerDict.get('__name__') if False else "N/A"} # Simplified

@app.post("/predict_action", response_model=InferenceResponse)
async def predict_action(request: InferenceRequest):
    """
    Receive environment state and return the optimal allocation strategy.
    """
    logger.info(f"Received prediction request: {request}")
    
    try:
        # 1. Discretize the input state
        demands = [request.demand_a, request.demand_b, request.demand_c]
        state = discretize_state(request.reservoir, demands)
        
        # 2. Get action from agent
        action_id = agent.choose_action(state)
        
        action_map = {
            0: "Equal Distribution",
            1: "Prioritize Zone A",
            2: "Prioritize Zone B",
            3: "Prioritize Zone C",
            4: "Conservation Mode"
        }
        
        recommended_action = action_map.get(action_id, "Unknown Action")
        
        logger.info(f"Recommended Action: {recommended_action} (ID: {action_id}) for State: {state}")
        
        return InferenceResponse(
            recommended_action_id=action_id,
            recommended_action=recommended_action,
            state_discretized=state
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during inference.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
