import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from sim.water_env import WaterDistributionEnv
from agents.qlearning_agent import QLearningAgent
from evaluate import rule_based_action

# --- DIRECTORY SETUP ---
# Ensure necessary directories exist for deployment environments
for folder in ["results", "logs", "policies"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- CONFIGURATION & SETTINGS ---
st.set_page_config(
    page_title="Water Distribution RL Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_config():
    config_path = os.path.join("configs", "qlearning.yaml")
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found at {config_path}")
        return None
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_agent(policy_name):
    agent = QLearningAgent(n_actions=5)
    path = os.path.join("policies", f"{policy_name}.pkl")
    if os.path.exists(path):
        try:
            agent.load_policy(path)
            agent.set_evaluation_mode()
            return agent
        except Exception as e:
            st.error(f"Error loading policy: {e}")
            return None
    return None

def run_sim_step_by_step(env, agent, steps, mode="RL"):
    history = []
    state = env.reset()
    
    # Progress bar for the simulation
    progress_bar = st.progress(0)
    
    for i in range(steps):
        if mode == "RL":
            action = agent.choose_action(state)
        elif mode == "Smart":
            action = rule_based_action(state, env)
        else: # Equal
            action = 0
            
        next_state, reward, done, info = env.step(action)
        
        # Capture step data
        step_data = {
            "step": i + 1,
            "reservoir": info['tank_utilization'] * 100,
            "reward": reward,
            "shortage": info['shortage'],
            "wastage": info['wastage'],
            "action": action,
            "demand_a": info['demands'][0],
            "demand_b": info['demands'][1],
            "demand_c": info['demands'][2],
            "supply_a": info['supply'][0],
            "supply_b": info['supply'][1],
            "supply_c": info['supply'][2],
        }
        history.append(step_data)
        state = next_state
        
        # Update UI progress
        progress_bar.progress((i + 1) / steps)
        
        if done:
            break
            
    return pd.DataFrame(history)

# --- MAIN APP ---
def main():
    config = load_config()
    if config is None:
        st.stop()
    
    # --- SIDEBAR ---
    st.sidebar.title("🎮 Controls")
    
    st.sidebar.header("Policy Selection")
    policy_option = st.sidebar.selectbox("Choose Policy", ["policy_v1", "policy_v2"])
    agent = load_agent(policy_option)
    
    if agent is None:
        st.sidebar.warning(f"⚠️ Policy '{policy_option}.pkl' not found. Please train the agent first using 'python train.py' or check the 'policies/' directory.")
    else:
        st.sidebar.success(f"✅ Policy {policy_option} loaded.")

    st.sidebar.divider()
    
    st.sidebar.header("Scenario Settings")
    scenario_option = st.sidebar.selectbox(
        "Simulation Scenario", 
        ["normal", "peak", "uneven", "shortage", "extreme"]
    )
    
    sim_steps = st.sidebar.slider("Simulation Steps", 50, 200, 100)
    
    run_button = st.sidebar.button("🚀 Run Simulation")

    # --- SECTION 1: OVERVIEW ---
    st.title("💧 Adaptive Water Distribution Optimization")
    st.markdown("### Reinforcement Learning for Sustainable Urban Water Management")
    
    col_ov1, col_ov2 = st.columns([2, 1])
    
    with col_ov1:
        st.info("""
        This project utilizes **Reinforcement Learning (Q-Learning)** to optimize water allocation 
        across multiple urban zones. The agent learns to balance immediate demand satisfaction 
        with long-term reservoir stability, especially under high-stress scenarios like peak demand 
        and extreme shortages.
        """)
        
    with col_ov2:
        st.markdown("#### 🎯 SDG Impact")
        st.write("✅ **SDG 6:** Clean Water & Sanitation")
        st.write("✅ **SDG 11:** Sustainable Cities & Communities")
        st.write(f"**Current Target Policy:** `{policy_option}`")

    st.divider()

    if run_button and agent:
        # --- EXECUTION ---
        # Initialize Environment
        env = WaterDistributionEnv(
            reservoir_capacity=config['environment']['reservoir_capacity'],
            refill_amount=config['environment']['refill_amount'],
            max_steps=sim_steps,
            leakage_rate=config['environment'].get('leakage_rate', 0.15)
        )
        
        # Setup scenario
        refills = config['environment'].get('scenario_refills', {})
        sc_refill = refills.get(scenario_option, 30)
        sc_range = config['environment']['demand_ranges'][scenario_option]
        sc_uneven = config['environment']['demand_ranges'].get('uneven') if scenario_option == 'uneven' else None
        
        env.set_scenario(scenario_option, sc_range, sc_refill, sc_uneven)
        
        # Run RL Simulation
        with st.spinner(f"Running {scenario_option} simulation..."):
            df_rl = run_sim_step_by_step(env, agent, sim_steps, mode="RL")
            
            # Run Baselines for comparison (same seeds)
            env.rng = np.random.default_rng(config['agent']['seed'])
            df_equal = run_sim_step_by_step(env, None, sim_steps, mode="Equal")
            
            env.rng = np.random.default_rng(config['agent']['seed'])
            df_smart = run_sim_step_by_step(env, None, sim_steps, mode="Smart")

        # --- SECTION 4: LIVE METRICS ---
        st.header("📊 Simulation Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        
        m1.metric("Total Reward", f"{df_rl['reward'].sum():.1f}")
        m2.metric("Total Shortage", f"{df_rl['shortage'].sum():.1f}")
        m3.metric("Total Wastage", f"{df_rl['wastage'].sum():.1f}")
        m4.metric("Avg Utilization", f"{df_rl['reservoir'].mean():.1f}%")

        # --- SECTION 5: VISUALIZATIONS ---
        tab1, tab2, tab3 = st.tabs(["📉 Time Series", "📊 Action Analysis", "⚖️ Comparison"])
        
        with tab1:
            st.subheader("System State Over Time")
            # Reservoir Level
            fig_res = px.line(df_rl, x="step", y="reservoir", title="Reservoir Utilization (%)", 
                             color_discrete_sequence=['#007bff'])
            fig_res.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Low Level Warning")
            st.plotly_chart(fig_res, use_container_width=True)
            
            # Shortage vs Wastage
            fig_met = go.Figure()
            fig_met.add_trace(go.Scatter(x=df_rl['step'], y=df_rl['shortage'], name="Shortage", fill='tozeroy'))
            fig_met.add_trace(go.Scatter(x=df_rl['step'], y=df_rl['wastage'], name="Wastage", fill='tonexty'))
            fig_met.update_layout(title="Shortage vs Wastage Trends", xaxis_title="Step", yaxis_title="Units")
            st.plotly_chart(fig_met, use_container_width=True)

        with tab2:
            st.subheader("Agent Decision Strategy")
            action_map = {0: "Equal", 1: "Priority A", 2: "Priority B", 3: "Priority C", 4: "Conservation"}
            df_rl['action_name'] = df_rl['action'].map(action_map)
            
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                fig_act = px.pie(df_rl, names="action_name", title="Action Distribution", hole=0.4)
                st.plotly_chart(fig_act, use_container_width=True)
                
            with col_a2:
                st.markdown("#### Timestep Action Log")
                st.dataframe(df_rl[['step', 'action_name', 'reward', 'reservoir']].tail(10), use_container_width=True)

        with tab3:
            st.subheader("RL vs Baselines")
            comparison_data = {
                "Metric": ["Total Reward", "Total Shortage", "Total Wastage", "Avg Utilization (%)"],
                "RL Policy": [df_rl['reward'].sum(), df_rl['shortage'].sum(), df_rl['wastage'].sum(), df_rl['reservoir'].mean()],
                "Equal Baseline": [df_equal['reward'].sum(), df_equal['shortage'].sum(), df_equal['wastage'].sum(), df_equal['reservoir'].mean()],
                "Smart Baseline": [df_smart['reward'].sum(), df_smart['shortage'].sum(), df_smart['wastage'].sum(), df_smart['reservoir'].mean()],
            }
            df_comp = pd.DataFrame(comparison_data)
            st.table(df_comp.set_index("Metric"))
            
            fig_comp = go.Figure(data=[
                go.Bar(name='RL', x=df_comp['Metric'][:3], y=df_comp['RL Policy'][:3]),
                go.Bar(name='Equal', x=df_comp['Metric'][:3], y=df_comp['Equal Baseline'][:3]),
                go.Bar(name='Smart', x=df_comp['Metric'][:3], y=df_comp['Smart Baseline'][:3])
            ])
            fig_comp.update_layout(barmode='group', title="Metric Comparison (Lower is Better for Shortage/Wastage)")
            st.plotly_chart(fig_comp, use_container_width=True)

    else:
        if not agent:
             st.warning("👈 Please select a valid policy from the sidebar to begin.")
        else:
             st.info("👈 Click 'Run Simulation' in the sidebar to start.")

    # --- SECTION 7: ARCHITECTURE ---
    with st.expander("🏗️ Project Architecture & MLOps"):
        col_ar1, col_ar2 = st.columns(2)
        with col_ar1:
            st.markdown("""
            **RL Framework:**
            - **Algorithm:** Tabular Q-Learning
            - **State Space:** Discretized Reservoir (8 bins) + Demands (6 bins each)
            - **Action Space:** 5 Discrete strategies
            - **Reward:** Multi-objective (Shortage, Wastage, Sustainability Bonus)
            """)
        with col_ar2:
            st.markdown("""
            **MLOps Features:**
            - **Containerization:** Docker & Docker Compose support
            - **Config Management:** YAML-driven environments
            - **Reproducibility:** Global seeding and RNG enforcement
            - **Versioning:** Automated policy serialization (.pkl)
            """)

    # --- SECTION 8: DOWNLOADS ---
    st.header("📂 Downloadable Reports")
    d1, d2, d3 = st.columns(3)
    
    report_path = os.path.join("results", "evaluation_report.md")
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            d1.download_button("📜 Evaluation Report (MD)", f, "evaluation_report.md")
    else:
        d1.info("Evaluation report not yet generated.")
            
    exp_logs_path = os.path.join("logs", "experiments.json")
    if os.path.exists(exp_logs_path):
        with open(exp_logs_path, "rb") as f:
            d2.download_button("📊 Experiment Logs (JSON)", f, "experiments.json")
    else:
        d2.info("Experiment logs not yet generated.")
            
    train_logs_path = os.path.join("logs", "training_logs_policy_v1.csv")
    if os.path.exists(train_logs_path):
        with open(train_logs_path, "rb") as f:
            d3.download_button("📈 Training History (CSV)", f, "training_history.csv")
    else:
        d3.info("Training history not yet generated.")

if __name__ == "__main__":
    main()
