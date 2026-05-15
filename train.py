import yaml
import os
import json
import pandas as pd
import numpy as np
import random
import argparse
import time
import matplotlib.pyplot as plt
from sim.water_env import WaterDistributionEnv
from agents.qlearning_agent import QLearningAgent
from datetime import datetime

def check_convergence(history, window, threshold=0.1):
    """
    Check if the moving average reward has stabilized.
    Uses relative change in moving average to identify convergence.
    """
    if len(history) < window * 2:
        return -1
    
    # Get moving averages
    recent_rewards = [h['moving_avg_reward'] for h in history[-window:]]
    prev_rewards = [h['moving_avg_reward'] for h in history[-2*window:-window]]
    
    recent_avg = np.mean(recent_rewards)
    prev_avg = np.mean(prev_rewards)
    
    # Calculate stability (avoid division by zero)
    if abs(prev_avg) < 1e-6:
        change = abs(recent_avg - prev_avg)
    else:
        change = abs(recent_avg - prev_avg) / abs(prev_avg)
        
    if change < threshold:
        return len(history)
    return -1

def save_training_plots(history, plots_dir, policy_version):
    """Generate and save professional training progress plots."""
    os.makedirs(plots_dir, exist_ok=True)
    df = pd.DataFrame(history)
    
    # 1. Reward & Moving Average
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['reward'], alpha=0.3, color='blue', label='Episode Reward')
    plt.plot(df['episode'], df['moving_avg_reward'], color='red', linewidth=2, label='Moving Avg (50)')
    plt.title(f"Training Reward: {policy_version}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"reward_{policy_version}.png"))
    plt.close()

    # 2. Shortage Progress
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['moving_avg_shortage'], color='orange', label='Avg Shortage')
    plt.title(f"Shortage Reduction: {policy_version}")
    plt.xlabel("Episode")
    plt.ylabel("Water Shortage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"shortage_{policy_version}.png"))
    plt.close()

    # 3. Action Distribution (Cumulative)
    plt.figure(figsize=(10, 6))
    actions = ['Equal', 'Pri A', 'Pri B', 'Pri C', 'Conserve']
    counts = [df[f'act_{i}_count'].sum() for i in range(5)]
    plt.bar(actions, counts, color='teal')
    plt.title(f"Cumulative Action Distribution: {policy_version}")
    plt.ylabel("Total Uses")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"actions_{policy_version}.png"))
    plt.close()

def run_experiment(config, policy_version, exp_params):
    exp_name = exp_params['name']
    alpha = exp_params['alpha']
    gamma = exp_params['gamma']
    
    print(f"\n{'='*60}")
    print(f" STARTING EXPERIMENT: {exp_name} ({policy_version})")
    print(f" Params: Alpha={alpha}, Gamma={gamma}")
    print(f"{'='*60}")
    
    seed = config['agent'].get('seed', 42)
    # Global Seed Enforcement
    random.seed(seed)
    np.random.seed(seed)

    env = WaterDistributionEnv(
        reservoir_capacity=config['environment']['reservoir_capacity'],
        refill_amount=config['environment']['refill_amount'],
        max_steps=config['environment']['max_steps'],
        demand_range=config['environment']['demand_ranges']['normal'],
        reward_weights=config['environment'].get('reward_weights'),
        leakage_rate=config['environment'].get('leakage_rate', 0.05),
        seed=seed
    )
    
    agent = QLearningAgent(
        n_actions=5,
        learning_rate=alpha,
        gamma=gamma,
        epsilon=config['agent']['epsilon'],
        epsilon_decay=config['agent']['epsilon_decay'],
        epsilon_min=config['agent']['min_epsilon'],
        seed=seed
    )

    episodes = config['training']['episodes']
    max_steps = config['environment']['max_steps']
    
    history = []
    start_time = time.time()
    convergence_episode = -1
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        total_shortage = 0
        total_wastage = 0
        action_counts = np.zeros(5)
        
        for step in range(max_steps):
            action = agent.choose_action(state)
            action_counts[action] += 1
            
            next_state, reward, done, info = env.step(action)
            
            # Bellman Update with terminal state handling
            agent.update(state, action, reward, next_state, done=done)
            
            state = next_state
            total_reward += reward
            total_shortage += info['shortage']
            total_wastage += info['wastage']
            
            if done:
                break
        
        agent.decay_epsilon()
        
        # Calculate moving averages
        window = config['training'].get('moving_avg_window', 50)
        mv_reward = total_reward
        mv_shortage = total_shortage
        mv_wastage = total_wastage
        if len(history) > 0:
            prev_rewards = [h['reward'] for h in history[-(window-1):]] + [total_reward]
            prev_shortages = [h['shortage'] for h in history[-(window-1):]] + [total_shortage]
            prev_wastages = [h['wastage'] for h in history[-(window-1):]] + [total_wastage]
            mv_reward = np.mean(prev_rewards)
            mv_shortage = np.mean(prev_shortages)
            mv_wastage = np.mean(prev_wastages)
            
        step_data = {
            "episode": ep + 1,
            "reward": total_reward,
            "shortage": total_shortage,
            "wastage": total_wastage,
            "moving_avg_reward": mv_reward,
            "moving_avg_shortage": mv_shortage,
            "moving_avg_wastage": mv_wastage,
            "epsilon": agent.epsilon
        }
        # Add action counts
        for i in range(5):
            step_data[f"act_{i}_count"] = action_counts[i]
            
        history.append(step_data)
        
        # Check convergence
        if convergence_episode == -1 and ep > window * 3:
            conv = check_convergence(history, window)
            if conv != -1:
                convergence_episode = conv

        if (ep + 1) % 500 == 0:
            print(f" Episode {ep+1:4d}/{episodes} | Avg Reward: {mv_reward:8.2f} | Epsilon: {agent.epsilon:.3f}")

    training_duration = time.time() - start_time
    stats = agent.get_statistics()
    
    # Save Policy
    policy_dir = config['training']['output_paths']['policy_dir']
    os.makedirs(policy_dir, exist_ok=True)
    policy_path = os.path.join(policy_dir, f"{policy_version}.pkl")
    agent.save_policy(policy_path)
    
    # Generate Plots
    plots_dir = config['training']['output_paths']['plots_dir']
    save_training_plots(history, plots_dir, policy_version)
    
    # MLOps Experiment Tracking
    logs_dir = config['training']['output_paths']['logs_dir']
    os.makedirs(logs_dir, exist_ok=True)
    
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(logs_dir, f"training_logs_{policy_version}.csv"), index=False)
    
    run_id = f"run_{policy_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_data = {
        "run_id": run_id,
        "policy_version": policy_version,
        "alpha": alpha,
        "gamma": gamma,
        "final_avg_reward": float(mv_reward),
        "final_avg_shortage": float(mv_shortage),
        "convergence_episode": convergence_episode,
        "learned_states": stats['states_learned'],
        "training_duration": training_duration
    }
    
    # Update master experiments file
    exp_json_path = os.path.join(logs_dir, "experiments.json")
    experiments = []
    if os.path.exists(exp_json_path):
        with open(exp_json_path, "r") as f:
            experiments = json.load(f)
    experiments.append(exp_data)
    with open(exp_json_path, "w") as f:
        json.dump(experiments, f, indent=4)
        
    # Print Professional Summary
    print(f"\n{'-'*40}")
    print(f" TRAINING SUMMARY: {policy_version}")
    print(f"{'-'*40}")
    print(f" Final Avg Reward:   {mv_reward:.2f}")
    print(f" Final Avg Shortage: {mv_shortage:.2f}")
    print(f" Learned States:     {stats['states_learned']}")
    print(f" Convergence At:     {convergence_episode if convergence_episode != -1 else 'No convergence'}")
    print(f" Duration:           {training_duration:.2f}s")
    print(f"{'-'*40}\n")
    
    return exp_data

def train(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = {}
    exp_configs = config['training']['experiments']
    
    for version, params in exp_configs.items():
        meta = run_experiment(config, version, params)
        results[version] = meta

    # Save aggregated metadata
    policy_dir = config['training']['output_paths']['policy_dir']
    with open(os.path.join(policy_dir, "policy_metadata.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nAll training experiments completed successfully. Plots saved to results/plots/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Water Distribution RL Agent")
    parser.add_argument("--config", type=str, default="configs/qlearning.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    train(args.config)
