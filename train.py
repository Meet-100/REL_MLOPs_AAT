import yaml
import os
import json
import pandas as pd
import numpy as np
import random
import argparse
import time
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sim.water_env import WaterDistributionEnv
from agents.qlearning_agent import QLearningAgent
from datetime import datetime

# Configure MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Water_Distribution_RL")

def check_convergence(history, window, threshold=0.1):
    """
    Check if the moving average reward has stabilized.
    """
    if len(history) < window * 2:
        return -1
    
    recent_rewards = [h['moving_avg_reward'] for h in history[-window:]]
    prev_rewards = [h['moving_avg_reward'] for h in history[-2*window:-window]]
    
    recent_avg = np.mean(recent_rewards)
    prev_avg = np.mean(prev_rewards)
    
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
    plot_path = os.path.join(plots_dir, f"reward_{policy_version}.png")
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path)

    # 2. Shortage Progress
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['moving_avg_shortage'], color='orange', label='Avg Shortage')
    plt.title(f"Shortage Reduction: {policy_version}")
    plt.xlabel("Episode")
    plt.ylabel("Water Shortage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    shortage_plot_path = os.path.join(plots_dir, f"shortage_{policy_version}.png")
    plt.savefig(shortage_plot_path)
    plt.close()
    mlflow.log_artifact(shortage_plot_path)

def run_experiment(config, policy_version, exp_params):
    exp_name = exp_params['name']
    alpha = exp_params['alpha']
    gamma = exp_params['gamma']
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{policy_version}_{exp_name}"):
        print(f"\n{'='*60}")
        print(f" STARTING EXPERIMENT: {exp_name} ({policy_version})")
        print(f" Params: Alpha={alpha}, Gamma={gamma}")
        print(f"{'='*60}")
        
        # Log Parameters to MLflow
        mlflow.log_params({
            "policy_version": policy_version,
            "alpha": alpha,
            "gamma": gamma,
            "episodes": config['training']['episodes'],
            "epsilon_decay": config['agent']['epsilon_decay'],
            "leakage_rate": config['environment'].get('leakage_rate', 0.15),
            "reward_weights": config['environment'].get('reward_weights')
        })
        
        seed = config['agent'].get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)

        env = WaterDistributionEnv(
            reservoir_capacity=config['environment']['reservoir_capacity'],
            refill_amount=config['environment']['refill_amount'],
            max_steps=config['environment']['max_steps'],
            demand_range=config['environment']['demand_ranges']['normal'],
            reward_weights=config['environment'].get('reward_weights'),
            leakage_rate=config['environment'].get('leakage_rate', 0.15),
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
            
            for step in range(max_steps):
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done=done)
                state = next_state
                total_reward += reward
                total_shortage += info['shortage']
                total_wastage += info['wastage']
                if done: break
            
            agent.decay_epsilon()
            
            window = config['training'].get('moving_avg_window', 50)
            mv_reward = total_reward
            if len(history) > 0:
                prev_rewards = [h['reward'] for h in history[-(window-1):]] + [total_reward]
                mv_reward = np.mean(prev_rewards)
                
            step_data = {
                "episode": ep + 1,
                "reward": total_reward,
                "shortage": total_shortage,
                "moving_avg_reward": mv_reward,
                "epsilon": agent.epsilon
            }
            history.append(step_data)
            
            if convergence_episode == -1 and ep > window * 3:
                conv = check_convergence(history, window)
                if conv != -1:
                    convergence_episode = conv

            if (ep + 1) % 500 == 0:
                print(f" Episode {ep+1:4d}/{episodes} | Avg Reward: {mv_reward:8.2f} | Epsilon: {agent.epsilon:.3f}")
                mlflow.log_metric("avg_reward", mv_reward, step=ep+1)

        training_duration = time.time() - start_time
        stats = agent.get_statistics()
        
        # Save & Log Policy
        policy_dir = config['training']['output_paths']['policy_dir']
        os.makedirs(policy_dir, exist_ok=True)
        policy_path = os.path.join(policy_dir, f"{policy_version}.pkl")
        agent.save_policy(policy_path)
        mlflow.log_artifact(policy_path)
        
        # Log final metrics
        mlflow.log_metrics({
            "final_avg_reward": float(mv_reward),
            "convergence_episode": float(convergence_episode),
            "learned_states": float(stats['states_learned']),
            "training_duration": float(training_duration)
        })
        
        # Generate & Log Plots
        plots_dir = config['training']['output_paths']['plots_dir']
        save_training_plots(history, plots_dir, policy_version)
        
        # Save local logs
        logs_dir = config['training']['output_paths']['logs_dir']
        os.makedirs(logs_dir, exist_ok=True)
        csv_path = os.path.join(logs_dir, f"training_logs_{policy_version}.csv")
        pd.DataFrame(history).to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact("configs/qlearning.yaml")

        # Tags
        mlflow.set_tag("policy_type", "QLearning")
        mlflow.set_tag("deployment_ready", "true" if convergence_episode != -1 else "false")
        
        exp_data = {
            "run_id": mlflow.active_run().info.run_id,
            "policy_version": policy_version,
            "final_avg_reward": float(mv_reward),
            "convergence_episode": convergence_episode,
            "learned_states": stats['states_learned'],
            "training_duration": training_duration
        }
        
        return exp_data

def train(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = {}
    for version, params in config['training']['experiments'].items():
        meta = run_experiment(config, version, params)
        results[version] = meta

    policy_dir = config['training']['output_paths']['policy_dir']
    with open(os.path.join(policy_dir, "policy_metadata.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("\nTraining completed. View results in MLflow: mlflow ui")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qlearning.yaml")
    args = parser.parse_args()
    train(args.config)
