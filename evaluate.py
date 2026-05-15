import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
import json
import mlflow
from sim.water_env import WaterDistributionEnv
from agents.qlearning_agent import QLearningAgent

# Configure MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Water_Distribution_Evaluation")

def rule_based_action(state, env):
    """Limited Static Heuristic Baseline."""
    utilization = env.reservoir_level / env.reservoir_capacity
    if utilization > 0.70:
        return 0 # Equal distribution
    else:
        return 3 # Priority for Zone C

def run_evaluation(env, agent, episodes, max_steps, mode="RL", episode_seeds=None):
    results = []
    for ep in range(episodes):
        if episode_seeds is not None:
            env.rng = np.random.default_rng(episode_seeds[ep])
        state = env.reset()
        total_shortage = 0
        total_wastage = 0
        total_reward = 0
        utilization = []
        for step in range(max_steps):
            if "RL" in mode:
                action = agent.choose_action(state)
            elif mode == "Smart_Baseline":
                action = rule_based_action(state, env)
            else: action = 0
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            total_shortage += info['shortage']
            total_wastage += info['wastage']
            utilization.append(info['tank_utilization'])
            if done: break
        results.append({
            "mode": mode, "scenario": env.scenario, "reward": total_reward,
            "shortage": total_shortage, "wastage": total_wastage,
            "avg_utilization": np.mean(utilization)
        })
    return pd.DataFrame(results)

def generate_visualizations(final_results, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Reward Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=final_results, x="scenario", y="reward", hue="mode")
    plt.title("Reward Comparison")
    reward_plot = os.path.join(plots_dir, "reward_comparison.png")
    plt.savefig(reward_plot)
    plt.close()
    mlflow.log_artifact(reward_plot)

    # Shortage Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=final_results, x="scenario", y="shortage", hue="mode")
    plt.title("Shortage Comparison")
    shortage_plot = os.path.join(plots_dir, "shortage_comparison.png")
    plt.savefig(shortage_plot)
    plt.close()
    mlflow.log_artifact(shortage_plot)

def evaluate(config_path, demo=False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with mlflow.start_run(run_name="Scientific_Evaluation"):
        eval_seed = config['evaluation'].get('seed', 42)
        np.random.seed(eval_seed)

        env = WaterDistributionEnv(
            reservoir_capacity=config['environment']['reservoir_capacity'],
            refill_amount=config['environment']['refill_amount'],
            max_steps=config['environment']['max_steps'],
            demand_range=config['environment']['demand_ranges']['normal'],
            reward_weights=config['environment'].get('reward_weights'),
            leakage_rate=config['environment'].get('leakage_rate', 0.15)
        )

        policies = ["policy_v1", "policy_v2"]
        agents = {}
        policy_dir = config['training']['output_paths']['policy_dir']
        for p in policies:
            agent = QLearningAgent(n_actions=5)
            path = os.path.join(policy_dir, f"{p}.pkl")
            if os.path.exists(path):
                agent.load_policy(path)
                agents[p] = agent

        if demo:
            run_demo(env, agents.get("policy_v1", QLearningAgent(5)), config)
            return

        test_episodes = config['evaluation']['test_episodes']
        max_steps = config['environment']['max_steps']
        master_seeds = np.random.randint(0, 100000, size=test_episodes)
        refills = config['environment'].get('scenario_refills', {})
        
        scenarios = {
            "normal": {"range": config['environment']['demand_ranges']['normal'], "refill": refills.get('normal', 30)},
            "peak": {"range": config['environment']['demand_ranges']['peak'], "refill": refills.get('peak', 15)},
            "uneven": {"range": [0,0], "uneven": config['environment']['demand_ranges']['uneven'], "refill": refills.get('uneven', 25)},
            "shortage": {"range": config['environment']['demand_ranges']['shortage'], "refill": refills.get('shortage', 3)},
            "extreme": {"range": config['environment']['demand_ranges']['extreme'], "refill": refills.get('extreme', 5)}
        }

        all_results = []
        for sc_name, sc_params in scenarios.items():
            env.set_scenario(sc_name, sc_params.get("range"), sc_params.get("refill"), sc_params.get("uneven"))
            for p_name, agent in agents.items():
                res = run_evaluation(env, agent, test_episodes, max_steps, mode=f"RL_{p_name}", episode_seeds=master_seeds)
                all_results.append(res)
            res_eq = run_evaluation(env, None, test_episodes, max_steps, mode="Equal_Baseline", episode_seeds=master_seeds)
            res_sm = run_evaluation(env, None, test_episodes, max_steps, mode="Smart_Baseline", episode_seeds=master_seeds)
            all_results.extend([res_eq, res_sm])

        final_results = pd.concat(all_results, ignore_index=True)
        ranking = final_results.groupby("mode")[["reward", "shortage", "wastage"]].mean()
        
        # Log metrics to MLflow
        for mode, row in ranking.iterrows():
            mlflow.log_metric(f"{mode}_avg_reward", row['reward'])
            mlflow.log_metric(f"{mode}_avg_shortage", row['shortage'])

        best_rl = ranking[ranking.index.str.contains("RL")]["reward"].idxmax()
        mlflow.set_tag("best_rl_policy", best_rl)
        
        generate_visualizations(final_results, config['evaluation']['plots_dir'])
        
        report_path = config['evaluation']['report_path']
        generate_report(final_results, ranking, config, best_rl, "Smart_Baseline", {})
        mlflow.log_artifact(report_path)
        mlflow.log_artifact("configs/qlearning.yaml")

    print(f"\nEvaluation Summary:\n{ranking.to_string()}")

def generate_report(results, ranking, config, best_rl, baseline, training_meta):
    """Generates a professional academic report."""
    report_path = config['evaluation']['report_path']
    with open(report_path, "w") as f:
        f.write("# Final Evaluation Report: Adaptive Water Distribution RL\n\n")
        f.write("## Comparative Performance\n")
        f.write(ranking.to_markdown() + "\n\n")
        f.write("## MLflow Tracking\n")
        f.write("Experiments and artifacts are tracked using MLflow. Run `mlflow ui` to view detailed logs.\n")

def run_demo(env, agent, config):
    env.set_scenario("normal", config['environment']['demand_ranges']['normal'])
    state = env.reset()
    for _ in range(15):
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        env.render()
        state = next_state
        if done: break
        time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qlearning.yaml")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    evaluate(args.config, args.demo)
