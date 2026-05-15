# Viva Preparation: Adaptive Water Distribution Optimization using RL

This document contains 15+ comprehensive questions and answers to prepare for the academic evaluation of this project.

## Reinforcement Learning & Q-Learning
**1. What is the main objective of your Reinforcement Learning agent?**
The agent's objective is to learn an optimal policy for distributing water from a central reservoir to three urban zones. It aims to minimize the total unmet demand (shortage) and minimize water wastage (overflow) while adapting to stochastic demand patterns.

**2. Why did you choose Q-Learning for this problem over other RL algorithms?**
Q-Learning is a model-free, off-policy algorithm that is highly effective for environments with discrete state and action spaces. Given that our water distribution problem could be discretized into specific reservoir levels and demand bins, tabular Q-Learning provided a transparent, easily interpretable, and computationally efficient solution compared to complex Deep RL methods (like DQN or PPO), which are harder to debug and explain in a baseline academic context.

**3. Explain the State and Action space of your environment.**
- **State**: The state is a tuple consisting of the discretized reservoir level (5 bins) and the discretized current demands for the three zones (4 bins each).
- **Action**: The action space is discrete with 4 options: (0) Equal distribution, (1) Prioritize Zone A, (2) Prioritize Zone B, (3) Prioritize Zone C.

**4. How is the Reward Function defined, and why?**
The reward function is defined as `Reward = -(w1 * shortage + w2 * wastage + w3 * overflow)`. It is entirely composed of penalties because the goal is minimization. By tying the reward directly to shortage and wastage, we align the agent's objective with the physical constraints of the real world.

**5. What is the Exploration vs. Exploitation dilemma, and how did you handle it?**
The dilemma is whether the agent should explore new, untried actions to find potentially better rewards, or exploit the best-known action to maximize immediate reward. We handled this using an $\epsilon$-greedy strategy, where the agent explores with probability $\epsilon$ and exploits with probability $1-\epsilon$. We apply an $\epsilon$-decay over training episodes so the agent explores heavily at first and exploits more as it converges.

**6. How do you determine if your Q-Learning agent has converged?**
We determine convergence by monitoring the moving average of the reward over a window of episodes (e.g., 50 or 100 episodes). When the difference between successive moving averages falls below a very small threshold, we consider the policy stabilized and converged.

## MLOps & Reproducibility
**7. Why is MLOps important in a Reinforcement Learning project?**
RL training is notoriously unstable and highly sensitive to hyperparameters. MLOps ensures that every training run is systematically tracked (hyperparameters, metrics, duration) and every resulting policy is versioned. This allows for rigorous experimentation, debugging, and deployment tracking.

**8. How did you ensure the reproducibility of your experiments?**
Reproducibility is guaranteed by centralizing all hyperparameters in a `qlearning.yaml` config file and applying a global seed across all random number generators (`numpy`, `random`, and the environment). This ensures that running the same config file yields the exact same training curve and policy.

**9. Explain your experiment tracking setup.**
Our `train.py` script automatically logs the run ID, timestamp, hyperparameters (alpha, gamma, epsilon decay), and key metrics (convergence episode, average shortage) to both a JSON file and a CSV file (`logs/experiments.csv`). It also saves a `policy_metadata.json` alongside the `.pkl` policy files to link the model artifact to its training context.

## Scenarios & Evaluation
**10. How did you baseline your RL agent's performance?**
We compared the RL agent against two baselines:
1. **Fixed Equal Distribution**: A naive approach that always divides available water by 3.
2. **Smart Rule-Based**: A heuristic approach that prioritizes the zone with the maximum current demand.

**11. Why did you implement Scenario-Based Stress Testing?**
In the real world, water demand is rarely uniform. By testing the policy against scenarios like "Peak-Hour", "Uneven Distribution", and "Reservoir Shortage", we prove the robustness of the RL policy and demonstrate its ability to adapt to non-standard conditions better than fixed rules.

**12. In which scenario did the RL agent struggle the most, and why?**
The agent typically struggles in "Extreme Random" scenarios where the demand exceeds the reservoir capacity completely. In these cases, shortage is mathematically inevitable, and the agent can only optimize who gets penalized, rather than preventing the penalty altogether.

## SDG Impact & Future Scope
**13. How does this project map to the Sustainable Development Goals (SDGs)?**
- **SDG 6 (Clean Water and Sanitation):** By penalizing wastage and optimizing shortage, the system promotes efficient use of fresh water resources.
- **SDG 11 (Sustainable Cities and Communities):** Smart infrastructure that dynamically adapts to urban demand makes cities more resilient to population growth and climate-induced water scarcity.

**14. What are the limitations of your current approach?**
The primary limitation is the use of Tabular Q-Learning, which requires a discretized state space. If we expand the system to 50 zones, the state space will explode (Curse of Dimensionality), making a Q-table unfeasible. The simulation also lacks physical dynamics like pipe pressure or travel time.

**15. If you had more time, how would you improve this project?**
I would upgrade the agent to use Deep Q-Networks (DQN) or Proximal Policy Optimization (PPO) to handle continuous state spaces. I would also enhance the simulator to include leakage simulation, pipe routing, and dynamic pricing based on time of day.
