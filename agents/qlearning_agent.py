import numpy as np
import random
import pickle

class QLearningAgent:
    """
    Q-Learning Agent for Adaptive Water Distribution Optimization.
    
    This agent implements the Tabular Q-Learning algorithm, a model-free reinforcement 
    learning technique used to learn the value of an action in a particular state.
    
    Mathematical Foundation:
        The agent learns the Q-function Q(s, a), which estimates the expected cumulative 
        future reward for taking action 'a' in state 's'. It follows the Bellman Equation:
        
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
        
        Where:
        - alpha (learning_rate): How much new information overrides old information.
        - gamma (discount factor): Importance of future rewards (0 to 1).
        - max(Q(s', a')): Estimate of optimal future value.
        
    Exploration Strategy:
        Uses Epsilon-Greedy exploration to balance the "Exploration vs. Exploitation" tradeoff.
        - Exploration: Trying random actions to discover new strategies (prob epsilon).
        - Exploitation: Using known best actions to maximize reward (prob 1-epsilon).
    """
    
    def __init__(self, n_actions, learning_rate=0.1, gamma=0.95, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01, seed=None):
        """
        Initialize the Q-learning agent.
        
        Args:
            n_actions (int): Number of possible actions.
            learning_rate (float): Step size for Q-updates (alpha).
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Multiplicative decay for epsilon.
            epsilon_min (float): Minimum exploration rate.
            seed (int): Random seed for reproducibility.
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Internal state
        self.is_training = True
        self.q_table = {}
        
        # Reproducibility
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def set_training_mode(self):
        """Enable exploration and Q-table updates."""
        self.is_training = True

    def set_evaluation_mode(self):
        """Disable exploration for deterministic performance testing."""
        self.is_training = False
        self.epsilon = 0.0

    def get_q_values(self, state):
        """
        Return Q-values for a state, initializing with zeros if the state is new.
        State is expected to be a hashable tuple (e.g., discretized bins).
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy with random tie-breaking.
        """
        # 1. Exploration (only during training)
        if self.is_training and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        
        # 2. Exploitation (Greedy action selection)
        q_values = self.get_q_values(state)
        max_q = np.max(q_values)
        
        # Random tie-breaking: Find all actions that share the maximum value
        # This prevents bias towards the first action index in the Q-table.
        best_actions = np.where(q_values == max_q)[0]
        return self.rng.choice(best_actions)

    def update(self, state, action, reward, next_state, done=False):
        """
        Update the Q-table using the Bellman Equation.
        Only updates during training mode.
        Handles terminal states correctly by excluding future rewards.
        """
        if not self.is_training:
            return

        if not (0 <= action < self.n_actions):
            raise ValueError(f"Invalid action index {action}. Must be in [0, {self.n_actions-1}]")

        # Current Q-value
        current_q = self.get_q_values(state)[action]
        
        # Temporal Difference (TD) Target
        if done:
            # For terminal states, there are no future rewards
            target = reward
        else:
            # Maximum possible Q-value for the next state
            max_next_q = np.max(self.get_q_values(next_state))
            target = reward + self.gamma * max_next_q
        
        # Update Q-value towards the target
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """
        Reduce the exploration rate epsilon, ensuring it doesn't drop below epsilon_min.
        Typically called at the end of each episode.
        """
        if self.is_training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_statistics(self):
        """
        Return metrics describing the current state of the Q-table.
        Useful for monitoring convergence and policy complexity.
        """
        if not self.q_table:
            return {"states_learned": 0, "avg_q": 0, "max_q": 0}
        
        all_q_values = np.array(list(self.q_table.values()))
        return {
            "states_learned": len(self.q_table),
            "avg_q": float(np.mean(all_q_values)),
            "max_q": float(np.max(all_q_values)),
            "table_size_bytes": self.q_table.__sizeof__()
        }

    def save_policy(self, file_path):
        """
        Save the learned policy and agent configuration for later use or deployment.
        """
        data = {
            "q_table": self.q_table,
            "hyperparameters": {
                "n_actions": self.n_actions,
                "learning_rate": self.lr,
                "gamma": self.gamma,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min
            },
            "metadata": {
                "states_learned": len(self.q_table),
                "last_epsilon": self.epsilon,
                "seed": self.seed
            }
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Policy saved to {file_path} (States: {len(self.q_table)})")

    def load_policy(self, file_path):
        """
        Load a saved policy and restore agent hyperparameters.
        Handles both the new structured format and the legacy direct Q-table format.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and "q_table" in data:
            self.q_table = data["q_table"]
            # Restore hyperparameters if available
            hp = data.get("hyperparameters", {})
            self.n_actions = hp.get("n_actions", self.n_actions)
            self.lr = hp.get("learning_rate", self.lr)
            self.gamma = hp.get("gamma", self.gamma)
        else:
            # Legacy format: data is just the q_table dictionary
            self.q_table = data
        
        # Switch to evaluation mode
        self.set_evaluation_mode()
        print(f"Policy loaded from {file_path}. Agent set to EVALUATION mode.")
