import numpy as np

class WaterDistributionEnv:
    """
    Adaptive Water Distribution Simulator for Reinforcement Learning.
    
    This environment simulates a central reservoir supplying water to three distinct zones
    (Zone A, Zone B, Zone C). The goal of the RL agent is to manage water allocation
    to minimize shortages and wastage while maintaining reservoir stability.
    
    State Space (Discretized):
        - Reservoir Level (8 bins: Empty to Full)
        - Demand Zone A (6 bins: Low to Very High)
        - Demand Zone B (6 bins: Low to Very High)
        - Demand Zone C (6 bins: Low to Very High)
        
    Action Space:
        - 0: Equal Distribution (Fairly split available water among all zones)
        - 1: Prioritize Zone A (Full demand for A, then split remainder)
        - 2: Prioritize Zone B (Full demand for B, then split remainder)
        - 3: Prioritize Zone C (Full demand for C, then split remainder)
        - 4: Conservation Mode (Allocate only 25% of demand to preserve reservoir)
    """
    
    def __init__(self, reservoir_capacity=100, refill_amount=30, max_steps=100, 
                 demand_range=(5, 25), reward_weights=None, leakage_rate=0.05, seed=None):
        """
        Initialize the environment.
        """
        self.reservoir_capacity = reservoir_capacity
        self.refill_amount = refill_amount
        self.max_steps = max_steps
        self.demand_range = demand_range
        self.leakage_rate = leakage_rate
        self.reward_weights = reward_weights or {"shortage": 1.2, "wastage": 1.0, "overflow": 1.5}
        
        # Scenario configuration
        self.scenario = "normal"
        self.uneven_ranges = None
        
        # Local Random Number Generator
        self.rng = np.random.default_rng(seed)
        
        # History tracking for analysis and evaluation
        self.history = {
            "reservoir": [],
            "reward": [],
            "shortage": [],
            "wastage": [],
            "action": [],
            "demands": [],
            "supply": []
        }
        
        # Initialize state
        self.reset()

    def set_scenario(self, scenario, demand_range, refill_amount=None, uneven_ranges=None):
        """
        Configure the environment for specific testing scenarios.
        """
        self.scenario = scenario
        self.demand_range = demand_range
        if refill_amount is not None:
            self.refill_amount = refill_amount
        self.uneven_ranges = uneven_ranges

    def reset(self):
        """
        Reset the environment to initial state for a new episode.
        """
        self.reservoir_level = self.reservoir_capacity
        self.current_step = 0
        self.demands = self.generate_demands()
        
        for key in self.history:
            self.history[key] = []
            
        return self.get_state()

    def generate_demands(self):
        """
        Generate water demand for 3 zones. 
        Includes stochastic spikes (10% probability) for better strategy testing.
        """
        if self.scenario == "uneven" and self.uneven_ranges is not None:
            d = [
                self.rng.integers(self.uneven_ranges[0][0], self.uneven_ranges[0][1] + 1),
                self.rng.integers(self.uneven_ranges[1][0], self.uneven_ranges[1][1] + 1),
                self.rng.integers(self.uneven_ranges[2][0], self.uneven_ranges[2][1] + 1)
            ]
            demands = np.array(d, dtype=float)
        else:
            demands = self.rng.integers(self.demand_range[0], self.demand_range[1] + 1, size=3).astype(float)
            
        # Introduce stochastic demand spikes (10% chance)
        if self.rng.random() < 0.10:
            target_zone = self.rng.integers(0, 3)
            spike_factor = self.rng.uniform(1.5, 2.0)
            demands[target_zone] *= spike_factor
            
        return demands

    def get_state(self):
        """
        Return the current state discretized for RL agents.
        Updated: 8 bins for reservoir, 6 bins for demands.
        """
        res_bins = 8
        dem_bins = 6
        
        reservoir_bin = min(int(self.reservoir_level / (self.reservoir_capacity / res_bins)), res_bins - 1)
        # Using a fixed reference for demand scaling (max possible demand around 60-80)
        max_d_ref = 60.0
        demand_bins = [min(int(d / (max_d_ref / dem_bins)), dem_bins - 1) for d in self.demands]
        return (reservoir_bin, *demand_bins)

    def _allocate_water(self, action):
        """
        Internal logic for water distribution based on selected strategy.
        """
        supply = np.zeros(3)
        remaining_water = self.reservoir_level
        
        if action == 0: # Equal Distribution
            for i in range(3):
                share = remaining_water / (3 - i)
                allocated = min(share, self.demands[i])
                supply[i] = allocated
                remaining_water -= allocated
                
        elif action in [1, 2, 3]: # Priority Distribution
            priority_idx = action - 1
            other_indices = [i for i in range(3) if i != priority_idx]
            
            allocated = min(remaining_water, self.demands[priority_idx])
            supply[priority_idx] = allocated
            remaining_water -= allocated
            
            for i, idx in enumerate(other_indices):
                share = remaining_water / (2 - i)
                allocated = min(share, self.demands[idx])
                supply[idx] = allocated
                remaining_water -= allocated

        elif action == 4: # Conservation Mode
            # Distribute only 25% of demand to preserve reservoir
            for i in range(3):
                allocated = min(remaining_water, self.demands[i] * 0.25)
                supply[i] = allocated
                remaining_water -= allocated
        else:
            raise ValueError(f"Invalid action {action}.")
                
        return supply

    def step(self, action):
        """
        Execute one time step in the environment.
        """
        current_demands = self.demands.copy()
        supply = self._allocate_water(action)
        
        self.reservoir_level -= np.sum(supply)
        
        # Calculate Physically Realistic Supply & Leakage
        effective_supply = supply * (1.0 - self.leakage_rate)
        leakage_wastage = np.sum(supply) * self.leakage_rate
        
        # Refill and Overflow
        self.reservoir_level += self.refill_amount
        overflow_wastage = 0.0
        if self.reservoir_level > self.reservoir_capacity:
            overflow_wastage = self.reservoir_level - self.reservoir_capacity
            self.reservoir_level = self.reservoir_capacity
            
        # Final Metrics
        total_wastage = leakage_wastage + overflow_wastage
        shortage = np.sum(np.maximum(0, current_demands - effective_supply))
        self.reservoir_level = max(0.0, self.reservoir_level)
        
        # Reward Calculation
        w_s = self.reward_weights.get("shortage", 1.2)
        w_w = self.reward_weights.get("wastage", 1.0)
        w_o = self.reward_weights.get("overflow", 1.5)
        reward = -(w_s * shortage + w_w * total_wastage + w_o * overflow_wastage)
        
        # ADVANCED REWARD SHAPING
        utilization = self.reservoir_level / self.reservoir_capacity
        
        # 1. Sustainability/Future Stability Bonus
        if 0.3 <= utilization <= 0.6:
            reward += 8.0
        elif utilization < 0.1:
            reward -= 20.0 # Heavy penalty for near-empty reservoir
            
        # 2. Conservation Mode Survival Bonus
        if action == 4 and utilization < 0.25:
            reward += 15.0 # Encourage preservation during scarcity
        
        # 3. Small bonus for keeping reservoir in the middle range (previous bonus kept)
        if 0.3 <= utilization <= 0.7:
             reward += 2.0 # Extra small nudge (total +10 if in 0.3-0.6)
        
        # History Tracking
        self.history["reservoir"].append(self.reservoir_level)
        self.history["reward"].append(reward)
        self.history["shortage"].append(shortage)
        self.history["wastage"].append(total_wastage)
        self.history["action"].append(action)
        self.history["demands"].append(current_demands)
        self.history["supply"].append(supply)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        self.demands = self.generate_demands()
        next_state = self.get_state()
        
        info = {
            "shortage": shortage,
            "wastage": total_wastage,
            "tank_utilization": utilization,
            "demands": current_demands,
            "supply": supply,
            "step": self.current_step
        }

        return next_state, reward, done, info

    def render(self):
        """
        Professional terminal-based visualization.
        """
        if not self.history["action"]:
            print("Environment reset. No step data to render yet.")
            return

        last_demands = self.history["demands"][-1]
        last_supply = self.history["supply"][-1]
        last_reward = self.history["reward"][-1]
        last_shortage = self.history["shortage"][-1]
        last_wastage = self.history["wastage"][-1]
        last_action = self.history["action"][-1]
        
        action_map = {0: "Equal", 1: "Pri A", 2: "Pri B", 3: "Pri C", 4: "Conserve"}

        print("\n" + "="*50)
        print(f" STEP: {self.current_step:3d} | SCENARIO: {self.scenario.upper()}")
        print(f" ACTION: {action_map.get(last_action, 'Unknown')}")
        print("-" * 50)
        print(f" Reservoir Level: {self.reservoir_level:6.2f} / {self.reservoir_capacity:3d}")
        print(f" Zone Demands:   A:{last_demands[0]:4.1f} | B:{last_demands[1]:4.1f} | C:{last_demands[2]:4.1f}")
        print(f" Zone Supplies:  A:{last_supply[0]:4.1f} | B:{last_supply[1]:4.1f} | C:{last_supply[2]:4.1f}")
        print(f" Shortage: {last_shortage:6.2f} | Wastage: {last_wastage:6.2f}")
        print(f" REWARD:   {last_reward:6.2f}")
        print("="*50 + "\n")
