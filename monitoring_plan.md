# Monitoring Plan: Adaptive Water Distribution Optimization

## 1. Introduction
Deploying a Reinforcement Learning agent in a real-world critical infrastructure setting like water distribution requires a robust, multi-layered monitoring strategy. This plan outlines the metrics, alerts, and fail-safes necessary for safe deployment.

## 2. Real-Time System Monitoring
- **Reservoir Telemetry**: Continuous monitoring of the central reservoir level to prevent catastrophic depletion or overflow.
- **Demand Sensors**: IoT flow meters at the zonal level to measure real-time consumption vs. predicted demand.
- **Leakage Detection**: Acoustic and pressure sensors along distribution lines to differentiate between legitimate demand and system leaks.

## 3. RL Policy Monitoring
- **Reward Tracking**: Monitor the moving average of the RL agent's reward in production. A sudden drop indicates a distribution shift (e.g., unexpected demand spikes, sensor failure).
- **Action Distribution**: Track the frequency of priority actions (e.g., prioritizing Zone A over B). If the agent becomes hyper-focused on one zone, it may have overfit or encountered biased sensor data.
- **State Space Coverage**: Monitor if the agent encounters states significantly different from those seen during training.

## 4. Safety Constraints & Fallbacks
- **Minimum Reservoir Level**: Hard-coded safety constraint ensuring the reservoir never drops below 10% capacity. If the RL agent attempts an action that would violate this, the system automatically overrides it.
- **Rule-Based Fallback**: If RL metrics degrade or sensor data quality drops below a threshold, the system will seamlessly switch to the `Smart Rule-Based Baseline` to guarantee baseline performance.

## 5. Anomaly Detection & Alerts
- **Demand Anomalies**: Statistical models (e.g., Isolation Forests) to detect abnormal demand spikes indicative of burst pipes or major events.
- **Alert Tiers**:
  - *Tier 1 (Warning)*: RL reward drops by 10%. Action: Log and notify engineers.
  - *Tier 2 (Critical)*: Reservoir drops below 20%. Action: Override RL agent and page on-call operator.
  - *Tier 3 (Emergency)*: Multiple sensor failures. Action: Switch to manual/fixed operation.

## 6. Ethical & Fairness Considerations
- **Equitable Distribution Audit**: Regular audits to ensure the RL policy does not systematically marginalize any specific socio-economic zone, even if it maximizes overall system reward.
- **Human-in-the-Loop**: Major distribution changes or overrides must be authorized by a human operator.

## 7. Continuous Learning Pipeline
- **Data Collection**: Store state-action-reward tuples from production.
- **Retraining**: Schedule periodic retraining (e.g., monthly) using fresh data to adapt to seasonal demand shifts.
- **Shadow Mode**: Deploy new policy versions (`policy_v2.pkl`) in shadow mode alongside the active policy to evaluate performance before a hard switch.
