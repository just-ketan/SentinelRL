
# SentinelRL

## Reinforcement Learning Robustness under Specification Drift

SentinelRL is a research-oriented reinforcement learning system designed to study **policy robustness under controlled non-stationarity**. 

Instead of optimizing benchmark performance, SentinelRL investigates:

> *What happens when a trained RL policy is deployed into an environment that changes?*

The project focuses on controlled environmental drift, degradation analysis, and algorithmic comparisons between DQN and Double DQN.

---

# 🚀 Project Purpose

Most reinforcement learning research assumes stationary environments. However, real-world systems change over time due to:

- Objective shifts
- Sensor noise
- Environmental drift
- Changing reward structures
- Distribution shift

SentinelRL provides a controlled experimental framework to measure:

- Graceful degradation
- Catastrophic collapse
- Estimation bias effects
- Algorithmic robustness

---

# 🏗 System Architecture

SentinelRL follows a modular architecture.

```
sentinelrl/
│
├── agents/
│   ├── base_agent.py
│   ├── dqn_agent.py
│   └── double_dqn_agent.py
│
├── environments/
│   ├── clean_env.py
│   └── drifted_env.py
│
├── replay/
│   └── replay_buffer.py
│
├── networks/
│   └── q_network.py
│
├── training/
│   └── train.py
│
├── evaluation/
│   ├── stress_test.py
│   ├── metrics.py
│   └── visualizer.py
│
├── results/
├── models/
└── requirements.txt
```

---

# 🧠 Architectural Overview
## High Level System Architecture
```
Agent  ↔  Environment  ↔  Replay Buffer
  │            │              │
  └── Network  │              │
       │       │              │
       └── Training Loop ─────┘
                │
          Evaluation Harness
                │
          Robustness Metrics
```
Key design principle: **every arrow is swappable**.

## 1. Agent Layer

- DQN implementation with:
  - Target network
  - Experience replay
  - Gradient clipping
  - Epsilon-greedy exploration
- Double DQN implementation:
  - Decoupled action selection and evaluation
  - Reduced overestimation bias

Both agents share the same training loop and evaluation interface.

---

## 2. Environment Layer

### CleanEnv
- Deterministic 1D control problem
- Distance-based reward shaping
- Stable convergence verified

### DriftedEnv
Supports:
- Reward scaling
- Reward flipping
- Target drift (true non-stationarity)

Target drift modifies the goal position over time, creating a dynamic MDP where the optimal action changes.

---

## 3. Evaluation & Metrics

Stress testing includes:

- Clean vs Drifted comparison
- Graded drift severity
- Fixed-horizon evaluation
- Deterministic evaluation mode

Metrics:

- Mean reward
- Standard deviation
- Reward variance
- Regret
- Collapse rate

---

# 🔬 Experimental Findings

## Baseline Performance

Both DQN and Double DQN converge reliably in stationary settings:

- Mean reward ≈ -45
- Deterministic trajectories
- Stable loss decay

Double DQN converges slightly faster and exhibits smoother learning curves.

---

## Reward Scaling (Early Attempt)

Scaling reward magnitude did not alter policy behavior.

Insight:

> Changing reward magnitude without altering optimal policy does not test robustness.

---

## Target Drift (True Non-Stationarity)

Gradual target movement introduced meaningful degradation.

### DQN Results

| Drift | Mean Reward |
|-------|------------|
| 0.0   | -45.0 |
| 0.1   | -50.6 |
| 0.2   | -57.6 |
| 0.5   | -95.0 |

Smooth degradation observed.

---

### Double DQN Results

| Drift | Mean Reward |
|-------|------------|
| 0.1   | -50.6 |
| 0.2   | -57.6 |
| 0.5   | -8512.0 |

Under severe drift, Double DQN exhibited catastrophic collapse.

---

# 🎯 Key Insights

1. Double DQN improves learning stability in stationary environments.
2. Mild non-stationarity affects both algorithms similarly.
3. Severe drift can induce sharper collapse in Double DQN.
4. Reducing estimation bias does not guarantee robustness.
5. Robustness requires adaptability, not just accurate value estimates.

---

# 📊 Visualization

Generate degradation plots:

```
python -m evaluation.visualizer
```

Results are saved to:

```
results/
```

Plots include:

- DQN degradation curve
- Double DQN degradation curve

---

# ▶️ Training

Train DQN:

```
python -m training.train --agent dqn
```

Train Double DQN:

```
python -m training.train --agent double_dqn
```

---

# 🧪 Stress Testing

Example:

```
python -m evaluation.stress_test   --checkpoint models/checkpoints/dqn.pt   --target_drift 0.2
```

---

# 🧠 Learning Outcomes

SentinelRL demonstrates:

- Importance of controlled experimentation
- Difference between estimation bias and robustness
- Non-linear collapse under extreme non-stationarity
- How deterministic environments mask variance-based signals
- How architectural confidence can amplify failure modes

---

# 📌 Status
SentinelRL is a controlled reinforcement learning robustness framework showing that improved value estimation does not necessarily imply resilience to environmental drift.
✔ Stable RL baseline  
✔ Double DQN comparison  
✔ Controlled non-stationarity  
✔ Graded degradation experiments  
✔ Visualized robustness curves  

