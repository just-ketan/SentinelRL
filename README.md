# SentinelRL
License: MIT

**SentinelRL** is an end-to-end reinforcement learning system designed to **probe, stress-test, and audit RL policies under specification and reward drift**. Unlike benchmark-focused RL projects, SentinelRL treats policy failure as a first-class concern and evaluates how agents behave when assumptions about the environment change after training.

This project is intentionally built as a **recruiter-probing system**: every module, experiment, and metric exists to surface design decisions, trade-offs, and failure modes during technical interviews.

---

## 1. Problem Statement

Most reinforcement learning agents are evaluated only in the environment they are trained on. In real systems, however:

* Reward functions drift
* Transition dynamics change
* Observations become noisy or delayed
* Previously optimal policies become brittle

**SentinelRL answers a simple but hard question:**

> *Does this policy still behave sensibly when the rules change?*

---

## 2. Core Idea

SentinelRL separates **training**, **environment specification**, and **evaluation** into explicit layers so that policies can be:

* Trained in clean environments
* Exposed to controlled drift
* Stress-tested under adversarial conditions

The system focuses on **robustness, stability, and degradation behavior**, not just cumulative reward.

---

## 3. System Architecture (High Level)

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

---

## 4. Supported Algorithms

Currently implemented and extensible:

* DQN
* Double DQN
* Dueling DQN
* Prioritized Experience Replay

Each agent conforms to a shared interface, enabling controlled comparisons under identical drift scenarios.

---

## 5. Environment Design

SentinelRL environments are Gym-compatible but intentionally modular:

* `clean_env`
  Nominal reward and transition dynamics

* `drifted_env`
  Gradual or abrupt reward/specification changes

* `adversarial_env`
  Worst-case transitions and deceptive rewards

Environment wrappers allow injecting:

* Reward noise
* Observation masking
* Delayed rewards

---

## 6. Training Pipeline

Training follows a disciplined RL workflow:

1. Initialize clean environment
2. Train agent with replay and target networks
3. Periodically checkpoint policy
4. Optionally apply curriculum learning
5. Freeze policy for evaluation

Training code is intentionally **boring and explicit** to make reasoning about failures easier.

---

## 7. Evaluation Philosophy

Reward alone is not enough.

SentinelRL tracks:

* Average episode reward
* Regret under drift
* Policy collapse frequency
* Sensitivity to reward perturbations
* Performance variance across seeds

Evaluation scripts never modify training artifacts.

---

## 8. Example Experiments

Located in `experiments/`:

* `exp_clean_vs_drift.py`
  Compare performance before and after specification drift

* `exp_reward_hacking.py`
  Detect policies exploiting unintended reward signals

* `exp_generalization.py`
  Test robustness across unseen environment parameters

Each experiment is reproducible and isolated.

---

## 9. Folder Structure Overview

```
sentinelrl/
├── agents/          # RL agents (DQN variants)
├── environments/    # Clean, drifted, adversarial envs
├── networks/        # Q-networks and variants
├── replay/          # Replay buffers
├── training/        # Training and curriculum logic
├── evaluation/      # Stress tests and metrics
├── experiments/     # Reproducible experiment scripts
├── configs/         # YAML-based configuration
├── models/          # Saved checkpoints
└── scripts/         # Run helpers
```

---

## 10. Reproducibility

All experiments:

* Are seed-controlled
* Use explicit configuration files
* Log metrics in machine-readable formats

Shell scripts are provided for one-command reproduction.

---

## 11. Intended Audience

This project is designed for:

* RL practitioners
* Systems-oriented ML engineers
* Interviewers probing RL depth beyond benchmarks

It is **not** optimized for leaderboard scores.

---

## 12. Talking Points for Interviews

Be prepared to discuss:

* Why reward drift matters
* Trade-offs between exploration and robustness
* Failure modes of value-based methods
* Why evaluation should be adversarial

SentinelRL exists to make these discussions concrete.

---

## 13. Status

This is an active research-style engineering project. New environments, agents, and metrics are intentionally easy to add.

Contributions and extensions are welcome.
