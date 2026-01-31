import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.dqn_agent import DQNAgent
from environments.clean_env import CleanEnv
from environments.drifted_env import DriftedEnv
from evaluation.metrics import (compute_regret, collapse_rate, reward_variance, summarize,)

def run_policy(env, agent, num_episodes=50, max_steps=200):
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
    return np.array(rewards)

def stress_test(checkpoint_path="models/dqn_checkpoint.pt", num_episodes=50, collapse_threshold=200.0, optimal_reward=-45.0,):
    # load agent
    clean_env = CleanEnv(terminate_on_goal=True)
    obs_dim = clean_env.observation_space.shape[0]
    action_dim = clean_env.action_space.n

    agent = DQNAgent(obs_dim=obs_dim, action_dim=action_dim)
    agent.load(checkpoint_path)
    agent.epsilon=0.0

    print("SentinelRL STRESS TEST (metrics enabled)")

    clean_rewards = run_policy(clean_env, agent, num_episodes=num_episodes)
    
    drifted_env = DriftedEnv(reward_flip=True, drift_start_step=1)
    drifted_env.terminate_on_goal=False
    drifted_rewards = run_policy(drifted_env, agent, num_episodes=num_episodes)

    # print metrics
    print("CLEAN ENV METRICS")
    for k,v in summarize(clean_rewards, name="clean").items():
        print(f"{k:>15}:{v:8.2f}")
    print(f"{'variance':>15}:{reward_variance(clean_rewards):8.2f}")

    print("DRIFTED ENV METRICS")
    for k,v in summarize(drifted_rewards, name="drifted").items():
        print(f" {k:>15}:{v:8.2f}")
    print(f" {'variance':>15}:{reward_variance(drifted_rewards):8.2f}")

    print("Robustness Indicators")
    print(f" {'mean_regret':>15}: {compute_regret(drifted_rewards, optimal_reward)['mean_regret']:8.2f}")
    print(f" {'collapse_rate':>15}: {collapse_rate(drifted_rewards, collapse_threshold):8.2f}")

if __name__ == "__main__":
    stress_test()