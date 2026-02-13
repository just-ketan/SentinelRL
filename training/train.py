import numpy as np
from collections import deque
import torch
import os
import sys
import argparse
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from environments.clean_env import CleanEnv
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent

def build_agent(agent_type, obs_dim, action_dim):
    if agent_type == "dqn":
        return DQNAgent(obs_dim=obs_dim, action_dim=action_dim)
    elif agent_type == "double_dqn":
        return DoubleDQNAgent(obs_dim=obs_dim, action_dim=action_dim)
    else:
        raise ValueError(f"Unknown agent type : {agent_type}")

def train(agent_type="dqn", num_episodes=500, max_steps=200, log_intervals=20, save_path="models/dqn_checkpoint.pt"):
    env = CleanEnv(max_steps=max_steps)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = build_agent(agent_type=agent_type, obs_dim=obs_dim, action_dim=action_dim)
    reward_window = deque(maxlen=100)

    print("STARTING SentinelRL TRAINING SEQUENCE......")

    for episode in range(1, num_episodes+1):
        obs, _ = env.reset()
        episode_reward=0.0

        for _ in range(max_steps):
            action = agent.select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, terminated)
            train_info = agent.train_step()

            obs = next_obs
            episode_reward+=reward

            if terminated or truncated:
                break
        
        reward_window.append(episode_reward)

        if episode%log_intervals == 0:
            avg_reward = np.mean(reward_window)
            eps = agent.epsilon
            loss = train_info["loss"] if train_info else None

            print(
                f"Episode {episode:4d} | "
                f"AvgReward {avg_reward:8.2f} | "
                f"Epsilon {eps:6.3f} | "
                f"Loss {loss if loss is not None else 'n/a'}"
            )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"{agent_type} saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "double_dqn"], help="Agent type to train")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    train(agent_type=args.agent, num_episodes=args.episodes,)
