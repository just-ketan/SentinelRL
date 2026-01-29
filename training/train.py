import numpy as np
from collections import deque
import torch

from environments.clean_env import CleanEnv
from agents.dqn_agent import DQNAgent

def train(num_episodes=500, max_steps=200, log_intervals=20, save_path="models/dqn_checkpoint.pt"):
    env = CleanEnv(max_steps=max_steps)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim=obs_dim, action_dim=action_dim)
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
        
        agent.save(save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()