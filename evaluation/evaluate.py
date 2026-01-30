import numpy as np
import torch
from environments.clean_env import CleanEnv
from agents.dqn_agent import DQNAgent

def evaluate(checkpoint_path, num_episodes=50, max_steps=200,):
    env = CleanEnv(max_steps=max_steps)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim=obs_dim, action_dim=action_dim,)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0 #disable exploration

    rewards = []
    print("Evaluating trained policy (no exploration)")

    for ep in range(1, max_steps+1):
        obs, _ = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(obs, explore=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if truncated or terminated:
                break
        
        rewards.append(episode_reward)
        print(f"Episode {ep:3d} | Reward: {episode_reward:8.2f}")
    
    rewards = np.array(reward)
    print("Summary")
    print("-"*60)
    print(f"Mean Reward : {rewards.mean():.2f}")
    print(f"Std Reward : {rewards.std():.2f}")
    print(f"Min Reward : {rewards.min():.2f}")
    print(f"Max Reward : {rewards.max():.2f}")

    return rewards

if __name__ == "__main__":
    evaluate("models/dqn_checkpoint.pt")