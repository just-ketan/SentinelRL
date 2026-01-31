import numpy as np

def compute_regret(rewards, optimal_reward):
    rewards = np.asarray(rewards)
    regret = optimal_reward-rewards
    return{
        "mean_regret":float(regret.mean()),
        "max_regret":float(regret.max()),
        "min_regret":float(regret.min()),
    }

def collapse_rate(rewards, threshold):
    rewards = np.asarray(rewards)
    return float(np.mean(rewards < threshold))

def reward_variance(rewards):
    rewards = np.asarray(rewards)
    return float(np.var(rewards))

def summarize(rewards, name="env"):
    rewards = np.asarray(rewards)
    return{
        f"{name}_mean":float(rewards.mean()),
        f"{name}_std":float(rewards.std()),
        f"{name}_min":float(rewards.min()),
        f"{name}_max":float(rewards.max()),
    }