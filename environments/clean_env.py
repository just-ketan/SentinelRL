import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environments.base_env import BaseEnv

class CleanEnv(BaseEnv):
    # simple deterministic env used as clean training baseline

    def __init__(self, target_position=10.0, max_steps=200, render_mode=None):
        super().__init__(render_mode)

        self.target_position = target_position
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.state=None
        self.steps=0

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state=np.array([0,0], dtype=np.float32)
        self.steps=0
        return self.state, {}
    
    def step(self, action):
        self.steps += 1
        move = -1.0 if action == 0 else 1.0
        self.state[0] += move

        dist = abs(self.target_position - self.state[0])
        reward = -dist

        terminated = dist < 0.5
        truncated = self.steps >= self.max_steps

        return self.state, reward, terminated, truncated, {}
    
    def inject_drift(self, **kwargs):
        # clean environment does not support drift
        # only to satisfy BaseEnv contract
        pass