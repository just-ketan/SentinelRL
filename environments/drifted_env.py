import numpy as np
from gymnasium import spaces
from environments.clean_env import CleanEnv

class DriftedEnv(CleanEnv):
    # gradual target drigt
    # rewward scling / flipping
    def __init__(
            self,
            target_position=10.0,
            max_steps=200,
            render_mode=None,
            drift_start_step=100,
            reward_flip=False,
            reward_scale=1.0,
            target_drift_per_step=0.0,
    ):
        super().__init__(target_position=target_position, max_steps=max_steps, render_mode=render_mode)
        # drift config
        self.initial_target = target_position
        self.drift_start_step = drift_start_step
        self.reward_flip = reward_flip
        self.reward_scale = reward_scale
        self.target_drift_per_step = target_drift_per_step

        self.global_step=0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.global_step=0
        self.target_position = self.initial_target
        return obs, info
    
    def step(self, action):
        self.global_step += 1
        #apply drift
        if self.global_step >= self.drift_start_step:
            self.target_position += self.target_drift_per_step
        
        #base transition
        obs, reward, terminated, truncated, info = super().step(action)

        #apply reward distortion
        if self.global_step >= self.drift_start_step:
            if self.reward_flip:
                reward -= reward
            reward *= self.reward_scale
        info["current_target"] = self.target_position
        return obs, reward, terminated, truncated, info

    def inject_drift(
            self,
            *,
            reward_flip=None,
            reward_scale=None,
            target_drift_per_step=None,
            drift_start_step=None,
    ):
        # dynamically modify drift parameters at Runtime
        if reward_flip is not None:
            self.reward_flip = reward_flip
        if reward_scale is not None:
            self.reward_scale = reward_scale
        if target_drift_per_step is not None:
            self.target_drift_per_step = target_drift_per_step
        if drift_start_step is not None:
            self.drift_start_step = drift_start_step