from abc import ABC, abstractmethod
import gymnasium as gym

class BaseEnv(gym.Env, ABC):
    # abstract base class for all sentinelRL environments

    metadata = {"render_modes: ": ["human", None]}

    def __init__(self, render_mode = None):
        super().__init__()
        self.render_mode = render_mode
    
    @abstractmethod
    def reset(self, *, seed=None, options=None):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def inject_drift(self, **kwargs):
        # apply specs or reward drift, called explicitly during eval or curriculum training
        pass