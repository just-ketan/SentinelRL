from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    # base interface for all RL agents. 
    # enforces separation among policy logic, training logic and persistence

    def __init__(self, obs_dim, action_dim, device=None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def select_action(self, observation, explore=True):
        #select action given observation
        pass

    @abstractmethod
    def train_step(self):
        #perform one optimization step using stored experience
        pass

    @abstractmethod
    def store_transition(self, obs, action, reward, next_obs, done):
        # store transition in replay memory
        pass

    @abstractmethod
    def save(self, path):
        # save agent parameters to disk
        pass

    @abstractmethod
    def load(self, path):
        # load agent parameter from disk
        pass