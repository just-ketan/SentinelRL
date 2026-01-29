import torch
import torch.nn.functional as F
import numpy as np

from agents.base_agent import BaseAgent
from networks.q_network import QNetwork
from replay.replay_buffer import ReplayBuffer

class DQMAgent(BaseAgent):
    def __init__(
            self,
            obs_dim, 
            action_dim, 
            lr=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.5,
            epsilon_decay=0.995
            buffer_capacity=100000,
            batch_size=64,
            target_update_freq=1000,
            device=None,
    ):
        super().__init__(obs_dim, action_dim, device)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.train_steps = 0


    def select_action(self, observation, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs)
            return int(torch.argmax(q_values, dim=1).item())
    

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_obs).max(dim=1)[0]
            targets = rewards + (1.0-dones)*self.gamma*next_q_values
        
        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)

        self._update_target()
        self._decay_epsilon()

        return {
            "loss" : float(loss.item()),
            "epsilon": float(self.epsilon),
        }
    
    def _update_target(self):
        self.train_steps
        if self.train_steps%self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.sate_dict())
            
    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon*self.epsilon_decay)

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_q_net" : self.target_q_net.state_dict(),
            "epsilon" : self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)

