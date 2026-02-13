# inherit from DQN and override train_step()

import torch
import torch.nn.functional as F
from agents.dqn_agent import DQNAgent

class DoubleDQNAgent(DQNAgent):
    # reduce overestimation bias by decoupling
    # separate action selection and evaluation step

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        obs = obs[:, :self.obs_dim] # enforce observation contract
        next_obs = next_obs[:, : self.obs_dim]

        # current Q estimate
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # select action using ONLINE network
            next_actions = torch.argmax(self.q_net(next_obs), dim=1)
            # evaluate action using TARGET network
            next_q_values = self.target_q_net(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (1.0-dones)*self.gamma*next_q_values

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self._update_target()
        self._decay_epsilon()

        return { 
            "loss":float(loss.item()),
            "epsilon":float(self.epsilon),
        }