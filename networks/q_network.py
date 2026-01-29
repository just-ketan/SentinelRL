import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwor(nn.Module):
    # simple feed forward Q network
    def __init__(self, obs_dim, action_dim, hidden_dims=(128,128)):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return self.model(x)