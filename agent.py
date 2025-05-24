import torch
import torch.nn as nn
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import MLP

class TradingActorCritic(nn.Module):
    """
    A simple MLP actor-critic network for continuous actions.
    Observation: shape (window, feature_dim)
    Action: scalar in [-1,1]
    """
    def __init__(self, obs_shape, action_range=(-1, 1), hidden_sizes=(128, 128)):
        super().__init__()
        in_size = obs_shape[0] * obs_shape[1]
        # actor network
        self.actor = MLP(in_size, action_range[0], hidden_sizes + (1,), activation=nn.ReLU)
        # critic network
        self.critic = MLP(in_size, 1, hidden_sizes, activation=nn.ReLU)
        self.register_buffer("act_min", torch.tensor(action_range[0], dtype=torch.float32))
        self.register_buffer("act_max", torch.tensor(action_range[1], dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        # flatten batch of windows
        batch = obs.view(obs.size(0), -1)
        mu = self.actor(batch)
        mu = torch.tanh(mu)  # bound to [-1,1]
        mu = (mu + 1) / 2 * (self.act_max - self.act_min) + self.act_min
        value = self.critic(batch).squeeze(-1)
        return mu, value
