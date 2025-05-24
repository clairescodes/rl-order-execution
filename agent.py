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
        super().__init__() # initialize nn.Module machinery 
        # compute input size by flattening window × feature_dim
        in_size = obs_shape[0] * obs_shape[1]
        # ACTOR NETWORK 
        # MLP mapping flattened state to raw action mean
        # Hidden layers: sizes in hidden_sizes, final layer size=
        # activation=ReLU for hidden layers
        self.actor = MLP(in_size, action_range[0], hidden_sizes + (1,), activation=nn.ReLU)
        # CRITIC NETWORK 
        # MLP mapping flattened state to state value scalar
        # Hidden layers: sizes in hidden_sizes, final layer size=1
        self.critic = MLP(in_size, 1, hidden_sizes, activation=nn.ReLU)
        # Store action bounds as non-trainable buffers for rescaling
        self.register_buffer("act_min", torch.tensor(action_range[0], dtype=torch.float32))
        self.register_buffer("act_max", torch.tensor(action_range[1], dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        # obs: (batch_size, window, feature_dim)
        # flatten temporal & feature dims into a single vector per batch item
        batch = obs.view(obs.size(0), -1) # (batch_size, window * feature_dim)
        # Actor: compute raw action mean -> (batch_size, 1)
        mu = self.actor(batch) 
        mu = torch.tanh(mu)  # squash mean into [-1,1]

        # Rescale to any arbitrary [act_min, act_max] interval
        mu = (mu + 1)/ 2 * (self.act_max - self.act_min) + self.act_min 
        value = self.critic(batch).squeeze(-1) # -> (batch_size, 1) -> (batch_size,)
        return mu, value # action mean & value estimate 

def build_policy(
    net: TradingActorCritic,
    optimizer: torch.optim.Optimizer,
    discount_factor: float = 0.99,
    gae_lambda: float = 0.95,
    eps_clip: float = 0.2,
    value_coef: float = 0.5,
    ent_coef: float = 0.0,
):
    """
    Create a PPO policy wrapping the actor-critic network.
    """
    return PPOPolicy(
        actor_critic=net,                           # network with forward(obs) -> (mu, value)
        optim=optimizer,                            # pyTorch optimizer for updating net's params
        dist_fn=lambda x: (x, torch.ones_like(x)),  # dist_fn maps network output `mu` -> distribution parameters (loc, scale). used fixed Normal for cont action
        discount_factor=discount_factor,            # how much to discount future rewards
        gae_lambda=gae_lambda,                      # mixing factor for GAE (bias–variance tradeoff)
        max_grad_norm=0.5,                          # Clip gradients with global norm > 0.5
        eps_clip=eps_clip,                          # PPO clipping epsilon for policy updates
        value_weight=value_coef,                    # Weight for value-function loss term
        ent_weight=ent_coef,                        # Weight for entropy bonus term
        reward_normalization=False,                 # Don’t normalize rewards internally
    )