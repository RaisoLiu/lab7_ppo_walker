import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM = 128

def initialize_uniformly(layer, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    if isinstance(layer, nn.Linear):
        layer.weight.data.uniform_(-init_w, init_w)
        layer.bias.data.uniform_(-init_w, init_w)
    elif isinstance(layer, nn.Sequential):
        for module in layer:
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-init_w, init_w)
                module.bias.data.uniform_(-init_w, init_w)
            elif isinstance(module, nn.Sequential):
                initialize_uniformly(module, init_w)

def mish(input):
    """Mish激活函數"""
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    """Mish激活函數模組"""
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.model = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            Mish(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            Mish(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            Mish(),
        )
        self.fc_mean = nn.Linear(HIDDEN_DIM, out_dim)
        self.fc_log_std = nn.Linear(HIDDEN_DIM, out_dim)

        initialize_uniformly(self.model)
        initialize_uniformly(self.fc_mean)
        initialize_uniformly(self.fc_log_std)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        X = self.model(state)
        mean = torch.tanh(self.fc_mean(X)) * 1
        log_std = self.fc_log_std(X)
        LOG_STD_MIN = -10 # Or -10, -5, a common choice for stability
        LOG_STD_MAX = 0    # Or 0,  a common choice for stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # 非常重要！
        std = torch.exp(log_std) + 1e-5 # 1e-5 也可以考慮調整，例如 1e-6
        # std = torch.exp(log_std) + 1e-5
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.model = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            Mish(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            Mish(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            Mish(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        initialize_uniformly(self.model)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        value = self.model(state)
        #############################

        return value