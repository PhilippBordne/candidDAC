import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, n_obs: int, n_act: int) -> None:
        """Q-network that maps from observations to Q-values for each action.
        Parameters
        ----------
        n_obs : int
            Dimensionality of the observation space.
        n_act : int
            Dimensionality of the action space.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_obs, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_act),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)
