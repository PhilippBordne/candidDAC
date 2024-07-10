"""
Implements utility functions to pre- and post-process gymnasium actions and observations
according to their respective spaces.
"""

from gymnasium.spaces import MultiDiscrete
import numpy as np
import torch
from dacbench.benchmarks import SigmoidBenchmark


def get_num_states_from_multidiscrete(space: MultiDiscrete) -> int:
    """Get the number of states (or actions) from a MultiDiscrete space.

    Parameters
    ----------
    space : MultiDiscrete
        MultiDiscrete space.

    Returns
    -------
    int
        Number of states.
    """
    return np.prod(space.nvec)


def flatten_multidiscrete_actions(action: torch.Tensor, space: MultiDiscrete) -> torch.Tensor:
    """ Flatten a single or Batch of MultiDiscrete actions into an integer or Batch of integer actions.
    Parameters
    ----------
    action : torch.Tensor
        A single action with shape (1, action_space_dim) or a Batch of actions with shape
        (batch_size, action_space_dim).
    space : MultiDiscrete
        MultiDiscrete action into which the action
    """

    if action.shape[0] == 1:
        return torch.tensor(np.ravel_multi_index(action.numpy(), space.nvec))
    else:
        return torch.tensor(np.ravel_multi_index(action.numpy().transpose(1, 0), space.nvec)).reshape(-1, 1)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    Implementation taken from cleanrl and copied to remove dependency on the library.
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def get_leader_follower_config(dim: int = 2):
    """
    Sets up config to run the sigmoid benchmark using leader-follower policies, but using the same instances as in the
    DAC framework paper.
    """
    # this is a hack. Call get_benchmark(dim) overwrites the config of the benchmark object. Which we can then modify to
    # specify the leader follower multi-agent setting.
    benchmark = SigmoidBenchmark()
    benchmark.get_benchmark(dim)  # config of benchmark now specifies the benchmark with given dim
    # retrieve the config and adapt it for leader follower
    config = benchmark.config
    config["leader_follower"] = True
    config["multi_agent"] = True
    return config
