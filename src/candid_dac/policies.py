import numpy as np
import torch
from torch.nn import functional as F
from torch import Tensor
from candid_dac.models import QNetwork
from typing import Callable, Type, Union


class AtomicPolicy():
    def __init__(
            self, dim_obs: int, n_act: int, optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
            lr: float = 0.001, loss: Callable[[Tensor, Tensor], Tensor] = F.mse_loss,
            model: Type[torch.nn.Module] = None, device: str = 'cpu', q_ckpt: str = None,
            target_ckpt: str = None) -> None:
        """
        Policy that maps from N-D state to 1-D action with M possible action choices using function approximation.

        Parameters
        ----------
        n_obs : int
            Number of observations in the state space.
        n_act : int
            Number of possible actions.
        model : torch.nn.Module, optional
            Function approximator to use.
        cpkt : str, optional
            Path to a checkpoint file to load the q_network and target from.
        """
        self.N = dim_obs
        self.n_act = n_act
        self.device = device

        if model is None:
            self.q_network = QNetwork(dim_obs, n_act)
            self.target = QNetwork(dim_obs, n_act)
            self.target.load_state_dict(self.q_network.state_dict())
        else:
            self.q_network = model(dim_obs, n_act)
            self.target = model(dim_obs, n_act)

        if q_ckpt is not None:
            self.q_network.load_state_dict(torch.load(q_ckpt, map_location=torch.device('cpu')))

        if target_ckpt is not None:
            self.target.load_state_dict(torch.load(target_ckpt, map_location=torch.device('cpu')))
        else:
            self.target.load_state_dict(self.q_network.state_dict())

        self.q_network.to(self.device)
        self.target.to(self.device)

        self.optimizer = optimizer(self.q_network.parameters(), lr=lr)
        self.loss = loss

    def __call__(self, state: Tensor) -> Tensor:
        """Returns the discrete action choice given the state on the cpu.

        Parameters
        ----------
        state : Tensor
            State of the environment.

        Returns
        -------
        Tensor
            Discrete action choice.
        """
        # make sure tensor dtype is float32 as we are interfacing with numpy which uses float64 as default
        state = state.to(self.device).float()
        return self.q_network(state).argmax(axis=-1).cpu()


class FactorizedPolicy():
    def __init__(
            self, dim_obs: int, dim_act: int, n_act: Union[int, list], autorecursive: bool = False,
            optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam, lr: float = 0.001,
            loss: Callable[[Tensor, Tensor], Tensor] = F.mse_loss, model: Type[torch.nn.Module] = None,
            device: str = 'cpu', path_to_q_ckpts: list[str] = None, path_to_target_ckpts: list[str] = None
            ) -> None:
        """
        Policy that maps from N-D state to K-D action space by factorizing the action space into K atomic policies.

        Parameters
        ----------
        dim_obs : int
            Dimension of the observation space.
        dim_act : int
            Dimension of the action space.
        n_act : int or list
            Number of possible actions for per dimension. If int, then all dimensions have the same number of actions.
        autorecursive : bool, optional
            If true actions are selected in sequence and prior action are appended to the state of the next policy.
        model : torch.nn.Module, optional
            Network architecture to use for the atomic policies' q-networks.
        path_to_q_ckpts : list[str], optional
            List of paths to the atomic policies' q-network checkpoints. As this is a factorized policy, number of
            paths must match dim_act.
        path_to_target_ckpts : list[str], optional
            List of paths to the atomic policies' target network checkpoints. As this is a factorized policy, number of
            paths must match dim_act.
        """

        self.N = dim_obs
        self.K = dim_act

        if isinstance(n_act, int):
            n_act = [n_act] * dim_act
        self.n_act = n_act

        if len(self.n_act) != self.K:
            raise ValueError("When providing a list for the number of actions per action dimension its number of "
                             "entries must match the dimension of the action space.")

        self.autorecursive = autorecursive

        if self.autorecursive:
            self.get_action = self._autorecursive_action
        else:
            self.get_action = self._independent_action

        self.device = device
        # Initialize the atomic policies
        # TODO: to allow for more flexibility consider encapsulating this in a function
        self.subpolicies: list[AtomicPolicy] = []
        for i in range(self.K):
            dim_obs = (self.N + i) if self.autorecursive else self.N
            q_ckpt = path_to_q_ckpts[i] if path_to_q_ckpts is not None else None
            target_ckpt = path_to_target_ckpts[i] if path_to_target_ckpts is not None else None
            self.subpolicies.append(
                AtomicPolicy(dim_obs, self.n_act[i], optimizer, lr, loss, model, self.device, q_ckpt, target_ckpt)
            )

    def get_state_values(self, state: np.ndarray) -> np.ndarray:
        """Returns the state values of all atomic policies given the state. If the policy is autorecursive
        state values are given for the greedy action sequence.
        Parameters
        ----------
        state : np.ndarray
            Single state of the environment. Shape (dim_obs, ).
        """
        state = torch.tensor(state).to(self.device)
        state_values = []
        for i in range(self.K):
            state_value = self.subpolicies[i].q_network(state.float()).max()
            state_values.append(state_value.item())
            if self.autorecursive:
                state = torch.cat([state, self.subpolicies[i](state).view(1).to(self.device)])
        return np.array(state_values)

    def _independent_action(self, state: np.ndarray) -> Tensor:
        """Returns the discrete action choice given the state on the cpu.

        Parameters
        ----------
        state : Tensor
            State of the environment.

        Returns
        -------
        Tensor
            Discrete action choice.
        """
        action = []
        for i in range(self.K):
            action.append(self.subpolicies[i](torch.tensor(state).to(self.device)))
        return torch.tensor(action).cpu()

    def _autorecursive_action(self, state: np.ndarray) -> Tensor:
        """Returns the discrete action choice given the state on the cpu.

        Parameters
        ----------
        state : Tensor
            State of the environment.

        Returns
        -------
        Tensor
            Discrete action choice.
        """
        state = torch.tensor(state).to(self.device)
        action = torch.Tensor([]).to(self.device)
        for i in range(self.K):
            atom = torch.tensor([self.subpolicies[i](torch.cat([state, action]))]).to(self.device)
            action = torch.cat([action, atom])
            # state = torch.cat([state, action])
        return action.cpu()
