import numpy as np
from numpy import ndarray
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from wandb.wandb_run import Run

from candid_dac.algorithms.base_dqn import DQN
from candid_dac.policies import AtomicPolicy, FactorizedPolicy
from omegaconf import DictConfig


class FactorizedDQN(DQN):
    """
    Implementation of DQN that factorizes a K-dimensional action space into K 1-D action spaces and learns a atomic
    policy for each action dimension. It uses a FactorizedPolicy to sample and execute actions in the environment.
    And operates on the FactorizedPolicy's subpolicies to update their Q-functions.
    """
    def __init__(
            self, env: Env, hp_config: DictConfig, algorithm_choices: DictConfig, steps_tot: int, seed: int,
            wandb_run: Run = None, **kwargs) -> None:
        """
        Version of DQN that factorizes a K-dimensional action space into K 1-D action spaces and learns a atomic policy
        for each action dimension.
        Parameters
        ----------
        use_single_buffer : bool
            If true, all agents use the same replay buffer. Otherwise each agent has its own replay buffer. Especially
            this decides wether the same transitions will be sampled for all agents upon policy update.
        """
        super().__init__(env, hp_config, steps_tot, seed, wandb_run=wandb_run, **kwargs)

        self.autorecursive = algorithm_choices["autorecursive"]

        if algorithm_choices.use_single_buffer:
            self.use_single_buffer = True
            self.single_batch = None
        else:
            self.use_single_buffer = False

        self.policy = FactorizedPolicy(self.dim_obs, self.dim_act, env.action_space.nvec,
                                       autorecursive=self.autorecursive, lr=hp_config['lr'],
                                       device=self.device)

        self._setup_replay_buffers()

    def sample_greedy_action(self, obs: np.ndarray = None) -> ndarray:
        obs = self.curr_obs if obs is None else obs
        return self.policy.get_action(obs).numpy().astype(np.int32)

    def store_to_buffer(self) -> None:

        for i in np.arange(self.dim_act):
            if self.autorecursive:
                # if the Q-function does not model the sub-MDP then the transition is the next_obs and the actions
                # selected by the leaders under the next observation and the full reward is received
                obs = np.concatenate([self.curr_obs, self.curr_action[:i]])
                next_act = self.policy.get_action(self.next_obs)
                next_obs = np.concatenate([self.next_obs, next_act[:i]])
                reward = self.curr_reward
                done = self.episode_done
            else:
                # if no autorecursiveness in the policy, the transition is from the current to the next observation
                # without any actions appended to the state. The reward is the full reward.
                obs = self.curr_obs
                next_obs = self.next_obs
                reward = self.curr_reward
                done = self.episode_done
            self.replay_buffers[i].add(obs, next_obs, self.curr_action[i], reward, done, {})

    def update_q(self) -> None:
        """
        Updates the Q-functions of all agents currently to be trained.
        """

        # if using a single buffer, sample the same batch for all agents and store it as a state of the algorithm
        # it might already have been sampled by a specialization of FDQN
        if self.use_single_buffer and self.single_batch is None:
            self.single_batch = self.single_buffer.sample(self.batch_size)

        for i in np.arange(self.dim_act):
            # if not using the same buffer for all agents, sample a batch per agent
            self._update_policy(i)

        # reset the batch to None, so that it is sampled again in the next update step
        self.single_batch = None

    # this subfunction allows to adapt the policy update itself for SDQN
    def _update_policy(self, i: int) -> None:
        """
        Implements the update step of a single agent's Q-function.

        Parameters
        ----------
        i : int
            Index of the agent to update.
        """
        batch = self.single_batch if self.use_single_buffer else self.replay_buffers[i].sample(self.batch_size)
        # compute target
        # here we only need to select the appropriate target policy
        # appending the action to the successor state in case of model_sub_mdp is handled when storing to buffer
        train_policy = self.policy.subpolicies[i]
        target_policy = self.policy.subpolicies[i]

        self._policy_optimization_step(train_policy, target_policy, batch.observations, batch.next_observations,
                                       batch.actions, batch.rewards, batch.dones)

    def update_target(self) -> None:
        for i in range(self.dim_act):
            for target_param, param in zip(self.policy.subpolicies[i].target.parameters(),
                                           self.policy.subpolicies[i].q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _policy_optimization_step(self, train_policy: AtomicPolicy, target_policy: AtomicPolicy,
                                  observations: torch.Tensor, next_observations: torch.Tensor,
                                  actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
                                  gamma: float = None, update_towards_q: bool = False) -> None:
        """
        Computes td-error, gradients and performs gradient step given the batch and the policies to update and target.
        Parameters
        ----------
        gamma : float
            Discount factor, might be adapted by specialization. E.g. when targeting against successor policy the
            discount factor is 1.
        update_towards_q : bool
            If true the policy is updated towards the Q-function instead of the target function.
        """

        gamma = self.gamma if gamma is None else gamma
        with torch.no_grad():
            if update_towards_q:
                argmax = target_policy.target(next_observations).argmax(axis=-1)
                q_next = target_policy.q_network(next_observations)
            else:
                argmax = target_policy.q_network(next_observations).argmax(axis=-1)
                q_next = target_policy.target(next_observations)
            q_next = q_next.gather(dim=-1, index=argmax.view(-1, 1))
            target = rewards.flatten() + gamma * q_next.flatten() * (1 - dones.flatten())

        # for the observations and next observations we must also reduce them in the case of modelling the sub-MDP
        q = train_policy.q_network(observations).gather(dim=-1, index=actions.view(-1, 1))
        loss = train_policy.loss(q.flatten(), target)

        train_policy.optimizer.zero_grad()
        loss.backward()
        train_policy.optimizer.step()

    def _setup_replay_buffers(self) -> None:

        if self.use_single_buffer:
            self.single_buffer = ReplayBuffer(self.buffer_size, self.env.observation_space, self.env.action_space,
                                              handle_timeout_termination=False, device=self.device)

        else:
            self.replay_buffers = []
            for i in range(self.dim_act):
                if self.autorecursive:
                    low = np.concatenate([self.env.observation_space.low, np.zeros(i)])
                    high = np.concatenate([self.env.observation_space.high, self.env.action_space.nvec[:i]])
                else:
                    low = self.env.observation_space.low
                    high = self.env.observation_space.high
                obs_space = Box(low=low, high=high)
                act_space = Discrete(self.env.action_space.nvec[i])
                buffer = ReplayBuffer(self.buffer_size, obs_space, act_space, handle_timeout_termination=False,
                                      device=self.device)
                self.replay_buffers.append(buffer)

    def _init_state_values_to_wandb(self) -> None:
        """
        Logs the current state value for all subpolicies to wandb.
        """

        # only log if the episodes completed are a multiple of the wandb frequency
        if self.episodes_completed % self.wandb_freq != 0:
            return

        state_values = self.policy.get_state_values(self.curr_obs)

        for i, val in enumerate(state_values):
            self.wandb_run.log({f"V_init_policy_{i}": val}, step=self.steps_taken)
