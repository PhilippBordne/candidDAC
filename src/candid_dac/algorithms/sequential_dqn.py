import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from wandb.wandb_run import Run
from typing import List
from omegaconf import DictConfig, open_dict

from candid_dac.algorithms.factorized_dqn import FactorizedDQN
from candid_dac.models import QNetwork


class SDQN(FactorizedDQN):
    def __init__(self, env: Env, hp_config: DictConfig, algorithm_choices: DictConfig, steps_tot: int, seed: int,
                 wandb_run: Run = None, **kwargs) -> None:
        """
        Specialized version of FactorizedDQN in which state transitions are the state observed by the agent and the
        next state are the state appended with the action taken by the agent and the value of the successor state is
        determined by the target network of the next agent.

        Parameters
        ----------
        update_towards_q : bool
            If true, all agents but the last agent are updated towards the Q-function of their successor policy.
        """

        self.use_upper_q = algorithm_choices["use_upper_q"]
        self.update_towards_q = algorithm_choices["update_towards_q"]
        with open_dict(algorithm_choices):
            algorithm_choices['autorecursive'] = True  # this is always true for SDQN
        super().__init__(env=env, hp_config=hp_config, algorithm_choices=algorithm_choices, steps_tot=steps_tot,
                         seed=seed, wandb_run=wandb_run, **kwargs)

        if self.use_upper_q:
            self.upper_q = QNetwork(self.dim_obs + len(self.env.action_space.nvec), 1).to(self.device)
            self.upper_q_optim = torch.optim.Adam(self.upper_q.parameters(), hp_config['lr'])

    def update_q(self) -> None:
        # first update the upper Q-function
        if self.use_upper_q:
            # if we use a single buffer fix the batch now
            if self.use_single_buffer:
                self.single_batch = self.single_buffer.sample(self.batch_size)
            self._update_upper_q()
        # then update the lower Q-functions of the actual policy we are training
        super().update_q()

    def _update_policy(self, i: int) -> None:
        # if we use an upper Q-function we need to update the last policy towards the upper Q-function
        if self.use_upper_q and i == self.dim_act - 1:
            self._update_subpolicy_towards_upper()
            return

        batch = self.single_batch if self.use_single_buffer else self.replay_buffers[i].sample(self.batch_size)

        # build observations and transitions depending on the agents position in the sequence from a batch of
        # transitions in the original environment
        train_policy = self.policy.subpolicies[i]
        observations = torch.hstack([batch.observations, batch.actions[:, :i]])
        actions = batch.actions[:, i]
        if i == self.dim_act - 1:
            next_observations = batch.next_observations
            rewards = batch.rewards
            dones = batch.dones
            # otherwise we will update towards the target of the first agent in the next state
            target_policy = self.policy.subpolicies[0]
            gamma = self.gamma  # last agent discounts towards first agent in next state
            update_towards_q = False  # for the next state always update towards target
        else:
            next_observations = torch.hstack([batch.observations, batch.actions[:, :i+1]])
            rewards = torch.zeros_like(batch.rewards)
            dones = torch.zeros_like(batch.dones)
            target_policy = self.policy.subpolicies[i+1]
            gamma = 1
            update_towards_q = self.update_towards_q

        self._policy_optimization_step(train_policy, target_policy, observations, next_observations, actions,
                                       rewards, dones, gamma=gamma, update_towards_q=update_towards_q)

    def _update_upper_q(self) -> None:
        # compute target
        # here we only need to select the appropriate target policy
        # appending the action to the successor state in case of model_sub_mdp is handled when storing to buffer

        if self.use_single_buffer:
            raise NotImplementedError("Single buffer not yet implemented for upper Q-function")

        batch = self.single_batch if self.use_single_buffer else self.upper_replay_buffer.sample(self.batch_size)

        self.upper_q_optim.zero_grad()

        with torch.no_grad():
            target = self.upper_q(batch.next_observations)
            target = batch.rewards + self.gamma * target * (1 - batch.dones)
            target = target.flatten()

        pred = self.upper_q(batch.observations)
        pred = pred.flatten()
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()

        self.upper_q_optim.step()

    def _update_subpolicy_towards_upper(self) -> None:
        """
        Special update step in case we use a upper Q-function. In this case the value of the last Q-function for the
        full action vector should be equal to the value of the upper Q-function for this action vector.
        """
        batch = self.single_batch if self.use_single_buffer else self.replay_buffers[-1].sample(self.batch_size)

        train_policy = self.policy.subpolicies[-1]

        # convert the multi-dimensional actions in the batch of shape (batch_size, action_dim )to an index of the
        # flattened action space of the upper Q-function.
        # flat_act = np.ravel_multi_index(batch.actions.numpy().T, self.env.action_space.nvec)

        upper_obs = torch.hstack([batch.observations, batch.actions])

        # compute the target value for the last Q-function
        with torch.no_grad():
            target = self.upper_q(upper_obs)
            target = target.flatten()

        # compute the current value of the last Q-function
        agent_observations = torch.hstack([batch.observations, batch.actions[:, :-1]])
        agent_actions = batch.actions[:, -1]
        q = train_policy.q_network(agent_observations).gather(dim=-1, index=agent_actions.view(-1, 1)).flatten()

        # compute the loss and perform the optimization step
        train_policy.optimizer.zero_grad()
        loss = train_policy.loss(q, target)
        loss.backward()
        train_policy.optimizer.step()

    def store_to_buffer(self) -> None:
        if self.use_single_buffer:
            self.single_buffer.add(self.curr_obs, self.next_obs, self.curr_action, self.curr_reward, self.episode_done,
                                   {})
        else:
            if self.use_upper_q:
                next_act = self.policy.get_action(self.next_obs)
                curr_obs = np.concatenate([self.curr_obs, self.curr_action])
                next_obs = np.concatenate([self.next_obs, next_act])
                self.upper_replay_buffer.add(curr_obs, next_obs, np.array([0]), self.curr_reward, self.episode_done, {})

            for i in np.arange(self.dim_act):
                self.replay_buffers[i].add(self.curr_obs, self.next_obs, self.curr_action, self.curr_reward,
                                           self.episode_done, {})

    def _setup_replay_buffers(self) -> None:
        if self.use_single_buffer:
            self.single_buffer = ReplayBuffer(self.buffer_size, self.env.observation_space, self.env.action_space,
                                              handle_timeout_termination=False, device=self.device)
        else:
            if self.use_upper_q:
                # setup the replay buffers for the upper Q-function
                obs_low = np.concatenate([self.env.observation_space.low, np.zeros_like(self.env.action_space.nvec)])
                obs_high = np.concatenate([self.env.observation_space.high, self.env.action_space.nvec])
                obs_space = Box(low=obs_low, high=obs_high)
                self.upper_replay_buffer = ReplayBuffer(self.buffer_size,
                                                        obs_space, Discrete(1),
                                                        handle_timeout_termination=False, device=self.device)

            # setup the replay buffers for the lower Q-functions
            # building agent specific observations will be taken over by the q_update function, so the buffers are
            # identical and only serve purpose to store transitions per agent in case of looped training
            self.replay_buffers: List[ReplayBuffer] = []
            for _ in range(self.dim_act):
                self.replay_buffers.append(
                    ReplayBuffer(self.buffer_size, self.env.observation_space, self.env.action_space,
                                 handle_timeout_termination=False, device=self.device)
                )

    def _init_state_values_to_wandb(self) -> None:
        """
        Logs the current state value for all subpolicies to wandb and of the upper q_function if it is used.
        """
        if self.episodes_completed % self.wandb_freq != 0:
            return

        super()._init_state_values_to_wandb()
        if self.use_upper_q:
            action = self.policy.get_action(self.curr_obs)
            state_value = self.upper_q(torch.concat([torch.tensor(self.curr_obs).to(self.device),
                                                     action]).float()).max()
            self.wandb_run.log({"V_init_upper_q": state_value}, step=self.steps_taken)
