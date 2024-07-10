import numpy as np
import torch

from gymnasium import Env
from gymnasium.spaces import Discrete
from stable_baselines3.common.buffers import ReplayBuffer
from wandb.wandb_run import Run

from candid_dac.algorithms.base_dqn import DQN
from candid_dac.policies import AtomicPolicy
from omegaconf import DictConfig


class AtomicDQN(DQN):
    def __init__(self, env: Env, hp_config: DictConfig, steps_tot: int, seed: int, wandb_run: Run = None,
                 **kwargs) -> None:
        super().__init__(env, hp_config, steps_tot, seed, wandb_run, **kwargs)

        # in the single agent setting we only have one policy and one buffer
        self.policy = AtomicPolicy(self.dim_obs, np.prod(self.env.action_space.nvec), lr=hp_config['lr'],
                                   device=self.device)

        # the buffer uses the 1-D action space
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, self.env.observation_space, Discrete(np.prod(self.env.action_space.nvec)),
            handle_timeout_termination=False, device=self.device)

    def sample_greedy_action(self, obs: np.ndarray = None) -> np.ndarray:
        obs = self.curr_obs if obs is None else obs
        flattened_action = self.policy(torch.tensor(obs)).numpy()
        unflattened_action = np.unravel_index(flattened_action, self.env.action_space.nvec)
        return np.array(unflattened_action)

    def store_to_buffer(self) -> None:
        self.replay_buffer.add(self.curr_obs, self.next_obs,
                               np.ravel_multi_index(self.curr_action, self.env.action_space.nvec),
                               self.curr_reward, self.episode_done, {})

    def update_q(self) -> None:
        batch = self.replay_buffer.sample(self.batch_size)

        # compute target
        with torch.no_grad():
            argmax = self.policy.q_network(batch.next_observations).argmax(axis=-1)
            q_next = self.policy.target(batch.next_observations).gather(dim=-1, index=argmax.view(-1, 1))
            target = batch.rewards.flatten() + self.gamma * q_next.flatten() * (1 - batch.dones.flatten())
        q = self.policy.q_network(batch.observations).gather(dim=-1, index=batch.actions.view(-1, 1)).flatten()

        loss = self.policy.loss(q, target)
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

    def update_target(self) -> None:
        for target_param, param in zip(self.policy.target.parameters(), self.policy.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _init_state_values_to_wandb(self) -> None:
        if self.episodes_completed % self.wandb_freq == 0:
            init_value = self.policy.q_network(torch.tensor(self.curr_obs).to(self.device).float()).max()
            self.wandb_run.log({"V_init_policy_0": init_value}, step=self.steps_taken)
