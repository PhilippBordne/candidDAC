import numpy as np
import torch
import random
from tqdm import tqdm
from wandb.wandb_run import Run
from pathlib import Path
from candid_dac.policies import AtomicPolicy, FactorizedPolicy
import time
from candid_dac.utils.utils import linear_schedule
import copy
from dacbench import AbstractEnv
import logging
from omegaconf import DictConfig


class DQN():
    def __init__(
            self, env: AbstractEnv, hp_config: DictConfig, steps_tot: int, seed: int, wandb_run: Run = None,
            wandb_det_freq: int = 10, disable_pbar: bool = False, model_dir: str = None, store_models=True,
            eval_train=False, eval_test=False, eval_freq: int = 100, predict_init: bool = False):
        """
        TODO: docstring

        Parameters
        ----------
        tau : float
            Soft update factor for target network.
        wandb_freq : int
            Frequency after every how many episodes to log detailed data such as the values of initial states to wandb.
        predict_init: bool
            Whether to predict the initial state values according to the learned policies.
        """
        # seed python, numpy and torch
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.env = env

        # general algorithm hyperparams
        self.lr = hp_config["lr"]
        self.start_e = hp_config["start_e"]
        self.end_e = hp_config["end_e"]
        self.expl_fraction = hp_config["expl_fraction"]
        self.tau = hp_config["tau"]
        self.gamma = hp_config["gamma"]
        self.batch_size = hp_config["batch_size"]
        self.buffer_size = hp_config["buffer_size"]
        self.freq_q = hp_config["freq_q"]
        self.freq_target = hp_config["target_update_freq"]
        self.learning_starts = hp_config["learning_starts"]
        self.seed = seed
        self.steps_tot = steps_tot

        # settings for policy evaluation during training
        self.eval_freq = eval_freq
        self.eval_train = eval_train
        self.eval_test = eval_test

        self.predict_init = predict_init

        # if running evaluations during execution use a copy of the environment to not interfere with the training
        if self.eval_train or self.eval_test:
            self.eval_env = copy.deepcopy(env)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device {self.device}")

        # extract some properties of environment for convenience
        self.dim_obs = env.observation_space.shape[0]
        self.dim_act = env.action_space.shape[0]

        # statistics to track algorithm progress
        self.steps_taken = 0
        self.episodes_completed = 0
        self.episode_reward = 0

        self.last_episodes_to_avg = len(self.env.instance_id_list)
        self.episodic_rewards = np.full(self.last_episodes_to_avg, np.nan)

        self.wandb_run = wandb_run
        self.wandb_freq = wandb_det_freq
        # from time to time store the policy networks (in total we want to store 10 models over the course of training)
        self.store_models = store_models
        self.model_store_freq = self.steps_tot / (10 * self.env.n_steps)

        self.cluster_mode = disable_pbar
        # store the models besides the wandb directory, using the run_id as subdirectory
        if self.wandb_run is not None and self.store_models:
            # using the pbar as cluster-mode
            if model_dir is not None:
                self.model_store_dir = str(Path(model_dir)/wandb_run.project_name()/wandb_run.id)
            else:
                self.model_store_dir = f"{Path(__file__).parents[2]}/models/{wandb_run.project_name()}/{wandb_run.id}"
            print(f"Storing models at {self.model_store_dir}")
            Path(self.model_store_dir).mkdir(parents=True, exist_ok=True)

        self.setup_for_next_episode()

    def train(self, n_steps: int = None):
        if n_steps is None:
            n_steps = self.steps_tot - self.steps_taken

        self.setup_for_next_episode()
        if self.cluster_mode:
            training_iterator = range(n_steps)
            start_time = time.time()
        else:
            training_iterator = tqdm(range(n_steps))

        for _ in training_iterator:
            # specific steps of different DQN versions

            # sample greedy action from policy and exploration mask where exploration might be different per dimension
            if np.random.rand() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.sample_greedy_action()

            self.curr_action = action

            # take action in environment
            # as the Sigmoid environments uses truncated for finished episodes, we check that instead of the done flag
            # TODO: change sigmoid benchmark to return done flag after episode is finished
            self.next_obs, self.curr_reward, _, self.episode_done, _ = self.env.step(action)

            # store the just finished transition in the replay buffer
            self.store_to_buffer()

            self.episode_reward += self.curr_reward

            # since all episodes truncated instead of done by Sigmoid we need to reset the environment if done
            if self.episode_done:
                self.episodes_completed += 1
                self.episodic_rewards[self.episodes_completed % self.last_episodes_to_avg] = self.episode_reward
                # update the progress bar
                if not self.cluster_mode:
                    training_iterator.set_description(f"Avg. episode reward: {np.nanmean(self.episodic_rewards):.2f}")
                elif self.episodes_completed % self.model_store_freq == 0:
                    time_elapsed = time.time() - start_time
                    time_elapsed_print = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
                    # int(time.time() - start_time).strftime("%H:%M:%S")
                    print(f"{self.steps_taken / self.steps_tot:.0%} | {time_elapsed_print}"
                          f" ({self.steps_taken / time_elapsed} iters / sec) | "
                          f"Avg. episode reward: {np.nanmean(self.episodic_rewards):.2f}")

                if self.episodes_completed % self.eval_freq == 0:
                    if self.eval_train:
                        self.eval_policy(test=False)
                    if self.eval_test:
                        self.eval_policy(test=True)

                self.setup_for_next_episode()

                if self.wandb_run is not None and self.episodes_completed % self.model_store_freq == 0:
                    self.save_policy_networks()

            else:
                self.curr_obs = self.next_obs

            # update the Q-function and target network if enough steps have been taken
            if self.steps_taken >= self.learning_starts:
                if self.steps_taken % self.freq_q == 0:
                    self.update_q()
                if self.steps_taken % self.freq_target == 0:
                    self.update_target()

            self.steps_taken += 1

        # log the final avg episodic reward
        if self.wandb_run is not None:
            self.wandb_run.log(
                {"episode_reward": self.episode_reward,
                 "avg_episodic_reward": np.nanmean(self.episodic_rewards)},
                step=self.steps_taken)

    def eval_policy(self, test: bool = False) -> None:
        """
        Evaluates the policy on all instances of the environment. Returns the average reward over all instances.

        Parameters
        ----------
        test : bool
            Whether to evaluate the policy on the test set.
        """
        if self.wandb_run is None:
            logging.error(f"Skipping policy evaluation on {'test' if test else 'train'} set as no wandb run is active.")
            return

        # only update the environment if the desired test set is not already active
        if test and not self.eval_env.test:
            self.eval_env.use_test_set()
        elif not test and self.eval_env.test:
            self.eval_env.use_training_set()
        else:
            pass

        eval_rewards = np.full(len(self.eval_env.instance_id_list), np.nan)
        for i, instance_id in enumerate(self.eval_env.instance_id_list):
            obs, _ = self.eval_env.reset(instance_id=instance_id)
            done = False
            episode_reward = 0
            while not done:
                action = self.sample_greedy_action(obs)
                obs, reward, _, done, _ = self.eval_env.step(action)
                episode_reward += reward
            eval_rewards[i] = episode_reward

        avg_reward = np.nanmean(eval_rewards)
        data_field = "avg_reward_test_set" if test else "avg_reward_train_set"
        self.wandb_run.log({data_field: avg_reward}, step=self.steps_taken)

    def save_policy_networks(self):
        if not self.store_models:
            return

        if self.wandb_run is None:
            raise NotImplementedError("Method to save policy networks currently requires an active wandb run.")

        index = "final" if self.episodes_completed == self.steps_tot / self.env.n_steps else self.episodes_completed
        # (Path(self.model_store_dir) / index).mkdir(parents=True, exist_ok=True)
        if isinstance(self.policy, AtomicPolicy):
            torch.save(self.policy.q_network.state_dict(),
                       f"{self.model_store_dir}/{index}_q_network_0.pth")
            torch.save(self.policy.target.state_dict(),
                       f"{self.model_store_dir}/{index}_target_network_0.pth")
            self.wandb_run.save(f"{self.model_store_dir}/{index}_q_network_0.pth")
            self.wandb_run.save(f"{self.model_store_dir}/{index}_target_network_0.pth")
        elif isinstance(self.policy, FactorizedPolicy):
            for i, policy in enumerate(self.policy.subpolicies):
                torch.save(policy.q_network.state_dict(),
                           f"{self.model_store_dir}/{index}_q_network_{i}.pth")
                torch.save(policy.target.state_dict(),
                           f"{self.model_store_dir}/{index}_target_network_{i}.pth")
                self.wandb_run.save(f"{self.model_store_dir}/{index}_q_network_{i}.pth")
                self.wandb_run.save(f"{self.model_store_dir}/{index}_target_network_{i}.pth")

        # raise NotImplementedError("Method to save policy networks needs to be implemented by subclass.")

    def setup_for_next_episode(self) -> None:

        self.curr_obs, _ = self.env.reset()
        self.episode_done = False
        self.curr_action = None
        self.curr_reward = None
        self.next_obs = -1 * np.ones(self.dim_obs)

        if self.wandb_run is not None and self.episodes_completed % self.wandb_freq == 0 and self.steps_taken > 0:
            self.wandb_run.log(
                {"episode_reward": self.episode_reward,
                 "avg_episodic_reward": np.nanmean(self.episodic_rewards)},
                step=self.steps_taken)
            # TODO: fix that the policy will not yet be defined when initializing the base DQN class
            if self.predict_init:
                self._init_state_values_to_wandb()

        self.episode_reward = 0

    def _init_state_values_to_wandb(self) -> None:
        """
        Logs the state value of the initial state to wandb.
        """
        pass

    def sample_greedy_action(self, obs: np.ndarray = None) -> np.ndarray:
        """
        Samples greedy actions from the current policy.

        Parameters
        ----------
        obs : np.ndarray
            Observation to sample action from. If None, uses the current observation.

        Returns
        -------
        np.ndarray
            Sampled action with shape (dim_act).
        """
        raise NotImplementedError

    def store_to_buffer(self) -> None:
        """
        Stores last completed transition (o_t, a_t, r_t, o_t+1) to the replay buffer.
        """
        raise NotImplementedError

    def update_q(self) -> None:
        """
        Updates the Q-function of the policy.
        """
        raise NotImplementedError

    def update_target(self) -> None:
        """
        Updates the target network of the policy.
        """
        raise NotImplementedError

    @property
    def epsilon(self) -> float:
        if self.steps_taken < self.learning_starts:
            return 1
        else:
            return linear_schedule(self.start_e, self.end_e, self.expl_fraction * self.steps_tot, self.steps_taken)
