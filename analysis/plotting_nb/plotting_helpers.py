"""
Some helper functions that translate between the wandb experiment logging and partly messy experiment data handling and how we want to plot things.
"""

import numpy as np
import pandas as pd
from dacbench.envs import DiffImportanceFineTuneSigmoidEnv, SigmoidEnv, PiecewiseLinearEnv
from dacbench.benchmarks import SigmoidBenchmark, PiecewiseLinearBenchmark
import torch
from candid_dac.policies import AtomicPolicy, FactorizedPolicy
from itertools import product


# some constants for the plottings, colorblind friendly according to https://gist.github.com/thriveth/8560036
METHOD_COLOURS = {'adqn': '#377eb8', 'fdqn': '#ff7f00', 'fdqn_a': '#4daf4a', 'sdqn': '#e41a1c',
                  'ddqn': '#377eb8', 'iql': '#ff7f00', 'saql': '#4daf4a'}

# where to get the data from
CONFIG_PATH = "../run_data/MP_smooth_config_data.csv"
TRAIN_REWARD_PATH = "../run_data/MP_smooth_avg_episodic_reward.csv"
CKPT_REWARD_PATH = "../run_data/MP_smooth_ckpt_eval.csv"
# CONFIG_PATH = "../run_data/MP_final_config_data.csv"
# TRAIN_REWARD_PATH = "../run_data/MP_final_avg_episodic_reward.csv"
# CKPT_REWARD_PATH = "../run_data/MP_final_ckpt_eval.csv"


# some more helper functions
def find_nearest(array, value):
    """
    helper function that returns the closest value in a sorted array to a given value
    """
    idx = np.searchsorted(array, value)
    if idx == 0:
        return array[0]
    elif idx == len(array):
        return array[-1]
    else:
        dist_low = value - array[idx - 1]
        dist_high = array[idx] - value
        return array[idx - 1] if dist_low < dist_high else array[idx]


def translate_run_name(run_name: str):
    """
    helper function that translates the run name to generally used names
    """
    if 'adqn' in run_name or 'ddqn' in run_name:
        return 'DDQN'
    elif 'fdqn_a' in run_name or 'saql' in run_name:
        return 'SAQL'
    elif 'fdqn' in run_name or 'iql' in run_name:
        return 'IQL'
    elif 'sdqn' in run_name:
        return 'simSDQN'
    else:
        return run_name


def compute_optimal_episode_reward(env: SigmoidEnv, possible_actions: np.ndarray,
                                   reward_shape: str = None, c: float = None) -> float:
    """
    helper function that computes the optimal episode reward achievable in the environment with its current instance.
    """
    if isinstance(env, DiffImportanceFineTuneSigmoidEnv):
        shift = env.shifts[0]
        slope = env.slopes[0]

        optim_episode_reward = 0

        # compute the optimal prediction for every time step
        for t in range(env.n_steps):
            truth = env._sig(t, slope, shift)
            best_pred_acc = find_nearest(possible_actions, truth)

            if reward_shape == 'exponential':
                optim_episode_reward += np.exp(-c * np.abs(truth - best_pred_acc))
            else:
                optim_episode_reward += 1 - np.abs(truth - best_pred_acc)

    elif isinstance(env, PiecewiseLinearEnv):
        optim_episode_reward = 0
        for t in range(env.n_steps):
            truth = env._get_target(t)

            best_pred_acc = find_nearest(possible_actions, truth)

            if reward_shape == 'exponential' or reward_shape == 'exp':
                optim_episode_reward += np.exp(-c * np.abs(truth - best_pred_acc))
            else:
                optim_episode_reward += 1 - np.abs(truth - best_pred_acc)

    else:
        optim_episode_reward = 0
        for t in range(env.n_steps):
            step_reward = 1
            for i in range(env.n_actions):
                truth = env._sig(t, env.slopes[i], env.shifts[i])
                best_pred_acc = find_nearest(possible_actions, truth)

                step_reward *= (1 - np.abs(truth - best_pred_acc))

            optim_episode_reward += step_reward

    return optim_episode_reward


def eval_policy_on_sigmoid(policy: AtomicPolicy | FactorizedPolicy, env: SigmoidEnv) -> np.ndarray:
    """
    helper function that evaluates the given policy on all instances of the sigmoid environment and returns
    the episodic reward per instance.
    """
    rewards_per_episode = np.full(len(env.instance_id_list), np.nan)

    for _ in range(len(env.instance_id_list)):
        obs, _ = env.reset()
        episodic_reward = 0
        done = False

        while not done:
            if isinstance(policy, FactorizedPolicy):
                action = policy.get_action(obs)
            else:
                action = policy(torch.tensor(obs))
                action = np.unravel_index(action, env.action_space.nvec)
            obs, reward, terminated, truncated, _ = env.step(action)
            episodic_reward += reward
            done = terminated or truncated

        rewards_per_episode[env.inst_id] = episodic_reward

    return rewards_per_episode


def load_policy_from_checkpoint(config: dict, env: SigmoidEnv, ckpt_directory: str = None, episode: int = 1000,
                                final: bool = False) -> AtomicPolicy | FactorizedPolicy:
    """
    helper function that initializes a policy from a checkpoint recorded at the given episode for the specified experiment.
    If no checkpoint directory is given, a randomly initialized policy is returned matching the experimental setup.
    """
    episode = 'final' if final else episode
    if config['fdqn']:
        if ckpt_directory is None:
            q_network_paths = None
        else:
            q_network_paths = []
            for i in range(config['dim']):
                q_network_paths.append(f'{ckpt_directory}/{episode}_q_network_{i}.pth')
        autorecursive = config['autorecursive'] or 'sdqn' in config['run_name']
        policy = FactorizedPolicy(env.observation_space.shape[0], len(env.action_space.nvec), int(env.action_space.nvec[0]),
                                  path_to_q_ckpts=q_network_paths, autorecursive=autorecursive)

    else:
        if ckpt_directory is None:
            q_network_path = None
            target_network_path = None
        else:
            q_network_path = f'{ckpt_directory}/{episode}_q_network_0.pth'
            target_network_path = f'{ckpt_directory}/{episode}_target_network_0.pth'
        policy = AtomicPolicy(env.observation_space.shape[0], np.prod(env.action_space.nvec), q_ckpt=q_network_path,
                              target_ckpt=target_network_path)

    return policy


def get_best_possible_avg_reward(dim: int, benchmark: str, reward_shape: str = None, c: float = None,
                                 importance_base: float = 0.5, n_acts: int = 3, max_dim: int = None) -> float:
    """
    helper function that computes the best possible average reward for the sigmoid environment.

    returns:
        optim_avg_reward: float
            the best possible average reward achievable in the environment
    """
    if benchmark in ['candid_sigmoid', 'piecewise_linear']:
        possible_actions_acc = get_actions_importance_sigmoid(max_dim, importance_base, n_acts)
    else:
        possible_actions_acc = np.array([1 / (n_acts - 1) * i for i in range(n_acts)])
        print(f"Available actions, accumulating up to dimension {max_dim}")
        print(possible_actions_acc)

    if benchmark == 'candid_sigmoid':
        benchmark = SigmoidBenchmark()
        importances = np.array([importance_base**i for i in range(dim)])
        env = benchmark.get_importances_benchmark(dimension=dim, seed=0,
                                                  multi_agent=False,
                                                  importances=importances,
                                                  reward_shape=reward_shape)

        # loop over the instances of the benchmark and compute the best possible reward
        optim_reward_per_episode_acc = np.full(len(env.instance_id_list), np.nan)
        env.reset()
        for i in range(len(env.instance_id_list)):
            optim_reward_per_episode_acc[i] = compute_optimal_episode_reward(env, possible_actions_acc, reward_shape, c)
            env.reset()
    elif benchmark == 'piecewise_linear':
        importances = np.array([importance_base**i for i in range(dim)])
        benchmark = PiecewiseLinearBenchmark()
        benchmark.set_action_values([n_acts for _ in range(dim)], importances)
        env = benchmark.get_environment()
        optim_reward_per_episode_acc = np.full(len(env.instance_id_list), np.nan)
        env.reset()
        for i in range(len(env.instance_id_list)):
            optim_reward_per_episode_acc[i] = compute_optimal_episode_reward(env, possible_actions_acc, reward_shape, c)
            env.reset()
    else:
        benchmark = SigmoidBenchmark()
        env = benchmark.get_benchmark(dimension=dim, seed=0, multi_agent=False)

        # loop over the instances of the benchmark and compute the best possible reward
        env.reset()
        optim_reward_per_episode_acc = np.full(len(env.instance_id_list), np.nan)
        for i in range(len(env.instance_id_list)):
            optim_reward_per_episode_acc[i] = compute_optimal_episode_reward(env, possible_actions_acc, reward_shape, c)
            env.reset()

    return np.mean(optim_reward_per_episode_acc)


def get_actions_importance_sigmoid(max_dim: int, importance_base: float = 0.5, n_acts: int = 3) -> np.ndarray:
    """
    helper function that computes the possible aggregated actions up to max_dim for the importance sigmoid environment.
    """
    possible_actions_per_dim = [[1 / (n_acts - 1) * i for i in range(n_acts)]]
    for i in range(1, max_dim):
        possible_actions_per_dim.append([importance_base**i * (j / (n_acts - 1) - 0.5) for j in range(n_acts)])

    possible_actions_per_dim = np.array(possible_actions_per_dim)

    indices = product(*[range(len(vec)) for vec in possible_actions_per_dim])
    possible_actions_acc = [sum(possible_actions_per_dim[i, j] for i, j in enumerate(indices_tuple)) for indices_tuple in indices]
    possible_actions_acc = np.array(possible_actions_acc)
    # remove all duplicates
    possible_actions_acc = np.unique(possible_actions_acc)
    possible_actions_acc = np.sort(possible_actions_acc)

    return possible_actions_acc


def get_run_configs(df_config: pd.DataFrame, dim: int, benchmark: str, config_id: int, reverse_agents: bool = False,
                    importance_base: float = 0.5, reward_shape: str = "exp", exp_reward: float = 4.6, n_act: int = float) -> pd.DataFrame:
    """
        Filters the given dataframe of experiment configurations to only contain the relevant experiments for the given
        setup.
    """

    if config_id == 'best':
        df_config = df_config[df_config['config_id'].str.contains('best')]
    else:
        df_config = df_config[(df_config['config_id'] == config_id)]

    df_config = df_config[
        (df_config['dim'] == dim) &
        (df_config['benchmark'] == benchmark) &
        (df_config['reverse_agents'] == reverse_agents) &
        (df_config['n_act'] == n_act)
    ]

    if benchmark in ['candid_sigmoid', 'piecewise_linear']:
        df_config = df_config[
            (df_config['importance_base'] == importance_base) &
            (df_config['reward_shape'] == reward_shape)
        ]
        if reward_shape == 'exp':
            df_config = df_config[
                (df_config['exp_reward'] == exp_reward)
            ]

    return df_config
