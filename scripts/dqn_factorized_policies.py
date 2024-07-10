# suppress import not at top of file warning, as we first want to suppress warnings
# flake8: noqa: E402
import os
from pathlib import Path
import numpy as np
import wandb
import logging
import hydra

# ignore all warnings from dacbench to keep output clean
import warnings

import wandb.wandb_run
warnings.filterwarnings(category=FutureWarning, action="ignore", module="dacbench.*")
warnings.filterwarnings(category=UserWarning, action="ignore", module="dacbench.*")

from dacbench.benchmarks.sigmoid_benchmark import SigmoidBenchmark
from dacbench.benchmarks.piecewise_linear_benchmark import PiecewiseLinearBenchmark, PIECEWISE_LINEAR_DEFAULTS
from dacbench.abstract_benchmark import objdict
from candid_dac.algorithms.atomic_dqn import AtomicDQN
from candid_dac.algorithms.factorized_dqn import FactorizedDQN
from candid_dac.algorithms.sequential_dqn import SDQN
from candid_dac.utils.cli import parse_args
from omegaconf import DictConfig, OmegaConf, open_dict


DEFAULT_RESULTS_DIR = f'{Path(__file__).parent.parent}/results'


# utils function to sample random hyperparameters
def sample_hyperparams(hp_config: DictConfig) -> OmegaConf:
    # extract the seed for the hyperparameter sampling
    if hp_config.seed is None:
        hp_seed = 42
        hp_config.seed = hp_seed
        logging.warning(f"No hyperparameter seed provided. Using default seed {hp_seed}.")
    else:
        hp_seed = hp_config.seed
    rng_hyperparams = np.random.default_rng(hp_seed)

    # the following lines define the hyperparameter search space

    # sample uniform on logscale
    lr = 10**rng_hyperparams.uniform(low=np.log10(1e-5), high=np.log10(0.1))
    batch_size = int(2**rng_hyperparams.uniform(low=np.log2(16), high=np.log2(256)))
    target_update_freq = rng_hyperparams.uniform(np.log10(10 * hp_config.freq_q),
                                                    np.log10(100 * hp_config.freq_q))
    target_update_freq = int(10**target_update_freq)

    # sample uniform
    tau = rng_hyperparams.uniform(0.1, 1)
    # in order to not change the other sampled hyperparams sample gamma even if fixed value is given
    gamma = rng_hyperparams.uniform(0.9, 1)
    start_e = rng_hyperparams.uniform(hp_config.end_e, 1)

    sampled_hps = OmegaConf.create({
            'lr': lr,
            'batch_size': batch_size,
            'target_update_freq': target_update_freq,
            'tau': tau,
            'gamma': gamma,
            'start_e': start_e
        })
    return sampled_hps

# function that wraps setup of the wandb run
def setup_wandb(config: DictConfig) -> wandb.wandb_run.Run:
    # TODO: rework this for the new implementation
    algorithm_name = config.algorithm.name

    run_name = f'{algorithm_name}_{config.benchmark.dim}D'

    # set the model directory to the current directory if not provided
    os.environ['WAND_DIR'] = DEFAULT_RESULTS_DIR if config.track_training.results_dir is None else config.track_training.results_dir
    
    benchmark_name = config.benchmark.name

    wandb_tags = config.wandb.tags
    wandb_run = wandb.init(project=config.wandb.project_name,
                            config={
                                'benchmark': benchmark_name,
                                'dim': config.benchmark.dim,
                                'config_id': config.hyperparameters.config_id,
                                'algorithm': algorithm_name,
                                # for backward compatibility
                                'sdqn': config.algorithm.name == 'sdqn',
                                'fdqn': config.algorithm.name in ['sdqn', 'iql', 'saql'],
                                'importance_sigmoid': benchmark_name == 'candid_sigmoid',
                                'piecewise_linear': benchmark_name == 'piecewise_linear',
                                'importance_base': config.benchmark.get('importance_base', None),
                                'reward_shape': config.benchmark.get('reward_shape', None),
                                'exp_reward': config.benchmark.get('exp_reward', None),
                                'reverse_agents': config.algorithm.reverse_agents,
                                'n_act': config.benchmark.get('n_act', None),
                                # extract the hyperparams from the config dict
                                **config.hyperparameters,
                                # extract algorithm design choices
                                **config.algorithm,
                                # make sure, the seed tracked by wandb is the experiment seed (might be ambiguous, e.g. hyperparam seed)
                                'seed': config.seed,
                                'total_episodes': config.total_episodes,
                            },
                            name=run_name,
                            tags=wandb_tags if wandb_tags is not None else [],
                            reinit=True)
    return wandb_run

# main function that is called by the CLI and manages the experiment, e.g.
# setting up the environment, the algorithm, the wandb run and if necessary sampling hyperparameters
@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig):
    
    # determine seed of the experiment
    if config.seed is None:
        experiment_seed = 42
        config.seed = experiment_seed
        logging.warning(f"No seed provided. Using default seed {experiment_seed}.")
    else:
        experiment_seed = config.seed

    # check whether an existing config is provided, otherwise sample for tunable hyperparameters
    if config.hyperparameters.sample:
        sampled_hps = sample_hyperparams(config.hyperparameters)
        config.hyperparameters = OmegaConf.merge(config.hyperparameters, sampled_hps)

    # setup wandb run if project name is provided
    if config.wandb.project_name is not None:
        wandb_run = setup_wandb(config)
    else:
        wandb_run = None

    # instantiate the benchmark, so we can later create our experimental environment from it
    if config.benchmark.name in ['candid_sigmoid', 'piecewise_linear']:
        # these benchmarks emulate CANDID and thus require importance per dimension
        importances = np.array([config.benchmark.importance_base**i for i in range(config.benchmark.dim)])
    if config.benchmark.name == 'candid_sigmoid':
        # the CANDID sigmoid benchmark wasn't reported in the paper, but is included for completeness 
        benchmark = SigmoidBenchmark()
        env = benchmark.get_importances_benchmark(dimension=config.benchmark.dim, seed=experiment_seed,
                                                  multi_agent=False,
                                                  importances=importances,
                                                  reward_shape=config.benchmark.reward_shape,
                                                  exp_reward=config.benchmark.exp_reward,
                                                  reverse_agents=config.algorithm.reverse_agents)
    elif config.benchmark.name == 'piecewise_linear':
        # configure the piecewise linear benchmark
        benchmark = PiecewiseLinearBenchmark()
        benchmark.read_instance_set()
        benchmark.read_instance_set(test=True)
        benchmark.set_action_values([config.benchmark.n_act for _ in range(config.benchmark.dim)], importances)
        benchmark.config['reverse_agents'] = config.algorithm.reverse_agents
        env = benchmark.get_environment()
    else:
        # otherwise fallback to the original sigmoid benchmark
        benchmark = SigmoidBenchmark()
        env = benchmark.get_benchmark(dimension=config.benchmark.dim, seed=experiment_seed, multi_agent=False)

    # seed the environment
    env.seed(experiment_seed)

    # set the directory where to store the model ckpts
    model_directory = DEFAULT_RESULTS_DIR + "/models" if config.track_training.results_dir is None else config.track_training.results_dir + "/models"
    # depending on the hydra config select the corresponding algorithm and parameterize it
    if config.algorithm.name == 'sdqn':
        algorithm = SDQN(env=env, hp_config=config.hyperparameters, algorithm_choices=config.algorithm,
                         steps_tot=config.total_episodes * env.n_steps,
                         seed=experiment_seed, wandb_run=wandb_run, disable_pbar=config.track_training.disable_pbar,
                         model_dir=model_directory, eval_train=config.track_training.eval_train,
                         eval_test=config.track_training.eval_test, store_models=config.track_training.store_ckpts,
                         predict_init=config.track_training.predict_initial_v)
    elif config.algorithm.name in ['iql', 'saql']:
        algorithm = FactorizedDQN(env=env, hp_config=config.hyperparameters, algorithm_choices=config.algorithm,
                                  steps_tot=config.total_episodes * env.n_steps,
                                  seed=experiment_seed, wandb_run=wandb_run, disable_pbar=config.track_training.disable_pbar,
                                  model_dir=model_directory, eval_train=config.track_training.eval_train,
                                  eval_test=config.track_training.eval_test, store_models=config.track_training.store_ckpts,
                                  predict_init=config.track_training.predict_initial_v)
    else:
        algorithm = AtomicDQN(env=env, hp_config=config.hyperparameters, steps_tot=config.total_episodes * env.n_steps,
                              seed=experiment_seed, wandb_run=wandb_run, disable_pbar=config.track_training.disable_pbar,
                              model_dir=model_directory, eval_train=config.track_training.eval_train,
                              eval_test=config.track_training.eval_test, store_models=config.track_training.store_ckpts,
                              predict_init=config.track_training.predict_initial_v)
    
    # train the algorithm for the configured number of episodes
    algorithm.train(n_steps=config.total_episodes * env.n_steps)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
