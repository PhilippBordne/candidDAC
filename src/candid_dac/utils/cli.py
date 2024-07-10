import argparse


# adapt the arguments to parse
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default='single-agent-sigmoid',
                        help="the name of this experiment")
    parser.add_argument("--disable-pbar", action="store_true",
                        help="Wether to disable the progress bar during training.")
    parser.add_argument("--store-ckpts", action="store_true",
                        help="If flag is set model checkpoints will be stored at several (10) time points during"
                             "training.")
    parser.add_argument("--eval-train", action="store_true",
                        help="If set, evaluate greedy policy on training set every 100 episodes.")
    parser.add_argument("--eval-test", action="store_true",
                        help="If set, evaluate greedy policy on test set every 100 episodes.")
    parser.add_argument("--predict-init", action="store_true",
                        help="If set, the initial state values of learned policies are logged to wandb.")

    # setup seeds and HP sweep
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the experimental setup, e.g. all but hyperparameter seed.")
    parser.add_argument("--total-episodes", type=int, default=10000,
                        help="number of episodes to complete for training")
    parser.add_argument("--hp-seed", type=int, default=0,
                        help="seed for RNG for random hyperparameter search.")
    parser.add_argument("--n-random-hyperparams", type=int, default=1,
                        help="Number of random hyperparameter configurations to evaluate. Starts counting from provided hp-seed")
    parser.add_argument("--use-default-hp", action="store_true",
                        help="Wether to use the default set of hyperparameters.")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="No of seeds to train the agent on. (increasing, start with seed provided through CLI)")
    parser.add_argument("--fixed-gamma", type=float, default=None,
                        help="If provided, and doing a hyperparameter search, use this gamma value.")

    # setup benchmark / experiment
    parser.add_argument("--importance-sigmoid", action="store_true",
                        help="Wether to use the importance (CANDID) sigmoid benchmark instead of the default sigmoid.")
    parser.add_argument("--piecewise-linear", action="store_true",
                        help="Wether to use the piecewise linear benchmark instead of the default sigmoid.")
    parser.add_argument("--importance-base", type=float, default=0.2,
                        help="aka importance decay in the report")
    parser.add_argument("--benchmark-dim", type=int, default=1,
                        help="Dimensionality of the sigmoid benchmark. 1D, 2D, 3D, 5D available.")
    parser.add_argument("--n-instances", type=int, default=None,
                        help="Limit the number of instances used for training the agent. "
                             "If None use all benchmark instances. Might be used for debugging.")
    parser.add_argument("--instance-update-scheme", type=str, default="round-robin")
    parser.add_argument("--reward-shape", type=str, default="exponential",
                        help="The shape of the reward function. Options: linear, exponential (only exponential used in the report)")
    parser.add_argument("--exp-reward", type=float, default=4.6,
                        help="The exponent of exponential reward function that scales the prediction error (c in the report).")
    parser.add_argument("--reverse-agents", action="store_true",
                        help="If activated, the agents will be ordered in reverse (e.g. the first agent controls least"
                             "important action).")

    # wandb
    parser.add_argument("--wandb-project-name", type=str, default=None,
                        help='WandB-projects where to store the runs. If not provided no tracking in WandB.')
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help='Name under which to list the run in WandB.')
    parser.add_argument("--wandb-tag", type=str, default=None,
                        help='Optional tag, e.g. to highlight the run.')
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory to store results of the experiment (wandb artifacts and model checkpoints).")

    # multi-agent setup general
    parser.add_argument("--fdqn", action="store_true",
                        help="Wether to use vanilla single agent or leader follower algorithm.")
    parser.add_argument("--autorecursive", action="store_true", help="If true, already selected actions in current "
                        "state are appended to the state of the agent to choose its action next.")
    parser.add_argument("--sdqn", action="store_true",
                        help="Wether to train autorecursive policies using SDQN.")
    parser.add_argument("--shared-buffer", action="store_true",
                        help="If activated all agents getting trained using the same batch from a shared buffer.")

    # multi-agent setup sdqn specific
    parser.add_argument("--use-upper-q", action="store_true",
                        help="Wether to use the upper Q-value to update the last sequential action against.")
    parser.add_argument("--update-towards-q", action="store_true",
                        help="If activated update leader Q-functions towards Q-function of their follower.")

    # overriding hyperparameters through CLI

    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=2500,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
                        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=2500,
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=1,
                        help="the frequency of training")

    args = parser.parse_args()

    return args
