"""
Defines utility functions for evaluating the performance of policies.
"""
from dacbench.benchmarks import SigmoidBenchmark
from candid_dac.leader_follower_policies import Policy
import numpy as np


def test_agent(policy: Policy, dim: int):
    bench = SigmoidBenchmark()
    env = bench.get_benchmark(dimension=dim, seed=0)
    env.use_test_set()

    print(f'Starting evaluation on test set. Found {len(env.instance_id_list)} test instances.')

    # evaluate on all test instances.
    episodic_returns = np.full(len(env.instance_id_list), np.nan)
    episode_reward = 0

    # interact with environment until we evaluated the agent on all instances from the test set
    obs, _ = env.reset()
    instances_evaluated = 0
    while True:
        action = policy.get_action(obs)

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if truncated:
            episodic_returns[instances_evaluated] = episode_reward
            instances_evaluated += 1

            episode_reward = 0

            if instances_evaluated >= len(env.instance_id_list):
                break

            obs, _ = env.reset()

    print(f'Evaluated on {instances_evaluated} test instances.')

    return np.nanmean(episodic_returns)
