# Main script for random mg experiments
import argparse
import random
import time
from collections import defaultdict
from typing import List

import tensorflow as tf
import numpy as np
import tree

from demos import Demonstrations
from irl_types import IRLParams, Config, RunData, MHConfig
from porp_mh import porp_mh_reward_step, porp_mh_policy_step
from drp_mh import drp_mh
from sgld.SGLD import RMSPropSGLDConfig, RMSPropSGLD
from drp import drp_ula
from tf_models import SoftmaxPolicies, SigmoidParametrisedReward, convert_to_irl_params
from porp import porp_ula_policy_step, porp_ula_reward_step
import multiprocessing as mp

# Generic random mg config
alpha = 1.2
gamma = 0.9
lambda_reg = 0.1
episode_length = 100
num_actions = 3
num_states = 5

policy_ula_config = RMSPropSGLDConfig(
    eta_0=0.05,
    alpha=0.,
    noise_scale=1.,
    num_epochs=6_000,
    num_minibatches=1,
    shuffle_minibatches=False,
    epsilon=1e-8,
    beta=0.99,
    warmup_frac=0.5
)

reward_ula_config = RMSPropSGLDConfig(
    eta_0=0.2,  # kl 0.5
    alpha=0.,
    noise_scale=1.,
    num_epochs=5000,
    num_minibatches=1,
    shuffle_minibatches=False,
    epsilon=1e-8,
    beta=0.99,
    warmup_frac=0.5,
)

policy_mh_config = MHConfig(
    num_samples=16_000,
    step_size=0.1,
    warmup_frac=0.5,
)

reward_mh_config = MHConfig(
    num_samples=5000,
    step_size=0.1,
    warmup_frac=0.5,
)

drp_ula_config = RMSPropSGLDConfig(
    eta_0=0.03,
    alpha=0.,
    noise_scale=1.,
    num_epochs=5000,
    num_minibatches=1,
    shuffle_minibatches=False,
    epsilon=1e-8,
    beta=0.99,
    warmup_frac=0.5,
)

drp_mh_config = MHConfig(
    num_samples=5000,
    step_size=0.1,
    warmup_frac=0.5,
)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def generate_solution(config):
    reward_func = np.zeros((num_states, num_actions))
    reward_func[num_states - 1, 0] = 1

    truth = IRLParams(
        fairness_levels=tuple(np.random.uniform(-1, 1, config.pool_size).tolist()),
        reward_func=reward_func,
    )

    print("Truth:", truth)

    return truth


def generate_mg(config):

    initial_state_dist = np.zeros(num_states)
    initial_state_dist[0] = 1

    partial_mg_config = {
        "seed"              : config.seed,
        "num_states"        : num_states,
        "num_actions"       : num_actions,
        "alpha"             : alpha,
        "episode_length"    : episode_length,
        "initial_state_dist": initial_state_dist,
    }
    return partial_mg_config


def run_for_config(config: Config):
    if config.run_exists():
        return

    print('Running with:', config)
    seed_everything(config.seed)
    truth = generate_solution(config)
    partial_mg_config = generate_mg(config)

    demos = Demonstrations(
        partial_mg_config,
        gamma,
        lambda_reg
    )

    demos.get_demos(
        config.num_trajectories_per_pairing,
        truth
    )

    if config.sampling_algo == "ULA":
        run_data =run_ula(
            config,
            demos,
            partial_mg_config,
            truth,
        )
    elif config.sampling_algo == "MH":
        run_data =run_mh(
            config,
            demos,
            partial_mg_config,
            truth,
        )

    config.save_run(run_data)


def run_ula(
        config: Config,
        demos: Demonstrations,
        partial_mg_config: dict,
        truth,
):

    reward_model = SigmoidParametrisedReward(
        num_agents=config.pool_size,
        num_states=num_states,
        num_actions=num_actions,
        fairness_bounds=(-1., 1.)
    )

    if config.sampling_method == "PORP":
        reward_model.optimiser = RMSPropSGLD(**reward_ula_config._asdict())
        reward_model.optimiser.init(reward_model.trainable_variables)
        policies, rewards, metrics = run_ula_porp(
            demos=demos,
            reward_model=reward_model,
            prior_type=config.prior,
            gap_function=config.gap_function
        )
        run_data = RunData(
            solution=truth,
            mg_config=partial_mg_config,
            reward_samples=rewards,
            policy_samples=policies,
            metrics=metrics,
        )
    elif config.sampling_method == "DRP":
        reward_model.optimiser = RMSPropSGLD(**drp_ula_config._asdict())
        reward_model.optimiser.init(reward_model.trainable_variables)
        rewards, metrics = run_ula_drp(
            demos=demos,
            reward_model=reward_model,
        )

        run_data = RunData(
            solution=truth,
            mg_config=partial_mg_config,
            reward_samples=rewards,
            metrics=metrics,
        )
    return run_data


def run_ula_porp(
        demos: Demonstrations,
        reward_model: SigmoidParametrisedReward,
        prior_type: str,
        gap_function: str,
):

    policies = SoftmaxPolicies(
        num_policies=demos.num_combinations * 2,
        num_states=num_states,
        num_actions=num_actions,
        optim=RMSPropSGLD(**policy_ula_config._asdict())
    )

    t = time.time()

    # sample policies
    _, sampled_policies, _ = porp_ula_policy_step(
        demos=demos,
        policy_model=policies,
        sgld_config=policy_ula_config
    )
    policies = list(np.array(sampled_policies)[np.random.choice(len(sampled_policies), 1000)])

    t2 = time.time()


    _, sampled_rewards, _ = porp_ula_reward_step(
        demos=demos,
        reward_model=reward_model,
        policy_samples=policies,
        sgld_config=reward_ula_config,
        prior_type=prior_type,
        gap_function=gap_function
    )

    t3 = time.time()

    sampled_rewards = [
        convert_to_irl_params(*r) for r in sampled_rewards
    ]

    return sampled_policies, sampled_rewards, {"policy_posterior_time": t2 - t, "reward_posterior_time": t3 - t2, "porp_time": t3 - t}


def run_ula_drp(
        demos: Demonstrations,
        reward_model: SigmoidParametrisedReward,
):

    t = time.time()

    _, sampled_rewards, _ = drp_ula(
        demos=demos,
        reward_model=reward_model,
        sgld_config=drp_ula_config,
    )

    t3 = time.time()

    sampled_rewards = [
        convert_to_irl_params(*r) for r in sampled_rewards
    ]

    return sampled_rewards, {"drp_time": t3 - t}


def run_mh(
        config: Config,
        demos: Demonstrations,
        partial_mg_config: dict,
        truth,
):
    reward_model = SigmoidParametrisedReward(
        num_agents=config.pool_size,
        num_states=num_states,
        num_actions=num_actions,
        optim=RMSPropSGLD(**reward_ula_config._asdict()),
        fairness_bounds=(-1., 1.)
    )

    if config.sampling_method == "PORP":
        reward_model.optimiser = RMSPropSGLD(**reward_mh_config._asdict())
        reward_model.optimiser.init(reward_model.trainable_variables)
        policies, rewards, metrics = run_mh_porp(
            demos=demos,
            reward_model=reward_model,
            prior_type=config.prior,
            gap_function=config.gap_function
        )
        run_data = RunData(
            solution=truth,
            mg_config=partial_mg_config,
            reward_samples=rewards,
            policy_samples=policies,
            metrics=metrics,
        )
    elif config.sampling_method == "DRP":
        reward_model.optimiser = RMSPropSGLD(**drp_mh_config._asdict())
        reward_model.optimiser.init(reward_model.trainable_variables)
        rewards, metrics = run_mh_drp(
            demos=demos,
            reward_model=reward_model,
        )

        run_data = RunData(
            solution=truth,
            mg_config=partial_mg_config,
            reward_samples=rewards,
            metrics=metrics,
        )
    return run_data


def run_mh_porp(
        demos: Demonstrations,
        reward_model: SigmoidParametrisedReward,
        prior_type: str,
        gap_function: str,
):

    policies = SoftmaxPolicies(
        num_policies=demos.num_combinations * 2,
        num_states=num_states,
        num_actions=num_actions,
        optim=RMSPropSGLD(**policy_ula_config._asdict())
    )

    t = time.time()

    # sample policies
    _, sampled_policies, _ = porp_mh_policy_step(
        demos=demos,
        policy_model=policies,
        mh_config=policy_mh_config
    )

    t2 = time.time()

    policies = list(np.array(sampled_policies)[np.random.choice(len(sampled_policies), 1000)])

    _, sampled_rewards, _ = porp_mh_reward_step(
        demos=demos,
        reward_model=reward_model,
        policy_samples=policies,
        mh_config=reward_mh_config,
        prior_type=prior_type,
        gap_function=gap_function
    )

    t3 = time.time()

    sampled_rewards = [
        convert_to_irl_params(*r) for r in sampled_rewards
    ]

    return sampled_policies, sampled_rewards, {"policy_posterior_time": t2 - t, "reward_posterior_time": t3 - t2, "porp_time": t3 - t}

def run_mh_drp(
        demos: Demonstrations,
        reward_model: SigmoidParametrisedReward,
):

    t = time.time()

    _, sampled_rewards, _ = drp_mh(
        demos=demos,
        reward_model=reward_model,
        mh_config=drp_mh_config,
    )

    t3 = time.time()

    sampled_rewards = [
        convert_to_irl_params(*r) for r in sampled_rewards
    ]

    return sampled_rewards, {"drp_time": t3 - t}


def evaluate_rewards(configs: List[Config]):

    # over the runs, we compute: reward error, value error, (policy error)
    evaluation = defaultdict(lambda: defaultdict(list))

    for config in configs:
        path = config.unseeded_path()
        run_data = config.load_run()

        run_data = RunData(
            solution=run_data.solution,
            mg_config=run_data.mg_config,
            reward_samples=run_data.reward_samples,
            metrics=run_data.metrics
        )

        evaluation[path]["run_data"].append(run_data)  # raw data still useful for plotting
        
        fairness_error, base_reward_error = compute_reward_error(run_data)
        evaluation[path]["fairness_error"].append(fairness_error)
        evaluation[path]["base_reward_error"].append(base_reward_error)

        # value_error, policy_error = compute_value_error(run_data)
        # evaluation[path]["value_error"].append(value_error)
        # evaluation[path]["policy_error"].append(policy_error)

    for path, e in evaluation.items():
        Config.save_eval(path, e)


def compute_reward_error(run_data: RunData):

    truth = run_data.solution
    sampled = run_data.reward_samples

    sampled_fairness = [s.fairness_levels for s in sampled]
    sampled_base_rewards = [s.reward_func for s in sampled]
    true_fairness = [truth.fairness_levels for _ in range(len(sampled))]
    tile_base_rewards = [truth.reward_func for _ in range(len(sampled))]

    fairness_errors = tree.map_structure(
        lambda a, b: np.abs(a - b),
        sampled_fairness,
        true_fairness
    )
    
    base_reward_errors = tree.map_structure(
        lambda a, b: np.abs(a - b),
        sampled_base_rewards,
        tile_base_rewards
    )

    return fairness_errors, base_reward_errors


def eval_sample(inp):
    mg_config, truth, true_values, true_equilibriums, s = inp
    d = Demonstrations(
        mg_config,
        gamma,
        lambda_reg
    )
    d.get_demos(0, s, compute_values=False)
    values = d.get_value_for(truth)

    value_error = tree.map_structure(
        lambda t, s: np.abs(t-s),
        true_values,
        values
    )

    policy_error = tree.map_structure(
        lambda t, s: np.sum(np.abs(t-s)),
        true_equilibriums,
        d.equilibriums
        )

    return np.sum(tree.flatten(value_error)), np.sum(tree.flatten(policy_error))


def compute_value_error(run_data: RunData):

    truth = run_data.solution
    sampled = run_data.reward_samples
    
    # sample some rewards, compute optimal policies, compare policies and their values

    x = Demonstrations(
        run_data.mg_config,
        gamma,
        lambda_reg
    )
    x.get_demos(0, truth, compute_values=True)

    true_values = x.values.copy()
    true_equilibriums = x.equilibriums.copy()

    samples = np.random.choice(len(sampled), 50)

    inputs = [[run_data.mg_config, truth, true_values, true_equilibriums, sampled[idx]] for idx in samples]
    processes = 1 # mp.cpu_count()
    with mp.Pool(processes=processes) as pool:
        results = pool.map(eval_sample, inputs)

    value_errors = [v for v, _ in results]
    policy_errors = [p for _, p in results]

    return value_errors, policy_errors


def main():
    parser = argparse.ArgumentParser(description="Posterior sampling and evaluation.")

    parser.add_argument(
        '-s', '--sampling',
        type=str,
        choices=['MH', 'ULA'],
        required=True,
        help='Sampling algorithm to use: "MH" (Metropolis-Hastings) or "ULA" (Unadjusted Langevin Algorithm)'
    )

    parser.add_argument(
        '-p', '--posterior',
        type=str,
        choices=['DRP', 'PORP'],
        required=True,
        help='Posterior type to use: "DRP" or "PORP"'
    )

    parser.add_argument(
        '-g', '--gap_function',
        type=str,
        choices=['NIG', 'MNIG', 'PSG'],
        default='NIG',
        help='Gap function to use for PORP.'
    )

    parser.add_argument(
        '-n', '--num_seeds',
        type=int,
        default=0,
        help='Number of seeds to run the experiment with. Set this to a positive number of seeds, OR pick a seed using --seed'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed to run the experiment for'
    )

    parser.add_argument(
        '-t', '--num_traj',
        type=int,
        default=1,
        help='Number of trajectories per pairing.'
    )

    parser.add_argument(
        '--pool_size',
        type=int,
        default=3,
        help='Number of agents in the pool.'
    )

    parser.add_argument(
        '-e', '--eval',
        action="store_true",
        help='Runs evaluation instead of computation.'
    )

    args = parser.parse_args()

    if args.num_seeds > 0:
        configs = [
            Config(
                seed=i,
                num_trajectories_per_pairing=args.num_traj,
                pool_size=args.pool_size,
                sampling_algo=args.sampling,
                sampling_method=args.posterior,
                gap_function=None if args.posterior == "DRP" else args.gap_function,
                prior="var" if args.gap_function in ("NIG", "MNIG") else None
            )
            for i in range(args.num_seeds)
        ]

    else:
        configs = [Config(
                seed=args.seed,
                num_trajectories_per_pairing=args.num_traj,
                pool_size=args.pool_size,
                sampling_algo=args.sampling,
                sampling_method=args.posterior,
                gap_function=None if args.posterior == "DRP" else args.gap_function,
                prior="var" if args.gap_function in ("NIG", "MNIG") else None
        )]

    if args.eval:
        evaluate_rewards(configs)
    else:
        for config in configs:
            run_for_config(config)

if __name__ == '__main__':
    main()
