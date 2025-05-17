import numpy as np
from tqdm import tqdm
import tensorflow as tf

from demos import Demonstrations
from irl_types import IRLParams, Config
from games.tf_regularised_mdp_solver import compute_batch_q_tf, qre_batch_kl_gap_sweep_tf, compute_batch_best_response_tf, \
    qre_batch_max_value_gap_tf, qre_batch_value_gap_tf
from porp import porp_ula_policy_step
from sgld.SGLD import RMSPropSGLDConfig, RMSPropSGLD
from tf_models import StaticReward, SoftmaxPolicies

# MDP config we want to eval Z for.
alpha = 1.2
gamma = 0.9
lambda_reg = 0.1
episode_length = 100
num_actions = 3
num_states = 5
seed = 0
num_trajectories_per_pairing = 1
gap_function = "NIG"

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

def compute_z(demos):
    policies = SoftmaxPolicies(
        num_policies=demos.num_combinations * 2,
        num_states=num_states,
        num_actions=num_actions,
        optim=RMSPropSGLD(**policy_ula_config._asdict())
    )

    # sample policies
    _, sampled_policies, _ = porp_ula_policy_step(
        demos=demos,
        policy_model=policies,
        sgld_config=policy_ula_config
    )

    Zs = []
    reward = StaticReward(demos.agent_pool_size, demos.base_mdp.observation_space, demos.base_mdp.action_space)
    n_z = 20_000
    n_pi = 200
    policy_indices = np.random.choice(len(sampled_policies), n_pi)
    policies = np.array(sampled_policies)[policy_indices]
    policies = policies.reshape((n_pi, demos.num_combinations, 2,
                                             demos.base_mdp.num_states, demos.base_mdp.num_actions))

    with tqdm(total=n_z, desc="Computing Z", dynamic_ncols=True) as pbar:
        for i in range(n_z):
            density = reward.sample_gaussian(truth, std=0.4)
            exponents = _compute_exponents(
                reward,
                demos.base_mdp,
                demos.combinations,
                policies,
                gamma=demos.gamma,
                gap_function=gap_function,
            )
            Zs.append(exponents.numpy() / density)

            pbar.set_postfix({
                "Log Likelihood": np.max(exponents),
            })
            pbar.update(1)

    Zs_estimates = np.cumsum(np.stack(Zs), axis=0)
    np.save(f"Z_{seed}_{num_states}_{num_actions}.npy", Zs_estimates)


@tf.function
def _compute_exponents(
        reward_model: StaticReward,
        mdp,
        combinations,
        policy_samples,
        *,
        gamma,
        gap_function,
        ** kwargs
):

    def process_pair(pair):
        i, j, k = pair[0], pair[1], pair[2]
        R_1 = reward_model.compute_R_1(i)
        R_2 = reward_model.compute_R_2(j)

        pi_1 = policy_samples[:, k, 0]
        pi_2 = policy_samples[:, k, 1]

        if gap_function == "PSG":
            Q1, Q2, _, _ = compute_batch_q_tf(
                mdp.transition_matrix,
                R_1,
                R_2,
                pi_1,
                pi_2,
                gamma,
            )

            qre_gap, optimality_probs = qre_batch_kl_gap_sweep_tf(Q1, Q2, pi_1, pi_2, mdp.transition_matrix, mdp.initial_state_dist, gamma, num_regs=20)
            likelihoods = tf.reduce_sum(tf.math.exp(-qre_gap) * tf.expand_dims(optimality_probs, axis=1), axis=0)
            loglikelihoods = tf.math.log(likelihoods + 1e-16)

        elif gap_function == "NIG" or gap_function == "MNIG":
            _, _, V_1, V_2 = compute_batch_q_tf(
                mdp.transition_matrix,
                R_1,
                R_2,
                pi_1,
                pi_2,
                gamma,
            )

            pi_br1, _, V_br1 = compute_batch_best_response_tf(
                mdp.transition_matrix,
                R_1,
                pi_2,
                gamma,
            )

            pi_br2, _, V_br2 = compute_batch_best_response_tf(
                mdp.transition_matrix,
                reward_model.compute_R_1(j),
                pi_1,
                gamma,
            )

            if gap_function == "NIG":
                loglikelihoods = - qre_batch_value_gap_tf(pi_1, pi_2, pi_br1, pi_br2, V_1, V_2, V_br1, V_br2,
                                                      mdp.transition_matrix, mdp.initial_state_dist, gamma) * 2
            else:
                loglikelihoods = - qre_batch_max_value_gap_tf(V_1, V_2, V_br1, V_br2) * 2
            loglikelihoods = loglikelihoods + reward_model.log_prior()

        return loglikelihoods

    loglikelihood = tf.map_fn(process_pair, combinations,
                                      dtype=tf.float32
                                      )
    pair_logproduct = tf.reduce_sum(loglikelihood, axis=0)

    return tf.math.exp(pair_logproduct)


if __name__ == '__main__':


    reward_func = np.zeros((num_states, num_actions), dtype=np.float32)
    reward_func[num_states - 1] = 1

    # construct some arbitrary true reward function.
    truth = IRLParams(
        fairness_levels=(0.2, 0., 1.),
        reward_func=reward_func,
    )

    initial_state_dist = np.zeros(num_states)
    initial_state_dist[0] = 1

    partial_mg_config = {
        "seed"              : seed,
        "num_states"        : num_states,
        "num_actions"       : num_actions,
        "alpha"             : alpha,
        "episode_length"    : episode_length,
        "initial_state_dist": initial_state_dist,
    }

    demos = Demonstrations(
        partial_mg_config,
        gamma,
        lambda_reg
    )

    demos.get_demos(
        num_trajectories_per_pairing,
        truth
    )

    compute_z(demos)

