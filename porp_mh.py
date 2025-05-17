from typing import Tuple, List

import numpy as np
import tree

from irl_types import Demo
from demos import Demonstrations
from games.tf_regularised_mdp_solver import compute_batch_q_tf, compute_batch_best_response_tf, qre_batch_value_gap_tf, \
    qre_batch_max_value_gap_tf
from games.tf_regularised_mdp_solver import qre_batch_kl_gap_sweep_tf
from irl_types import MHConfig
from tf_models import SigmoidParametrisedReward, SoftmaxPolicies
import tensorflow as tf
from tqdm import tqdm


def get_epochs(
        batch: Demo,
        num_epochs: int,
        num_minibatches: int = 1,
        shuffle_minibatches: bool = False,
        axis=1,
        batch_size: int = None,
        **kwargs
):
    if isinstance(batch, Demo):
        B = batch.done.shape[axis]
    else:
        B = batch_size
    ordering = np.arange(B)
    minibatch_indices = np.split(ordering, num_minibatches)
    minibatch_size = B // num_minibatches


    if B % minibatch_size != 0:
        raise ValueError(
            f"The minibatch ({minibatch_size}) size must divide the full batch size ({B})."
        )
    for k in range(num_epochs):

        if shuffle_minibatches:
            ordering = np.arange(B)
            np.random.shuffle(minibatch_indices)
            minibatch_indices = np.split(ordering, num_minibatches)

        for indices in minibatch_indices:
            minibatch = tree.map_structure(
                lambda b: np.take(b, indices, axis=axis),
                batch
            )
            yield minibatch


def try_unpack_tf(x):
    if hasattr(x, "numpy"):
        x = x.numpy()
    return x


def mean_metrics(
        metrics: dict,
        new_metrics: dict,
        count: int
) -> Tuple[dict, int]:
    if metrics is None:
        return tree.map_structure(
            try_unpack_tf,
            new_metrics
        ), count + 1

    return tree.map_structure(
        lambda old, new: count * old / (count + 1) + try_unpack_tf(new) / (count + 1),
        metrics,
        new_metrics
    ), count + 1

def mean_posterior(
        samples
):
    unpacked = tree.map_structure(
        try_unpack_tf,
        samples
    )
    return tree.map_structure(
        lambda *s: np.mean(s, axis=0),
        *unpacked
    )


def porp_mh_policy_step(
        demos: Demonstrations,
        policy_model: SoftmaxPolicies,
        mh_config: MHConfig,
):
    time_major_batch = Demo.merge(demos.demonstrations).batched()

    samples = []
    loglikelihoods = []

    last_loglikelihood = -1e8
    last_sample = None
    last_parameters = None
    step_size = mh_config.step_size
    n = 0
    with tqdm(total=mh_config.num_samples, desc="Sampling from Policy Posterior", dynamic_ncols=True) as pbar:
        # its called minibatch, but we do full batches.
        for k, minibatch in enumerate(get_epochs(
                time_major_batch,
                mh_config.num_samples,
        )):

            loglikelihood = _eval_policy(
                policy_model,
                demos.combinations,
                **minibatch._asdict(),
            )
            loglikelihoods.append(loglikelihood)

            logq = loglikelihood - last_loglikelihood

            if np.log(np.random.random()) < logq:
                last_sample = policy_model.get_values()
                last_parameters = policy_model.parameters()
                last_loglikelihood = loglikelihood
                n += 1
            else:
                policy_model.set_parameters(last_parameters)

            samples.append(last_sample)

            acceptance_rate = n / (k + 1)
            if acceptance_rate > 0.25:
                step_size = min(mh_config.max_step_size, step_size + mh_config.increment_size)
            elif acceptance_rate < 0.21:
                step_size = max(mh_config.min_step_size, step_size - mh_config.increment_size)

            pbar.set_postfix({
                "Log Likelihood": f"{loglikelihood:.4e}",
                "Acceptance rate": f"{acceptance_rate:.4e}",
                "Step size": f"{step_size:.3e}",
            })
            pbar.update(1)

            policy_model.walk(step_size)

    num_warmup_samples = round(mh_config.warmup_frac * len(samples))
    samples = samples[num_warmup_samples:]
    return mean_posterior(samples), samples, loglikelihoods

@tf.function(
    #jit_compile=True
)
def _eval_policy(
        policy: SoftmaxPolicies,
        combinations,
        *,
        agent_1_idx,
        agent_2_idx,
        state,
        action_1,
        action_2,
        ** kwargs
):

    state_actions_1 = tf.stack([state, action_1], axis=-1)
    state_actions_2 = tf.stack([state, action_2], axis=-1)

    def process_pair(pair):
        i, j, k = pair[0], pair[1], pair[2]

        mask = tf.logical_and(tf.equal(agent_1_idx, i), tf.equal(agent_2_idx, j))

        masked_state_actions_1 = tf.boolean_mask(state_actions_1, mask)
        masked_state_actions_2 = tf.boolean_mask(state_actions_2, mask)

        logp_1 = policy.get_logp(2 * k, masked_state_actions_1)
        logp_2 = policy.get_logp(2 * k +1, masked_state_actions_2)

        x = tf.reduce_sum(logp_1 + logp_2)
        return x

    loglikelihood = tf.map_fn(process_pair, combinations,
                                         dtype=tf.float32
                                         )

    return tf.reduce_sum(loglikelihood)



def porp_mh_reward_step(
        demos: Demonstrations,
        reward_model: SigmoidParametrisedReward,
        policy_samples: List[np.ndarray],
        mh_config: MHConfig,
        prior_type: None | str,
        gap_function: str
):

    policy_samples = np.stack(policy_samples)
    N = len(policy_samples)

    policy_samples = policy_samples.reshape((N, demos.num_combinations, 2,
                                             demos.base_mdp.num_states, demos.base_mdp.num_actions))

    samples = []
    loglikelihoods = []

    last_loglikelihood = -1e8
    last_sample = None
    last_parameters = None
    step_size = mh_config.step_size
    n = 0

    with tqdm(total=mh_config.num_samples, desc="Sampling from Reward Posterior", dynamic_ncols=True) as pbar:
        for k, minibatch in enumerate(get_epochs(
                policy_samples,
                num_epochs=mh_config.num_samples,
                axis=0,
                batch_size=N,
        )):

            loglikelihood = _eval_reward(
                reward_model,
                demos.base_mdp,
                demos.combinations,
                minibatch,
                gamma=demos.gamma,
                prior_type=prior_type,
                gap_function=gap_function,
            )

            loglikelihoods.append(loglikelihood)

            logq = loglikelihood - last_loglikelihood
            if np.log(np.random.random()) < logq:
                last_sample = reward_model.get_values()
                last_loglikelihood = loglikelihood
                last_parameters = reward_model.parameters()
                n += 1
            else:
                reward_model.set_parameters(last_parameters)

            samples.append(last_sample)

            acceptance_rate = n / (k + 1)

            if acceptance_rate > 0.25:
                step_size = min(mh_config.max_step_size, step_size + mh_config.increment_size)
            elif acceptance_rate < 0.21:
                step_size = max(mh_config.min_step_size, step_size - mh_config.increment_size)


            pbar.set_postfix({
                "Log Likelihood": f"{loglikelihood:.4e}",
                "Acceptance rate": f"{acceptance_rate:.4e}",
                "Step size": f"{step_size:.4e}",
            })
            pbar.update(1)

            reward_model.walk(step_size)

    num_warmup_samples = round(mh_config.warmup_frac * len(samples))
    samples = samples[num_warmup_samples:]
    loglikelihoods = loglikelihoods[num_warmup_samples:]
    return mean_posterior(samples), samples, loglikelihoods


@tf.function
def _eval_reward(
        reward_model: SigmoidParametrisedReward,
        mdp,
        combinations,
        policy_samples, # [n, n_combs, agent_idx, s, a]
        *,
        gamma,
        prior_type,
        gap_function,
        ** kwargs
):

    # TODO: probably can process whole batch instead of mapping over combinations ?
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

            # qre_gap = qre_batch_kl_gap_tf_for_reg(Q1, Q2, pi_1, pi_2, lambda_reg)
            # loglikelihoods = - qre_gap * 100.
            qre_gap, optimality_probs = qre_batch_kl_gap_sweep_tf(Q1, Q2, pi_1, pi_2, num_regs=20)
            likelihoods = tf.reduce_sum(tf.math.exp(-qre_gap * 100.) * tf.expand_dims(optimality_probs, axis=1), axis=0)
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
                                                      mdp.transition_matrix, mdp.initial_state_dist, gamma) * 50.
            else:
                loglikelihoods = - qre_batch_max_value_gap_tf(V_1, V_2, V_br1, V_br2) * 25.
            loglikelihoods = loglikelihoods

        return loglikelihoods + reward_model.log_prior(prior_type)

    loglikelihood = tf.map_fn(process_pair, combinations,
                                      dtype=tf.float32
                                      )
    pair_logproduct = tf.reduce_sum(loglikelihood, axis=0)

    return tf.math.reduce_logsumexp(pair_logproduct)