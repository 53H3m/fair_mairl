from typing import Tuple, List

import numpy as np
import tree
from matplotlib import pyplot as plt

from irl_types import Demo, IRLParams
from sgld.SGLD import SGLDConfig
from demos import Demonstrations
from games.tf_regularised_mdp_solver import compute_batch_q_tf, compute_batch_best_response_tf, qre_batch_value_gap_tf, \
    qre_batch_kl_gap_tf_for_reg, qre_batch_max_value_gap_tf
from games.tf_regularised_mdp_solver import qre_batch_kl_gap_sweep_tf
from tf_models import SigmoidParametrisedReward, SoftmaxPolicies, StaticReward
import tensorflow as tf
from tqdm import tqdm


def get_epochs(
        batch: Demo,
        num_epochs: int,
        num_minibatches: int,
        shuffle_minibatches: bool,
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
            np.random.shuffle(ordering)
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


def porp_ula_policy_step(
        demos: Demonstrations,
        policy_model: SoftmaxPolicies,
        sgld_config: SGLDConfig,
):
    time_major_batch = Demo.merge(demos.demonstrations).batched()
    metrics = None
    N = time_major_batch.done.shape[1]
    num_minibatch = sgld_config.num_epochs * sgld_config.num_minibatches

    frac = num_minibatch

    samples = []
    loglikelihoods = []

    with tqdm(total=num_minibatch, desc="Sampling from Policy Posterior", dynamic_ncols=True) as pbar:
        for minibatch in get_epochs(
                time_major_batch,
                **sgld_config._asdict(),
        ):

            minibatch_metrics = _train_policy(
                policy_model,
                demos.combinations,
                frac,
                **minibatch._asdict(),
            )

            samples.append(policy_model.get_values())
            loglikelihoods.append(minibatch_metrics["log_likelihood"])

            pbar.set_postfix({
                "Log Likelihood": f"{loglikelihoods[-1]:.4e}",
                #"Grad Norm": f"{np.linalg.norm(minibatch_metrics['grads']):.4e}",
                "Step size": f"{policy_model.optimiser.curr_step_size().numpy():.4e}",
            })
            pbar.update(1)

    num_warmup_samples = round(sgld_config.warmup_frac * len(samples))
    samples = samples[num_warmup_samples:]
    loglikelihoods = loglikelihoods[num_warmup_samples:]
    return mean_posterior(samples), samples, loglikelihoods

@tf.function(
    #jit_compile=True
)
def _train_policy(
        policy: SoftmaxPolicies,
        combinations,
        frac,
        *,
        agent_1_idx,
        agent_2_idx,
        state,
        action_1,
        action_2,
        ** kwargs
):

    with tf.GradientTape() as tape:

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
        loss = -tf.reduce_sum(loglikelihood) * frac

    grads = tape.gradient(loss, policy.trainable_variables)
    policy.optimiser.apply(grads, policy.trainable_variables)

    return {"log_likelihood": -loss / frac, "grads": grads}


def porp_ula_reward_step(
        demos: Demonstrations,
        reward_model: SigmoidParametrisedReward,
        policy_samples: List[np.ndarray],
        sgld_config: SGLDConfig,
        prior_type: None | str,
        gap_function: str
):

    policy_samples = np.stack(policy_samples)
    N = len(policy_samples)

    policy_samples = policy_samples.reshape((N, demos.num_combinations, 2,
                                             demos.base_mdp.num_states, demos.base_mdp.num_actions))

    num_minibatch = sgld_config.num_epochs * sgld_config.num_minibatches

    frac = num_minibatch

    samples = []
    loglikelihoods = []

    with tqdm(total=num_minibatch, desc="Sampling from Reward Posterior", dynamic_ncols=True) as pbar:
        for minibatch in get_epochs(
                policy_samples,
                axis=0,
                batch_size=N,
                **sgld_config._asdict()
        ):

            minibatch_metrics = _train_reward(
                reward_model,
                demos.base_mdp,
                demos.combinations,
                minibatch,
                frac,
                gamma=demos.gamma,
                prior_type=prior_type,
                gap_function=gap_function,
            )

            samples.append(reward_model.get_values())
            loglikelihoods.append(minibatch_metrics["log_likelihood"])

            pbar.set_postfix({
                "Log Likelihood": f"{loglikelihoods[-1]:.4e}",
                "Mean Gap": f"{minibatch_metrics['mean_gap']:.2e}",
            })
            pbar.update(1)

    num_warmup_samples = round(sgld_config.warmup_frac * len(samples))
    samples = samples[num_warmup_samples:]
    loglikelihoods = loglikelihoods[num_warmup_samples:]
    return mean_posterior(samples), samples, loglikelihoods

@tf.function
def _train_reward(
        reward_model: SigmoidParametrisedReward,
        mdp,
        combinations,
        policy_samples, # [n, n_combs, agent_idx, s, a]
        frac,
        *,
        gamma,
        prior_type,
        gap_function,
        ** kwargs
):

    with tf.GradientTape() as tape:

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

                qre_gap, optimality_probs = qre_batch_kl_gap_sweep_tf(Q1, Q2, pi_1, pi_2, mdp.transition_matrix, mdp.initial_state_dist, gamma, num_regs=20)
                likelihoods = tf.reduce_sum(tf.math.exp(-qre_gap * 15.) * tf.expand_dims(optimality_probs, axis=1), axis=0)
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
        #
        pair_logproduct = tf.reduce_sum(loglikelihood, axis=0)
        mean_gap = tf.reduce_mean(-loglikelihood)
        loss = - tf.math.reduce_logsumexp(pair_logproduct) * frac

    grads = tape.gradient(loss, reward_model.trainable_variables)
    reward_model.optimiser.apply(grads, reward_model.trainable_variables)
    reward_model.clip()

    return {"log_likelihood": -loss / frac, "grads": grads, "mean_gap": mean_gap}