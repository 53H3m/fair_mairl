from typing import Tuple

import numpy as np
import tree

from irl_types import Demo

from demos import Demonstrations
from games.tf_regularised_mdp_solver import compute_soft_nash_equilibrium_tf, compute_soft_nash_equilibrium_sweep_tf
from sgld.SGLD import SGLDConfig
from irl_types import MHConfig
from tf_models import SigmoidParametrisedReward
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


def drp_mh(
        demos: Demonstrations,
        reward_model: SigmoidParametrisedReward,
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

    with tqdm(total=mh_config.num_samples, desc="Sampling from the Reward Posterior", dynamic_ncols=True) as pbar:
        for k, minibatch in enumerate(get_epochs(
                time_major_batch,
                mh_config.num_samples,
        )):

            loglikelihood = _train(
                reward_model,
                demos.base_mdp.transition_matrix,
                demos.combinations,
                **minibatch._asdict(),
                gamma=demos.gamma,
            )

            loglikelihoods.append(loglikelihood)

            logq = loglikelihood - last_loglikelihood

            if np.log(np.random.random()) < logq:
                last_sample = reward_model.get_values()
                last_parameters = reward_model.parameters()
                last_loglikelihood = loglikelihood
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
                "Step size": f"{step_size:.3e}",
            })
            pbar.update(1)

            reward_model.walk(step_size)

    num_warmup_samples = round(mh_config.warmup_frac * len(samples))
    samples = samples[num_warmup_samples:]
    loglikelihoods = loglikelihoods[num_warmup_samples:]
    return mean_posterior(samples), samples, loglikelihoods

@tf.function(
)
def _train(
        reward_model: SigmoidParametrisedReward,
        transition_matrix,
        combinations,
        *,
        agent_1_idx,
        agent_2_idx,
        state,
        action_1,
        action_2,
        gamma,
        ** kwargs
):


    state_actions_1 = tf.stack([state, action_1], axis=-1)
    state_actions_2 = tf.stack([state, action_2], axis=-1)

    def process_pair(pair):
        i, j = pair[0], pair[1]

        R_1 = reward_model.compute_R_1(i)
        R_2 = reward_model.compute_R_2(j)
        pi_a, pi_b, optimality_probs = compute_soft_nash_equilibrium_sweep_tf(
            transition_matrix,
            R_1,
            R_2,
            gamma,
            num_regs=20,
        )

        optimality_probs = tf.expand_dims(tf.expand_dims(optimality_probs, axis=-1), axis=-1)
        pi_a = tf.reduce_sum(pi_a * optimality_probs, axis=0)
        pi_b = tf.reduce_sum(pi_b * optimality_probs, axis=0)

        mask = tf.logical_and(tf.equal(agent_1_idx, i), tf.equal(agent_2_idx, j))
        masked_state_actions_1 = tf.boolean_mask(state_actions_1, mask)
        masked_state_actions_2 = tf.boolean_mask(state_actions_2, mask)

        log_pi_a = tf.math.log(tf.gather_nd(pi_a, masked_state_actions_1))
        log_pi_b = tf.math.log(tf.gather_nd(pi_b, masked_state_actions_2))

        return tf.reduce_sum(log_pi_a + log_pi_b)

    loglikelihood = tf.map_fn(process_pair, combinations, dtype=tf.float32)

    return tf.reduce_sum(loglikelihood)