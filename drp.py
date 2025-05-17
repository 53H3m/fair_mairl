from typing import Tuple

import numpy as np
import tree

from irl_types import Demo
from sgld.SGLD import SGLDConfig
from demos import Demonstrations
from games.tf_regularised_mdp_solver import compute_soft_nash_equilibrium_tf
from games.tf_regularised_mdp_solver import compute_soft_nash_equilibrium_sweep_tf
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


def drp_ula(
        demos: Demonstrations,
        reward_model: SigmoidParametrisedReward,
        sgld_config: SGLDConfig,
):
    time_major_batch = Demo.merge(demos.demonstrations).batched()
    N = time_major_batch.done.shape[1]
    num_minibatch = sgld_config.num_epochs * sgld_config.num_minibatches

    frac = 1 / N

    samples = []
    loglikelihoods = []
    with tqdm(total=num_minibatch, desc="Sampling from the Reward Posterior", dynamic_ncols=True) as pbar:
        for minibatch in get_epochs(
                time_major_batch,
                **sgld_config._asdict()
        ):

            minibatch_metrics = _train(
                reward_model,
                demos.base_mdp.transition_matrix,
                demos.combinations,
                frac,
                **minibatch._asdict(),
                gamma=demos.gamma,
            )

            samples.append(reward_model.get_values())

            print(reward_model.to_irl_params())
            loglikelihoods.append(minibatch_metrics["log_likelihood"])

            pbar.set_postfix({
                "Log Likelihood": f"{loglikelihoods[-1]:.4e}",
                #"Grad Norm": f"{np.linalg.norm(minibatch_metrics['grads']):.4e}",
                "Step size": f"{reward_model.optimiser.curr_step_size().numpy():.4e}",
            })
            pbar.update(1)

    num_warmup_samples = round(sgld_config.warmup_frac * len(samples))
    samples = samples[num_warmup_samples:]
    loglikelihoods = loglikelihoods[num_warmup_samples:]
    return mean_posterior(samples), samples, loglikelihoods

@tf.function(
)
def _train(
        reward_model: SigmoidParametrisedReward,
        transition_matrix,
        combinations,
        frac,
        *,
        agent_1_idx,
        agent_2_idx,
        state,
        action_1,
        action_2,
        gamma,
        ** kwargs
):

    with tf.GradientTape() as tape:

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
        loss = -tf.reduce_sum(loglikelihood) * frac

    grads = tape.gradient(loss, reward_model.trainable_variables)
    reward_model.optimiser.apply(grads, reward_model.trainable_variables)
    reward_model.clip()

    return {"log_likelihood": -loss / frac, "grads": grads}