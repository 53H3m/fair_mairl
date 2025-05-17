from typing import Tuple

import gymnasium
import tensorflow as tf
from scipy.stats import multivariate_normal, truncnorm
from tf_keras.activations import sigmoid
import numpy as np

from tensorflow.math import reduce_logsumexp
import sonnet as snt
from irl_types import IRLParams


class StaticReward:
    def __init__(
            self,
            n_agents: int,
            observation_space: gymnasium.Space,
            action_space: gymnasium.Space,
            reward_bounds: Tuple[float, float] = (0., 1.),
            fairness_bounds: Tuple[float, float] = (0., 1.),
    ):

        self.observation_space = observation_space
        self.action_space = action_space
        self.fairness_bounds = fairness_bounds
        self.reward_bounds = reward_bounds
        self.n_agents = n_agents



    def get_values(self):
        return self.fairness_levels, self.reward_parameters

    def to_irl_params(self):
        return convert_to_irl_params(*self.get_values())

    def sample_gaussian(self, mean: IRLParams, std: float):

        fairness_np =  np.array(mean.fairness_levels)

        fairness_lower = (0 - fairness_np) / std
        fairness_upper = (1 - fairness_np) / std
        fairness_levels = truncnorm.rvs(fairness_lower, fairness_upper, loc=fairness_np, scale=std)

        reward_lower = (0 - mean.reward_func ) / std
        reward_upper = (1 - mean.reward_func ) / std
        reward_func = truncnorm.rvs(reward_lower, reward_upper, loc=mean.reward_func, scale=std)

        self.fairness_levels = tf.Variable(fairness_levels, dtype=tf.float32, name="fairness_levels")
        self.reward_parameters = tf.Variable(reward_func, dtype=tf.float32, name="reward_parameters")

        concat_lower = np.concatenate([fairness_lower, reward_lower.flatten()], axis=0)
        concat_upper = np.concatenate([fairness_upper, reward_upper.flatten()], axis=0)
        concat_mean = np.concatenate([fairness_np, mean.reward_func.flatten()], axis=0)
        concat_sample = np.concatenate([fairness_levels, reward_func.flatten()], axis=0)
        densities = truncnorm.pdf(concat_sample, concat_lower, concat_upper, loc=concat_mean, scale=std)

        return np.prod(densities) # product over the dimensions of the multivariate (vectory of densities)

    def sample_uniform(self):
        self.fairness_levels = tf.Variable(np.random.uniform(*self.fairness_bounds, self.n_agents), dtype=tf.float32, name="fairness_levels")
        self.reward_parameters = tf.Variable(np.random.uniform(*self.reward_bounds, (self.observation_space.n, self.action_space.n)),
                                             dtype=tf.float32, name="reward_parameters")

    def volume(self):

        db_f = self.fairness_bounds[1] - self.fairness_bounds[0]
        db_r = self.reward_bounds[1] - self.reward_bounds[0]

        return (db_f ** self.n_agents) * (db_r ** np.prod(self.reward_parameters.shape))

    def compute_R_1(
            self,
            agent_1_idx,
    ):
        fairness = tf.gather(self.fairness_levels, agent_1_idx)

        r1 = tf.expand_dims(self.reward_parameters, axis=2)
        r2 = tf.expand_dims(self.reward_parameters, axis=1)

        # TODO: normalise
        return r1 + r2 * fairness

    def compute_R_2(
            self,
            agent_2_idx,
    ):
        fairness = tf.gather(self.fairness_levels, agent_2_idx)

        r1 = tf.expand_dims(self.reward_parameters, axis=1)
        r2 = tf.expand_dims(self.reward_parameters, axis=2)
        return r1 + r2 * fairness

    def log_prior(self):
        alpha = 0.3
        var = tf.math.reduce_variance(self.reward_parameters)
        return alpha * var


class SigmoidParametrisedReward(snt.Module):

    def __init__(
            self,
            num_agents: int,
            num_states: int,
            num_actions: int,
            optim: snt.Optimizer = None,
            reward_bounds: Tuple[float, float] = (0., 1.),
            fairness_bounds: Tuple[float, float] = (0., 1.),
            init_range = (-0.1, 0.1)
    ):
        super().__init__("reward_model")

        self.optimiser = optim

        self.reward_min = reward_bounds[0]
        self.reward_scale = reward_bounds[1] - reward_bounds[0]
        self.fairness_min = fairness_bounds[0]
        self.fairness_scale = fairness_bounds[1] - fairness_bounds[0]

        self.fairness_levels = tf.Variable(np.random.uniform(*init_range, num_agents), dtype=tf.float32, name="fairness_levels")
        self.reward_parameters = tf.Variable(np.random.uniform(*init_range, (num_states, num_actions)),
                                             dtype=tf.float32, name="reward_parameters")

        # self.fairness_levels = tf.Variable([-10., 0., 0.], dtype=tf.float32, name="fairness_levels")
        # self.reward_parameters = tf.Variable([[-10., -10.], [10., 10.]],
        #                                      dtype=tf.float32, name="reward_parameters")
        if optim is not None:
            self.optimiser.init(self.trainable_variables)

    def clip(self):
        self.fairness_levels.assign(tf.clip_by_value(self.fairness_levels, -7., 7.))
        self.reward_parameters.assign(tf.clip_by_value(self.reward_parameters, -7., 7.))

    def get_values(self):
        return sigmoid(self.fairness_levels) * self.fairness_scale + self.fairness_min, sigmoid(self.reward_parameters) * self.reward_scale + self.reward_min

    def to_irl_params(self):
        return convert_to_irl_params(*self.get_values())

    def compute_R_1(
            self,
            agent_1_idx,
    ):
        fairness = sigmoid(tf.gather(self.fairness_levels, agent_1_idx)) * self.fairness_scale + self.fairness_min

        r1 = sigmoid(tf.expand_dims(self.reward_parameters, axis=2)) * self.reward_scale + self.reward_min
        r2 = sigmoid(tf.expand_dims(self.reward_parameters, axis=1)) * self.reward_scale + self.reward_min
        return r1 + r2 * fairness

    def compute_R_2(
            self,
            agent_2_idx,
    ):
        fairness = sigmoid(tf.gather(self.fairness_levels, agent_2_idx)) * self.fairness_scale + self.fairness_min

        r1 = sigmoid(tf.expand_dims(self.reward_parameters, axis=1)) * self.reward_scale + self.reward_min
        r2 = sigmoid(tf.expand_dims(self.reward_parameters, axis=2)) * self.reward_scale + self.reward_min
        return r1 + r2 * fairness

    def log_prior(self, prior_type):
        if prior_type == "var":
            r = sigmoid(self.reward_parameters)
            var = tf.math.reduce_variance(r)
            if var > 5E-2:
                return -100.
            else:
                return 0.
        elif prior_type == "L1":
            r = sigmoid(self.reward_parameters)
            return -tf.reduce_mean(r) * 2.
        return 0.

    def parameters(self):
        return (self.fairness_levels.value(), self.reward_parameters.value())

    def set_parameters(self, p):
        f, r = p
        self.fairness_levels.assign(f)
        self.reward_parameters.assign(r)

    def walk(self, step_size: float):

        self.fairness_levels.assign(
            self.fairness_levels + tf.random.normal(self.fairness_levels.shape, 0, step_size)
        )
        self.reward_parameters.assign(
            self.reward_parameters + tf.random.normal(self.reward_parameters.shape, 0, step_size)

        )
        self.clip()


class StateSwapReward(SigmoidParametrisedReward):

    def __init__(
            self,
            num_agents: int,
            num_states: int,
            num_actions: int,
            state_swaps: list,
            optim: snt.Optimizer = None,
            reward_bounds: Tuple[float, float] = (0., 1.),
            fairness_bounds: Tuple[float, float] = (0., 1.),
            init_range=(-0.1, 0.1),
    ):
        super().__init__(
            num_agents, num_states, num_actions, optim, reward_bounds, fairness_bounds, init_range
        )
        self.state_swaps = state_swaps


    def compute_R_1(
            self,
            agent_1_idx,
    ):
        fairness = sigmoid(tf.gather(self.fairness_levels, agent_1_idx)) * self.fairness_scale + self.fairness_min

        swapped_reward_parameters = tf.gather(self.reward_parameters, self.state_swaps)
        r1 = sigmoid(tf.expand_dims(self.reward_parameters, axis=2)) * self.reward_scale + self.reward_min
        r2 = sigmoid(tf.expand_dims(swapped_reward_parameters, axis=1)) * self.reward_scale + self.reward_min
        return r1 + r2 * fairness

    def compute_R_2(
            self,
            agent_2_idx,
    ):
        fairness = sigmoid(tf.gather(self.fairness_levels, agent_2_idx)) * self.fairness_scale + self.fairness_min

        swapped_reward_parameters = tf.gather(self.reward_parameters, self.state_swaps)
        r1 = sigmoid(tf.expand_dims(swapped_reward_parameters, axis=1)) * self.reward_scale + self.reward_min
        r2 = sigmoid(tf.expand_dims(self.reward_parameters, axis=2)) * self.reward_scale + self.reward_min
        return r1 + r2 * fairness


def convert_to_irl_params(fairness_levels, reward_func):
    return IRLParams(
        fairness_levels=tuple(list(fairness_levels.numpy())),
        reward_func=reward_func.numpy()
    )


class SoftmaxPolicies(snt.Module):

    def __init__(
            self,
            num_policies: int,
            num_states: int,
            num_actions: int,
            optim: snt.Optimizer = None,
            init_range = (-1., 1.)
    ):
        super().__init__(name="SoftmaxPolicy")

        self.optimiser = optim

        self.logits = tf.Variable(
            np.random.uniform(*init_range, (num_policies, num_states, num_actions)), dtype=tf.float32, name="logits"
        )

        self.optimiser.init(self.trainable_variables)


    def get_logp(self, policy_idx, state_action):
        # Extract the logits corresponding to the selected policy
        policy_logits = self.logits[policy_idx]  # Shape: [obs_n, action_n]

        # Gather logits for the selected states
        policy_probs = tf.nn.softmax(policy_logits, axis=-1)
        policy_probs = tf.gather_nd(policy_probs, state_action)  # Shape: [batch_size, action_n]

        # Compute log probability
        return tf.math.log(policy_probs + 1e-8)

    def get_values(self):
        return tf.nn.softmax(self.logits, axis=-1)

    def parameters(self):
        return self.logits.value()

    def set_parameters(self, p):
        self.logits.assign(p)

    def walk(self, step_size: float):

        self.logits.assign(
            self.logits + tf.random.normal(self.logits.shape, 0, step_size)
        )
