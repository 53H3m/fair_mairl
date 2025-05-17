from typing import NamedTuple

import sonnet as snt
import tensorflow as tf


class SGLD(snt.Optimizer):
    def __init__(self, eta_0=1e-2, alpha=0.55, noise_scale=1., **kwargs):
        """
        Stochastic Gradient Langevin Dynamics Optimizer with diminishing step size.

        Args:
            eta_0: Initial learning rate.
            alpha: Decay exponent (0.5 < alpha <= 1).
        """
        super().__init__(name="SGLD")
        self.eta_0 = eta_0
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.step = tf.Variable(0, dtype=tf.float32, trainable=False)

    def curr_step_size(self):
        return self.eta_0 / tf.pow(1.0 + self.step, self.alpha)

    def apply(self, updates, parameters):
        """Applies SGLD update rule with decaying step size."""
        eta_t = self.curr_step_size()

        for param, grad in zip(parameters, updates):
            if grad is not None:
                noise = tf.random.normal(shape=tf.shape(param), stddev=tf.sqrt(eta_t) * self.noise_scale)
                param.assign_sub(0.5 * eta_t * grad + noise)

        self.step.assign_add(1)  # Update step counter


class RMSPropSGLD(snt.Optimizer):
    def __init__(self, eta_0=1e-2, alpha=0.55, noise_scale=1.0, beta=0.99, epsilon=1e-8, **kwargs):
        """
        RMSProp-based Stochastic Gradient Langevin Dynamics (SGLD) Optimizer.

        Args:
            eta_0: Initial learning rate.
            alpha: Decay exponent for step size (0.5 < alpha <= 1).
            noise_scale: Scale of Langevin noise.
            beta: Exponential decay rate for moving average of squared gradients.
            epsilon: Small constant to prevent division by zero.
        """
        super().__init__(name="RMSPropSGLD")
        self.eta_0 = eta_0
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.beta = beta
        self.epsilon = epsilon
        self.step = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.accumulated_sq_grads = {}  # Dictionary to track per-parameter squared gradient averages

    def curr_step_size(self):
        """Returns the current step size following the SGLD decay schedule."""
        return self.eta_0 / tf.pow(1.0 + self.step, self.alpha)

    def init(self, params):
        # Initialize moving average of squared gradients
        for param in params:
            if param.ref() not in self.accumulated_sq_grads:
                self.accumulated_sq_grads[param.ref()] = tf.Variable(tf.zeros_like(param), trainable=False)

    def apply(self, updates, parameters):
        """Applies RMSProp-SGLD update rule with adaptive step size."""
        eta_t = self.curr_step_size()

        for param, grad in zip(parameters, updates):
            if grad is not None:

                v_t = self.accumulated_sq_grads[param.ref()]

                # Update moving average of squared gradients
                v_t.assign(self.beta * v_t + (1.0 - self.beta) * tf.square(grad))

                # Adaptive learning rate using RMSProp-style adjustment
                adj_eta_t = eta_t / (tf.sqrt(v_t) + self.epsilon)

                # Add Langevin noise
                noise = tf.random.normal(shape=tf.shape(param), stddev=tf.sqrt(adj_eta_t) * self.noise_scale)

                # Apply update
                param.assign_sub(0.5 * adj_eta_t * grad + noise)

        # Increment step count
        self.step.assign_add(1)


class SGLDConfig(NamedTuple):
    eta_0: float
    alpha: float
    noise_scale: float
    num_epochs: int
    num_minibatches: int
    shuffle_minibatches: bool = False
    warmup_frac: float = 0.25


class RMSPropSGLDConfig(NamedTuple):
    eta_0: float
    alpha: float
    noise_scale: float
    beta: float
    epsilon: float
    num_epochs: int
    num_minibatches: int
    shuffle_minibatches: bool = False
    warmup_frac: float = 0.25

