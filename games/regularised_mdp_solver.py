import matplotlib.pyplot as plt
import numpy as np

from games.mdp import TwoPlayerMDP, RandomTwoPlayerMDP

def compute_soft_nash_equilibrium(
        mdp: TwoPlayerMDP,
        gamma=0.99,
        lambda_reg=0.1,
        f1=0.,
        f2=0.,
        tol=2e-5,
        damping=0.05,
        max_iters=5000):
    """
    Computes the entropy-regularized Nash equilibrium using soft value iteration.

    Args:
        mdp: TwoPlayerMDP instance.
        gamma: Discount factor.
        lambda_reg: Entropy regularization coefficient.
        f1: fairness level of player 1.
        f2: fairness level of player 2.
        tol: Convergence tolerance.
        max_iters: Maximum iterations.

    Returns:
        (policy_1, policy_2, V_1, V_2): Regularized Nash equilibrium policies and their utilities.
    """

    num_states = mdp.num_states
    num_actions = mdp.num_actions

    R_1 = mdp.construct_fair_rewards(f1, 1)
    R_2 = mdp.construct_fair_rewards(f2, 2).swapaxes(1, 2)


    V_1 = np.zeros(num_states)
    V_2 = np.zeros(num_states)

    policy_1 = np.ones((num_states, num_actions)) / num_actions
    policy_2 = np.ones((num_states, num_actions)) / num_actions

    for _ in range(max_iters):

        Q_1 = R_1 + gamma * np.sum(mdp.transition_matrix * V_1[np.newaxis, np.newaxis, np.newaxis, :], axis=-1)
        Q_2 = R_2 + gamma * np.sum(mdp.transition_matrix * V_2[np.newaxis, np.newaxis, np.newaxis, :], axis=-1)

        EQ_1 = np.sum(Q_1 * policy_2[:, None], axis=2)
        EQ_2 = np.sum(Q_2 * policy_1[:, :, None], axis=1)

        npolicy_1 = softmax(EQ_1 / lambda_reg, axis=1)
        npolicy_2 = softmax(EQ_2 / lambda_reg, axis=1)

        policy_1 = damping * npolicy_1 + (1 - damping) * policy_1
        policy_2 = damping * npolicy_2 + (1 - damping) * policy_2

        V_1 = np.sum(policy_1[:, :, np.newaxis] * policy_2[:, np.newaxis] * Q_1, axis=(1,2))
        V_2 = np.sum(policy_1[:, :, np.newaxis] * policy_2[:, np.newaxis] * Q_2, axis=(1,2))

    check_qre(Q_1, Q_2, policy_1, policy_2, lambda_reg, tol=tol*5)

    return policy_1, policy_2, V_1, V_2


def compute_best_response(
        mdp: TwoPlayerMDP,
        gamma,
        lambda_reg,
        f,
        policy_2,
        tol=2e-5,
        damping=1.,
        max_iters=200):
    num_states = mdp.num_states
    num_actions = mdp.num_actions

    R_1 = mdp.construct_fair_rewards(f)

    V_br = np.zeros(num_states)

    policy_1 = np.ones((num_states, num_actions)) / num_actions

    vs = []
    for _ in range(max_iters):

        Q_1 = R_1 + gamma * np.sum(mdp.transition_matrix * V_br[np.newaxis, np.newaxis, np.newaxis, :], axis=-1)

        EQ_1 = np.sum(Q_1 * policy_2[:, None], axis=2)

        npolicy_1 = softmax(EQ_1 / lambda_reg, axis=1)

        policy_1 = damping * npolicy_1 + (1 - damping) * policy_1

        V_br = np.sum(policy_1[:, :, np.newaxis] * policy_2[:, np.newaxis] * Q_1, axis=(1,2))

        vs.append(np.sum(V_br * mdp.initial_state_dist))

    return np.sum(V_br * mdp.initial_state_dist)


def softmax(x, axis=None):
    xm = x - np.max(x)
    ex = np.exp(xm)
    return ex / ex.sum(axis=axis, keepdims=True)


def check_qre(Q1, Q2, pi1, pi2, reg, tol=5e-3):
    """Verify that (pi1, pi2) is a QRE"""

    # check there is no deviation incentivisation + equality
    pi1_v = softmax(np.sum(pi2[:, None, :] * Q1, axis=2) / reg, axis=1)
    pi2_v = softmax(np.sum(pi1[:, :, None] * Q2, axis=1) / reg, axis=1)

    stability = np.maximum(np.max(np.abs(pi1 - pi1_v)), np.max(np.abs(pi2 - pi2_v)))

    assert np.all(np.abs(pi1 - pi1_v) < tol), (pi1, pi1_v, np.abs(pi1 - pi1_v))
    assert np.all(np.abs(pi2 - pi2_v) < tol), (pi2, pi2_v,  np.abs(pi2 - pi2_v))

    return stability


def stable_kl(pi, logits):
    # logits: pre-softmax values
    log_pi = np.log(pi + 1e-8)
    dlogits = logits - np.max(logits)
    log_softmax = dlogits - np.log(np.sum(np.exp(dlogits), axis=1, keepdims=True))

    return np.mean(np.sum(pi * (log_pi - log_softmax), axis=1))

def qre_gap(
        Q1, Q2, pi1, pi2,
        reg
):

    logits1 = np.sum(pi2[:, None, :] * Q1, axis=2) / reg
    logits2 = np.sum(pi1[:, :, None] * Q2, axis=1) / reg

    kl1 = stable_kl(pi1, logits1)
    kl2 = stable_kl(pi2, logits2)

    return np.maximum(kl1, kl2)


def compute_value(policy_1, policy_2, mdp, gamma, f1, f2):
    num_states = mdp.num_states

    R_1 = mdp.construct_fair_rewards(f1)
    R_2 = mdp.construct_fair_rewards(f2).swapaxes(1, 2)

    V_1 = np.zeros(num_states)
    V_2 = np.zeros(num_states)

    for _ in range(100):
        V_1_old, V_2_old = V_1.copy(), V_2.copy()

        Q_1 = R_1 + gamma * np.sum(mdp.transition_matrix * V_1[np.newaxis, np.newaxis, np.newaxis, :], axis=-1)
        Q_2 = R_2 + gamma * np.sum(mdp.transition_matrix * V_2[np.newaxis, np.newaxis, np.newaxis, :], axis=-1)

        V_1 = np.sum(policy_1[:, :, np.newaxis] * policy_2[:, np.newaxis] * Q_1, axis=(1, 2))
        V_2 = np.sum(policy_1[:, :, np.newaxis] * policy_2[:, np.newaxis] * Q_2, axis=(1, 2))

        if np.max(np.abs(V_1 - V_1_old)) < 1e-5 and np.max(np.abs(V_2 - V_2_old)) < 1e-5:
            break

    return V_1, V_2

def compute_visitation(pi_1, pi_2, transition_matrix, initial_state_dist, gamma, iters=100):

    visitation = initial_state_dist

    pi_1 = pi_1[:, :, np.newaxis, np.newaxis]
    pi_2 = pi_2[:, np.newaxis, :, np.newaxis]

    pi_transition = transition_matrix * pi_1 * pi_2

    visitations = [visitation]

    for i in range(iters):
        next_vis = visitation[:, np.newaxis, np.newaxis, np.newaxis] * pi_transition
        next_vis = np.sum(next_vis, axis=(0,1,2))
        visitations.append(next_vis * gamma ** (i + 1.))
        visitation = next_vis

    visitation = (1 - gamma) * np.sum(visitations, axis=0)

    return visitation
