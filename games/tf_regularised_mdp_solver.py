import tensorflow as tf

def compute_soft_nash_equilibrium_tf(
        transition_matrix,
        R_1,
        R_2,
        gamma=0.99,
        lambda_reg=0.1,
        damping=0.05,
        iters=500,
):
    shape = tf.shape(transition_matrix)
    num_states = shape[0]
    num_actions = shape[1]

    V_1 = tf.zeros([num_states], dtype=tf.float32)
    V_2 = tf.zeros([num_states], dtype=tf.float32)

    policy_1 = tf.fill([num_states, num_actions], 1.0 / tf.cast(num_actions, dtype=tf.float32))
    policy_2 = tf.fill([num_states, num_actions], 1.0 / tf.cast(num_actions, dtype=tf.float32))

    def soft_value_iteration(V_1, V_2, policy_1, policy_2):
        exp_policy_1 = tf.expand_dims(policy_1, axis=2)
        exp_policy_2 = tf.expand_dims(policy_2, axis=1)

        for _ in tf.range(iters):

            Q_1 = R_1 + gamma * tf.linalg.matvec(transition_matrix, V_1)
            Q_2 = R_2 + gamma * tf.linalg.matvec(transition_matrix, V_2)

            EQ_1 = tf.reduce_sum(Q_1 * exp_policy_2, axis=2)
            EQ_2 = tf.reduce_sum(Q_2 * exp_policy_1, axis=1)

            policy_1 = tf.nn.softmax(EQ_1 / lambda_reg, axis=1) * damping + policy_1 * (1. - damping)
            policy_2 = tf.nn.softmax(EQ_2 / lambda_reg, axis=1) * damping + policy_2 * (1. - damping)

            exp_policy_1 = tf.expand_dims(policy_1, axis=2)
            exp_policy_2 = tf.expand_dims(policy_2, axis=1)

            # Compute value functions
            pprod = exp_policy_1 * exp_policy_2
            V_1 = tf.reduce_sum(pprod * Q_1, axis=[1, 2])
            V_2 = tf.reduce_sum(pprod * Q_2, axis=[1, 2])

        return policy_1, policy_2, Q_1, Q_2

    return soft_value_iteration(V_1, V_2, policy_1, policy_2)


def compute_soft_nash_equilibrium_sweep_tf(
        transition_matrix,
        R_1,
        R_2,
        gamma=0.99,
        damping=0.05,
        iters=500,
        num_regs=20,
):
    regs = tf.linspace(1., 0.05, num_regs)

    exp_regs = tf.expand_dims(tf.expand_dims(regs, axis=-1), axis=-1)

    shape = tf.shape(transition_matrix)
    num_states = shape[0]
    num_actions = shape[1]

    V_1 = tf.zeros([num_regs, num_states], dtype=tf.float32)
    V_2 = tf.zeros([num_regs, num_states], dtype=tf.float32)


    policy_1 = tf.fill([num_regs, num_states, num_actions], 1.0 / tf.cast(num_actions, dtype=tf.float32))
    policy_2 = tf.fill([num_regs, num_states, num_actions], 1.0 / tf.cast(num_actions, dtype=tf.float32))

    def soft_value_iteration(V_1, V_2, policy_1, policy_2):
        exp_policy_1 = tf.expand_dims(policy_1, axis=3)
        exp_policy_2 = tf.expand_dims(policy_2, axis=2)

        for _ in tf.range(iters):

            Q_1 = R_1 + gamma *  tf.einsum("xabs,ns->nxab", transition_matrix, V_1)
            Q_2 = R_2 + gamma *  tf.einsum("xabs,ns->nxab", transition_matrix, V_2)

            EQ_1 = tf.reduce_sum(Q_1 * exp_policy_2, axis=3)
            EQ_2 = tf.reduce_sum(Q_2 * exp_policy_1, axis=2)

            policy_1 = tf.nn.softmax(EQ_1 / exp_regs, axis=2) * damping + policy_1 * (1. - damping)
            policy_2 = tf.nn.softmax(EQ_2 / exp_regs, axis=2) * damping + policy_2 * (1. - damping)

            exp_policy_1 = tf.expand_dims(policy_1, axis=3)
            exp_policy_2 = tf.expand_dims(policy_2, axis=2)

            # Compute value functions
            pprod = exp_policy_1 * exp_policy_2
            V_1 = tf.reduce_sum(pprod * Q_1, axis=[2, 3])
            V_2 = tf.reduce_sum(pprod * Q_2, axis=[2, 3])

        return policy_1, policy_2


    c = 6.
    optimality_exps = tf.math.exp(-c * tf.concat([[100.], regs], axis=0))
    optimality_probs = optimality_exps[1:] - optimality_exps[:-1]

    pi_1, pi_2 = soft_value_iteration(V_1, V_2, policy_1, policy_2)
    # shape of policies (num_reg, s, a)
    return pi_1, pi_2, optimality_probs


def compute_batch_best_response_tf(
        transition_matrix,
        R_1,
        policy_2,
        gamma=0.99,
        iters=128,
        lambda_reg=0.04,
        damping=1.,
):
    shape = tf.shape(transition_matrix)
    num_states = shape[0]
    num_actions = shape[1]
    n = policy_2.shape[0]
    V_1 = tf.zeros([n, num_states], dtype=tf.float32)
    R_1 = tf.expand_dims(R_1, axis=0)


    def value_iteration(V_1, policy_2):
        exp_policy_2 = tf.expand_dims(policy_2, axis=2)
        policy_1 = tf.fill([n, num_states, num_actions], 1.0 / tf.cast(num_actions, dtype=tf.float32))

        for _ in tf.range(iters):

            Q_1 = R_1 + gamma * tf.einsum("xabs,ns->nxab", transition_matrix, V_1)


            EQ_1 = tf.reduce_sum(Q_1 * exp_policy_2, axis=3)

            policy_1 = tf.nn.softmax(EQ_1 / lambda_reg, axis=2) * damping + policy_1 * (1. - damping)
            exp_policy_1 = tf.expand_dims(policy_1, axis=3)

            # Compute value functions
            pprod = exp_policy_1 * exp_policy_2
            V_1 = tf.reduce_sum(pprod * Q_1, axis=[2, 3])

        return policy_1, Q_1, V_1

    return value_iteration(V_1, policy_2)


def compute_visitation(pi_1, pi_2, transition_matrix, initial_state_dist, gamma, iters=60):

    visitation = tf.tile(tf.expand_dims(initial_state_dist, axis=0), [tf.shape(pi_1)[0] , 1])

    pi_1 = tf.expand_dims(tf.expand_dims(pi_1, axis=3), axis=-1)
    pi_2 = tf.expand_dims(tf.expand_dims(pi_2, axis=2), axis=-1)

    pi_transition = transition_matrix * pi_1 * pi_2

    def loop(acc, idx):
        cumsum, vis = acc
        #next_vis = tf.einsum("nsabp,ns->nsp", pi_transition , vis)

        next_vis = tf.expand_dims(tf.expand_dims(tf.expand_dims(vis, axis=-1), axis=-1), axis=-1) * pi_transition
        next_vis = tf.reduce_sum(next_vis, axis=[1,2,3])
        return [vis + next_vis * (gamma ** (idx + 1.)), next_vis]

    x = tf.scan(
        loop,
        tf.range(iters, dtype=tf.float32),
        initializer=[visitation, visitation]
    )[0]

    visitation = (1 - gamma) * x[-1]
    return visitation


def qre_batch_value_gap_tf(pi_1, pi_2, pi_br1, pi_br2, V_1, V_2, V_br1, V_br2,
                           transition_matrix, initial_state_dist, gamma
                           ):
    """
    Computes in a batch the value gap

    inputs are of shape (n, s)
    """

    visitation_12 = compute_visitation(pi_1, pi_2, transition_matrix, initial_state_dist, gamma)
    visitation_br1 = compute_visitation(pi_br1, pi_2, transition_matrix, initial_state_dist, gamma)
    visitation_br2 = compute_visitation(pi_1, pi_br2, transition_matrix, initial_state_dist, gamma)

    V_1 = tf.reduce_sum(visitation_12 * V_1, axis=1)
    V_2 = tf.reduce_sum(visitation_12 * V_2, axis=1)
    V_br1 = tf.reduce_sum(visitation_br1 * V_br1, axis=1)
    V_br2 = tf.reduce_sum(visitation_br2 * V_br2, axis=1)

    # max over policy index
    return tf.maximum(tf.abs(V_br1 - V_1), tf.abs(V_br2 - V_2))

def qre_batch_max_value_gap_tf(V_1, V_2, V_br1, V_br2):
    """
    Computes in a batch the value gap

    inputs are of shape (n, s)
    """

    return tf.maximum(tf.reduce_mean(tf.abs(V_br1 - V_1), axis=1), tf.reduce_mean(tf.abs(V_br2 - V_2), axis=1))


def stable_kl(pi, logits):
    # logits: pre-softmax values
    log_pi = tf.math.log(pi + 1e-8)
    dlogits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    log_softmax = tf.math.log_softmax(dlogits, axis=-1)

    # average over states
    return tf.reduce_sum(pi * (log_pi - log_softmax), axis=-1)
    #return tf.reduce_mean(tf.reduce_sum(pi * (log_pi - log_softmax), axis=-1), axis=-1)


def qre_batch_kl_gap_tf_for_reg(
        Q1, Q2,
        pi1, pi2,
        reg,
        #transition_matrix, initial_state_dist, gamma,
):
    logits1 = tf.reduce_sum(tf.expand_dims(pi2, axis=2) * Q1, axis=3) / reg
    logits2 = tf.reduce_sum(tf.expand_dims(pi1, axis=3) * Q2, axis=2) / reg


    kl1 = tf.reduce_mean(stable_kl(pi1, logits1), axis=-1)
    kl2 = tf.reduce_mean(stable_kl(pi2, logits2), axis=-1)

    return tf.maximum(kl1, kl2)


def qre_batch_kl_gap_sweep_tf(Q1, Q2, pi1, pi2, num_regs=10):
    """
    Evaluate QRE batch gap over a range of regularization coefficients.

    Args:
        Q1, Q2: Q-values of shape (n, s, a, a)
        pi1, pi2: Policies of shape (n, s, a)
        num_regs: Number of regularization coefficients to sample between 0 and 1

    Returns:
        gaps: Tensor of shape (num_regs, n) representing the QRE gap for each reg value
    """
    
    regs = tf.linspace(1., 0.04, num_regs)

    # Expand dimensions to broadcast over reg values
    Q1_exp = tf.expand_dims(Q1, axis=0)         # (1, n, s, a, a)
    Q2_exp = tf.expand_dims(Q2, axis=0)
    pi1_exp = tf.expand_dims(pi1, axis=0)       # (1, n, s, a)
    pi2_exp = tf.expand_dims(pi2, axis=0)
    regs_exp = tf.reshape(regs, (-1, 1, 1, 1))   # (num_regs, 1, 1, 1)

    logits1 = tf.reduce_sum(tf.expand_dims(pi2_exp, axis=3) * Q1_exp, axis=4) / regs_exp  # (num_regs, n, s, a)
    logits2 = tf.reduce_sum(tf.expand_dims(pi1_exp, axis=4) * Q2_exp, axis=3) / regs_exp  # (num_regs, n, s, a)

    kl1 = tf.reduce_mean(stable_kl(pi1_exp, logits1), axis=-1)  # should return shape (num_regs, n)
    kl2 = tf.reduce_mean(stable_kl(pi2_exp, logits2), axis=-1)

    gap = tf.maximum(kl1, kl2)  # (num_regs, n)

    c = 6.
    optimality_exps = tf.math.exp(-c * tf.concat([[100.], regs], axis=0))
    optimality_probs = optimality_exps[1:] - optimality_exps[:-1]
    return gap, optimality_probs


def compute_batch_q_tf(
        transition_matrix,
        R_1,
        R_2,
        policy_1, # [n, s, a]
        policy_2, # [n, s, a]
        gamma=0.99,
        iters=60
):
    num_states = policy_1.shape[1]
    n = policy_1.shape[0]

    V_1 = tf.zeros([n, num_states], dtype=tf.float32)
    V_2 = tf.zeros([n, num_states], dtype=tf.float32)
    R_1 = tf.expand_dims(R_1, axis=0)
    R_2 = tf.expand_dims(R_2, axis=0)


    def policy_eval(V_1, V_2, policy_1, policy_2):

        exp_policy_1 = tf.expand_dims(policy_1, axis=3)
        exp_policy_2 = tf.expand_dims(policy_2, axis=2)
        pprod = exp_policy_1 * exp_policy_2
        for _ in tf.range(iters):

            Q_1 = R_1 + gamma * tf.einsum("xabs,ns->nxab", transition_matrix, V_1)
            Q_2 = R_2 + gamma * tf.einsum("xabs,ns->nxab", transition_matrix, V_2)

            V_1 = tf.reduce_sum(pprod * Q_1, axis=[2, 3])
            V_2 = tf.reduce_sum(pprod * Q_2, axis=[2, 3])


        return Q_1, Q_2, V_1, V_2

    return policy_eval(V_1, V_2, policy_1, policy_2)
