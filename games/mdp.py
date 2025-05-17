from copy import deepcopy

import gymnasium.spaces
import numpy as np
from typing import Tuple, Callable, NamedTuple, List

import tree


class TimeStep(NamedTuple):
    state: int
    next_state: int
    action_1: int
    action_2: int
    reward_1: float
    reward_2: float
    done: bool = False

    @staticmethod
    def stack(timesteps):
        return tree.map_structure(
            lambda *v: np.stack(v),
            *timesteps
        )


def compute_returns(rollout: List[TimeStep], gamma: float) -> List[float]:
    """Computes the empirical return G_t for each time step in a rollout.

    Args:
        rollout: List of TimeStep objects forming an episode.
        gamma: Discount factor.

    Returns:
        A list of returns G_t, where G_t is the discounted sum of future rewards from t onward.
    """
    T = len(rollout)
    returns = [0.0] * T
    G = 0.0

    for t in reversed(range(T)):
        G = rollout[t].reward + (gamma * G if not rollout[t].done else 0)
        returns[t] = G

    return returns

class TwoPlayerMDP:
    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 episode_length: int,
                 initial_state_dist: np.ndarray,
                 **kwargs
                 ):
        """
        Args:
            num_states: Number of states in the MDP.
            num_actions: Number of actions in the MDP.
            episode_length: Maximum episode length.
            initial_state_dist: probability distribution over initial states.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.episode_length = episode_length
        self.initial_state_dist = np.float32(initial_state_dist)

        self.transition_matrix = np.empty((num_states, num_actions, num_actions, num_states), dtype=np.float32)

        # assumed structure for the reward function
        # r(s,a,b) = r(s,a)
        self.reward_matrix = np.empty((num_states, num_actions), dtype=np.float32)

        self.action_space = gymnasium.spaces.Discrete(self.num_actions)
        self.observation_space = gymnasium.spaces.Discrete(self.num_states)

    def construct_fair_rewards(self, fairness_level: float, p_idx: int):
        matrix = np.empty((self.num_states, self.num_actions, self.num_actions), dtype=np.float32)
        matrix[:] = self.reward_matrix[:, :, np.newaxis] + fairness_level * self.reward_matrix[:, np.newaxis]
        return matrix

    def step(self,
             state: int,
             action_1: int,
             action_2: int
             ) -> Tuple[int, float, float]:
        """
        Executes a step in the MDP.

        Args:
            state: Current state.
            action_1: Action taken by player 1.
            action_2: Action taken by player 2.

        Returns:
            next_state: The next state reached after taking the action.
            reward: The reward received for the transition.
        """
        next_state = np.random.choice(
            self.num_states,
            p=self.transition_matrix[state, action_1, action_2]
        )
        reward_1 = self.reward_matrix[state, action_1]
        reward_2 = self.reward_matrix[state, action_2]

        return next_state, reward_1, reward_2

    def reset(self) -> int:
        """
        Resets the MDP to a random initial state.

        Returns:
            An initial state sampled according to the initial state distribution.
        """
        return np.random.choice(self.num_states, p=self.initial_state_dist)

    def rollout(
            self,
            policy_1,
            policy_2,

    ) -> List[TimeStep]:
        """
        Deploys the policies on the mdp and performs a rollout.
        Returns: collected timesteps
        """

        timesteps = []

        s = self.reset()
        for t in range(self.episode_length):
            a_1 = np.random.choice(self.num_actions, p=policy_1[s])
            a_2 = np.random.choice(self.num_actions, p=policy_2[s])

            next_s, r_1, r_2 = self.step(s, a_1, a_2)

            done = t == self.episode_length - 1

            timesteps.append(
                TimeStep(
                    state=s,
                    next_state=next_s,
                    action_1=a_1,
                    action_2=a_2,
                    reward_1=r_1,
                    reward_2=r_2,
                    done=done
                )
            )

            s = next_s

        return timesteps


class RandomTwoPlayerMDP(TwoPlayerMDP):
    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 episode_length: int,
                 initial_state_dist: np.ndarray,
                 reward_func: Callable[[np.random.Generator, int, int], float] | np.ndarray = None,
                 alpha: float = 0.5,
                 seed: int = None,
                 **kwargs
                 ):
        """
        Constructs a random MDP.

        Args:
            num_states: Number of states in the MDP.
            num_actions: Number of actions in the MDP.
            episode_length: Maximum episode length.
            reward_func: Function to generate rewards for transitions, given
                         (state, action).
            initial_state_dist: Function returning a probability distribution
                                over initial states.
            alpha: Dirichlet concentration parameter for transitions.
            seed: Seed used to generate the random MDP.
        """
        super().__init__(num_states, num_actions, episode_length, initial_state_dist)
        self.random_state = np.random.default_rng(seed)
        self.alpha = alpha
        self.transition_matrix = self._generate_transition_matrix()
        if isinstance(reward_func, np.ndarray):
            self.reward_matrix = reward_func.copy()
        elif reward_func is None:
            self.reward_matrix = np.zeros((num_states, num_actions), dtype=np.float32)
        else:
            self.reward_matrix = self._generate_reward_matrix(reward_func)

    def _generate_transition_matrix(self) -> np.ndarray:
        """
        Generates a transition matrix using the Dirichlet distribution.

        Returns:
            A transition probability matrix of shape
            (num_states, num_actions, num_states).
        """
        matrix = np.zeros((self.num_states, self.num_actions, self.num_actions, self.num_states), dtype=np.float32)
        for s in range(self.num_states):
            for a_1 in range(self.num_actions):
                for a_2 in range(self.num_actions):
                    # Sample a probability vector from the Dirichlet distribution
                    matrix[s, a_2, a_1] = matrix[s, a_1, a_2] = self.random_state.dirichlet(
                        np.ones(self.num_states) * self.alpha
                    )
        return matrix

    def _generate_reward_matrix(self, reward_func) -> np.ndarray:
        """
        Generates a reward matrix using the reward function.

        Returns:
            A reward matrix of shape (num_states, num_actions, num_states).
        """
        return np.array([
            [
                reward_func(self.random_state, s, a)
                for a in range(self.num_actions)
            ]
            for s in range(self.num_states)
        ])