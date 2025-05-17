import numpy as np
import itertools

ACTION_PAIRS = ["CC", "CD", "DC", "DD"]
ACTION_TO_IDX = {pair: i for i, pair in enumerate(ACTION_PAIRS)}
IDX_TO_ACTION = [(0, 0), (0, 1), (1, 0), (1, 1)]

class IPD(TwoPlayerMDP):
    def __init__(self,
                 episode_length: int,
                 reward_func: np.ndarray = None,
                 window_length: int = 3,
                 **kwargs):

        self.window_length = window_length
        self.episode_length = episode_length

        # All possible histories of w action pairs
        self.all_histories = list(itertools.product(range(4), repeat=window_length))
        self.state_to_idx = {h: i for i, h in enumerate(self.all_histories)}
        self.idx_to_state = {i: h for h, i in self.state_to_idx.items()}
        num_states = len(self.all_histories)
        self.num_states = num_states
        self.num_actions = 2

        # Initial state: all 'CC'
        initial_state = tuple([0] * window_length)  # 'CC' encoded as 0
        initial_state_idx = self.state_to_idx[initial_state]
        initial_state_dist = np.zeros(num_states)
        initial_state_dist[initial_state_idx] = 1.

        super().__init__(num_states, 2, episode_length, initial_state_dist)
        self.transition_matrix = self._generate_transition_matrix()

        if isinstance(reward_func, np.ndarray):
            self.reward_matrix = reward_func.copy()
        elif reward_func is None:
            self.reward_matrix = self.get_reward_matrix_for()
        else:
            self.reward_matrix = self._generate_reward_matrix(reward_func)

    def _generate_transition_matrix(self) -> np.ndarray:
        num_states = self.num_states
        matrix = np.zeros((num_states, 2, 2, num_states), dtype=np.float32)

        for s_idx, history in self.idx_to_state.items():
            for a1 in [0, 1]:
                for a2 in [0, 1]:
                    new_pair = self._action_pair_to_idx((a1, a2))
                    new_history = history[1:] + (new_pair,)
                    next_state_idx = self.state_to_idx[new_history]
                    matrix[s_idx, a1, a2, next_state_idx] = 1.0

        return matrix

    def _action_pair_to_idx(self, actions):
        return ACTION_TO_IDX[self._actions_to_str(actions)]

    def _actions_to_str(self, actions):
        return ''.join(['C' if a == 0 else 'D' for a in actions])

    def _generate_reward_matrix(self, reward_func):
        return np.array([
            [
                reward_func(self.random_state, s, (a1, a2))
                for a1 in range(2)
                for a2 in range(2)
            ]
            for s in range(self.num_states)
        ]).reshape((self.num_states, 2, 2, 2))

    def get_reward_matrix_for(self, cc=(4, 4), cd=(0, 5), dc=(5, 0), dd=(1, 1)):
        reward_matrix = np.zeros((self.num_states, 2, 2, 2), dtype=np.float32)
        payoff_dict = {
            (0, 0): cc,
            (0, 1): cd,
            (1, 0): dc,
            (1, 1): dd
        }
        for s in range(self.num_states):
            for a1 in range(2):
                for a2 in range(2):
                    r1, r2 = payoff_dict[(a1, a2)]
                    reward_matrix[s, a1, a2] = [r1, r2]
        return reward_matrix