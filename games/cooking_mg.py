import itertools
import os
import pickle
from enum import Enum
from typing import NamedTuple
import imageio

from games.mdp import TwoPlayerMDP
from games.regularised_mdp_solver import compute_soft_nash_equilibrium

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

donut="""##P##
#   #
DSM O
#   #
##T##"""

cramped="""#MP#
#S #
D  O
##T#"""

small="""##P#
#S #
D  O
##T#"""

layouts = {
    "donut": donut,
    "cramped": cramped,
    "small": small,
}


class hashabledict(dict):
    def __key(self):
        return tuple((k,self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


class PlayerState(Enum):
    NORMAL = 0
    HAS_TOMATO = 1
    HAS_PLATE = 2
    HAS_DISH = 3


class Action(Enum):
    INTERACT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

# Icons or Labels for player states
PLAYER_COLORS = {
    PlayerState.NORMAL: "#FFFFFF00",
    PlayerState.HAS_TOMATO: "red",
    PlayerState.HAS_PLATE: "white",
    PlayerState.HAS_DISH: "green"
}

TILE_COLORS = {
    "O": None,
    "T": "red",
    "D": "white",
    "M": ("#FFFFFF00", "red"),
    "P": ("black", "orange")
}


class State(NamedTuple):
    p1: tuple
    p2: tuple
    player_states: tuple[PlayerState]

    pot_states: hashabledict[tuple, bool]
    desk_states: hashabledict[tuple, bool]


def replace(t: NamedTuple, **fields):

    d = t._asdict()
    d.update(fields)

    return t.__class__(**d)


def action_to_pos(pos, action):
    x, y = pos
    if action == action.INTERACT:
        return pos
    elif action == action.UP:
        return x, y + 1
    elif action == action.RIGHT:
        return x + 1, y
    elif action == action.DOWN:
        return x, y - 1
    elif action == action.LEFT:
        return x - 1, y


def replace_player_state(state, idx, s):
    if idx == 0:
        player_states = (s, state.player_states[1])
    else:
        player_states = (state.player_states[0], s)
    return player_states


def interact(state: State, pos: tuple, p_idx:int,  layout):
    x, y = pos
    next_positions = [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
    ]
    neighbouring = {
        layout[p]: p for p in next_positions
    }

    s = state.player_states[p_idx]

    if s == PlayerState.NORMAL:

        # We can either pick up a plate, pick up a tomato from the dispenser, of from the desk
        # we can have the case where we both have dispenser, desk or plate as neighbours
        if "D" in neighbouring:
            other_p_idx = (p_idx + 1) % 2
            # forbid two players with plate
            if state.player_states[other_p_idx] != PlayerState.HAS_PLATE:
                player_states = replace_player_state(state, p_idx, PlayerState.HAS_PLATE)
                return replace(state, player_states=player_states), 0
        if "M" in neighbouring:
            desk_state = state.desk_states[neighbouring["M"]]
            if desk_state:
                # has a tomato here
                desk_states = hashabledict({
                    **state.desk_states,
                    neighbouring["M"]: False
                })
                player_states = replace_player_state(state, p_idx, PlayerState.HAS_TOMATO)
                return replace(state, player_states=player_states, desk_states=desk_states), 0
        if "T" in neighbouring:
            player_states = replace_player_state(state, p_idx, PlayerState.HAS_TOMATO)
            return replace(state, player_states=player_states), 0

    elif s == PlayerState.HAS_TOMATO:
        # if we have a tomato, we can either put it into the pot, or on a desk
        if "P" in neighbouring:
            pot_state = state.pot_states[neighbouring["P"]]
            if not pot_state:
                # the pot is empty, we can put a tomato there.
                pot_states = hashabledict({
                    ** state.pot_states,
                    neighbouring["P"]: True
                })
                player_states = replace_player_state(state, p_idx, PlayerState.NORMAL)
                return replace(state, player_states=player_states, pot_states=pot_states), 0
        if "M" in neighbouring:
            desk_state = state.desk_states[neighbouring["M"]]
            if not desk_state:
                # has no tomato here
                desk_states = hashabledict({
                    ** state.desk_states,
                    neighbouring["M"]: True
                })

                player_states = replace_player_state(state, p_idx, PlayerState.NORMAL)
                return replace(state, player_states=player_states, desk_states=desk_states), 0

    elif s == PlayerState.HAS_PLATE:
        if "P" in neighbouring:
            pot_state = state.pot_states[neighbouring["P"]]
            if pot_state:
                # the pot is ready, we can pick up the soup
                pot_states = hashabledict({
                    **state.pot_states,
                    neighbouring["P"]: False
                })
                player_states = replace_player_state(state, p_idx, PlayerState.HAS_DISH)
                return replace(state, player_states=player_states, pot_states=pot_states), 0

    elif s == PlayerState.HAS_DISH:
        if "O" in neighbouring:
            player_states = replace_player_state(state, p_idx, PlayerState.NORMAL)
            return replace(state, player_states=player_states), 1.

    return state, 0


def get_state_neighbours(state: State, layout):

    neighbours = {}
    rewards = {}

    p1 = state.p1
    p2 = state.p2

    for a1 in Action:
        for a2 in Action:
            ns = state
            np1 = action_to_pos(p1, a1)
            np2 = action_to_pos(p2, a2)

            if layout[np1] not in "S ":
                np1 = p1
            if layout[np2] not in "S ":
                np2 = p2

            if np1 == np2 or (np1 == p2 and np2 == p1):
                np1 = p1
                np2 = p2


            ns = replace(ns, p1=np1, p2=np2)

            r = (0., 0.)
            if a1 == a2 == Action.INTERACT:
                ns1, r1 = interact(ns, p1, 0, layout)
                ns1, r2 = interact(ns1, p2, 1, layout)


                ns2, r1 = interact(ns, p2, 1, layout)
                ns2, r2 = interact(ns2, p1, 0, layout)

                if ns1 != ns2:
                    # break randomly ties instead
                    neighbours[(a1, a2)] = {
                        ns1: 0.5,
                        ns2: 0.5
                    }
                    r = 0., 0.
                else:
                    neighbours[(a1, a2)] = {
                        ns1: 1.
                    }

                    r = r1, r2

            elif a1 == Action.INTERACT:
                ns, r1 = interact(ns, p1, 0, layout)
                r = r1, 0.
                neighbours[(a1, a2)] = {
                    ns: 1.
                }

            elif a2 == Action.INTERACT:
                ns, r2 = interact(ns, p2, 1, layout)
                r = 0., r2
                neighbours[(a1, a2)] = {
                    ns: 1.
                }

            else:
                neighbours[(a1, a2)] = {
                    ns: 1.
                }

            rewards[(a1, a2)] = r


    return neighbours, rewards


def build_from_layout(layout_name: str):
    layout_str = layouts[layout_name]
    layout = np.array([list(l) for l in layout_str.split("\n")])

    num_free_tiles = layout_str.count(" ") + 1
    walkables = [tuple(w.tolist()) for w in list(np.argwhere(layout==" ")) + list(np.argwhere(layout=="S"))]
    num_pots = layout_str.count("P")
    num_desks = layout_str.count("M")

    num_combined_positions = num_free_tiles * (num_free_tiles - 1)

    num_desk_states = 2 ** num_desks

    player_state = 2 ** len(PlayerState)

    pot_state = 2 ** num_pots

    num_states = num_combined_positions * num_desk_states * player_state * pot_state + 1

    init_pos = tuple(np.argwhere(layout=="S")[0].tolist())
    pot_pos = [tuple(p.tolist()) for p in np.argwhere(layout=="P")]
    desk_pos = [tuple(p.tolist()) for p in np.argwhere(layout=="M")]

    initial_state = State(
            p1=init_pos,
            p2=init_pos,
            player_states=(PlayerState.NORMAL, PlayerState.NORMAL),
            pot_states=hashabledict({
                p: False
                for p in pot_pos
            }),
            desk_states=hashabledict({
                p: False
                for p in desk_pos
            }),
    )

    transitions, rewards = get_state_neighbours(initial_state, layout)

    t = {
        initial_state: transitions
    }

    r = {initial_state: rewards}

    for p1 in walkables:
        for p2 in walkables:
            for desk_states in itertools.product(*[[True, False] for _ in range(num_desks)]):
                for pot_states in itertools.product(*[[True, False] for _ in range(num_pots)]):
                    for s1 in PlayerState:
                        for s2 in PlayerState:
                            if p1 == p2 != init_pos:
                                continue
                            if s1 == s2 == PlayerState.HAS_PLATE:
                                continue

                            state = State(
                                p1=p1,
                                p2=p2,
                                player_states=(s1, s2),
                                pot_states=hashabledict({
                                    p: s
                                    for p, s in zip(pot_pos, pot_states)
                                }),
                                desk_states=hashabledict({
                                    p: s
                                    for p, s in zip(desk_pos, desk_states)
                                }),
                            )

                            transitions, rewards = get_state_neighbours(state, layout)

                            t[state] = transitions
                            r[state] = rewards

    s_i = {
        s: i
        for i, s in enumerate(t)
    }
    state_meanings = {
        i: s
        for i, s in enumerate(t)
    }

    state_swaps = []
    for state, idx in s_i.items():
        s = state.player_states[1], state.player_states[0]
        state_sym = replace(state, p1=state.p2, p2=state.p1, player_states=s)

        idx_sym = s_i[state_sym]

        state_swaps.append(idx_sym)

    num_states = len(s_i)
    num_actions = len(Action)
    # build matrices
    transition_matrix = np.zeros((num_states, num_actions, num_actions, num_states))
    base_reward_func1 = np.zeros((num_states, num_actions))
    base_reward_func2 = np.zeros((num_states, num_actions))

    for state, neighbours in t.items():
        for (a1, a2), next_states in neighbours.items():
            a1_idx = a1.value
            a2_idx = a2.value
            for next_state, prob in next_states.items():
                transition_matrix[s_i[state], a1_idx, a2_idx, s_i[next_state]] = prob
            r1, r2 = r[state][a1, a2]
            base_reward_func1[s_i[state], a1_idx] = r1
            base_reward_func2[s_i[state], a2_idx] = r2

    s_0 = s_i[initial_state]

    print(f"Generated the Markov game for layout {layout_name}: {num_states} states and {num_actions} actions.")
    return s_0, s_i, transition_matrix, base_reward_func1, base_reward_func2, state_meanings, state_swaps

def get_mdp_for_layout(layout_name):
    path = f"mdp/compiled_{layout_name}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            ret = pickle.load(f)
        return ret

    ret = build_from_layout(layout_name)
    with open(path, "wb") as f:
        pickle.dump(ret, f)

    return ret


class LightCollaborativeCooking(TwoPlayerMDP):
    def __init__(self,
                 episode_length=100,
                 layout="donut",
                 **kwargs
                 ):

        s_0, s_i, transition_matrix, base_reward_func1, base_reward_func2, state_meanings, state_swaps = get_mdp_for_layout(layout)

        num_actions = len(Action)
        num_states = len(s_i)

        initial_state_dist = np.zeros(num_states)
        initial_state_dist[s_0] = 1.

        super().__init__(num_states, num_actions, episode_length, initial_state_dist)
        self.layout = layout
        self.transition_matrix = transition_matrix
        self.state_swaps = state_swaps
        self.reward_func = base_reward_func1

        # R_1 is a permutation of R_2
        self.R = {1: base_reward_func1, 2: base_reward_func2}

        self.state_meanings = state_meanings

    def construct_fair_rewards(self, fairness_level: float, p_idx: int):
        matrix = np.empty((self.num_states, self.num_actions, self.num_actions), dtype=np.float32)
        matrix[:] = self.R[p_idx][:, :, np.newaxis] + fairness_level * self.R[(p_idx % 2) + 1][:, np.newaxis]
        return matrix

    def step(self,
             state: int,
             action_1: int,
             action_2: int
             ) -> tuple[int, float, float]:
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
        reward_1 = self.R[1][state, action_1]
        reward_2 = self.R[2][state, action_2]

        return next_state, reward_1, reward_2


class RolloutVisualiser:
    def __init__(self, layout, rollout, delay=0.5):
        self.layout = np.array([list(row) for row in layout.split("\n")])
        self.rollout = rollout
        self.delay = delay
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

    def render_state(self, state, cumulated=0, step=0):
        self.ax.clear()

        # Draw the grid layout
        for (y, x), value in np.ndenumerate(self.layout):
            color = "white"
            if value == "P":
                color = "gray"
            elif value == "D":
                color = "gray"
            elif value == "M":
                color = "gray"
            elif value == "O":
                color = "green"
            elif value == "T":
                color = "darkred"
            elif value == "#":
                color = "black"

            self.ax.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor=color))

            tile_color = TILE_COLORS.get(value)
            if tile_color is not None:
                if (y, x) in state.pot_states:
                    tile_color = tile_color[int(state.pot_states[y, x])]
                elif (y, x) in state.desk_states:
                    tile_color = tile_color[int(state.desk_states[y, x])]

                self.ax.add_patch(patches.Circle((x + 0.5, y + 0.5), 0.15, color=tile_color))

            # Add text labels
            if value not in "S #":
                self.ax.text(x + 0.5, y + 0.5, value, va='center', ha='center')

        # Draw players
        p1_pos, p2_pos = state.p1, state.p2
        p1_state, p2_state = state.player_states

        self.ax.add_patch(patches.Circle((p1_pos[1] + 0.5, p1_pos[0] + 0.5), 0.3, color='cyan', label="P1"))
        self.ax.add_patch(patches.Circle((p1_pos[1] + 0.5, p1_pos[0] + 0.5), 0.15, color=PLAYER_COLORS[p1_state]))

        self.ax.add_patch(patches.Circle((p2_pos[1] + 0.5, p2_pos[0] + 0.5), 0.3, color='tomato', label="P2"))
        self.ax.add_patch(patches.Circle((p2_pos[1] + 0.5, p2_pos[0] + 0.5), 0.15, color=PLAYER_COLORS[p2_state]))

        # Draw state info
        self.ax.set_title(f"Delivered Soups: {cumulated}")
        self.ax.set_xlim(0, self.layout.shape[1])
        self.ax.set_ylim(0, self.layout.shape[0])
        self.ax.invert_yaxis()
        self.ax.set_aspect('equal')
        self.ax.legend(loc='upper right')

        self.ax.axis("off")
        f = f"/tmp/cooking_frame_{step}.png"
        plt.savefig(f)
        return f

    def rollout_visualisation(self, name=""):
        frames = []
        cumulated = 0
        for idx, (state, reward) in enumerate(self.rollout):
            cumulated += reward
            frames.append(self.render_state(state, cumulated, idx))

        with imageio.get_writer(f"{name}.gif", mode="I", duration=0.5) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                writer.append_data(image)
                os.remove(frame)


if __name__ == '__main__':

    layout = "small"

    episode_length = 200 # for short gifs
    mdp = LightCollaborativeCooking(episode_length=episode_length, layout=layout)

    for f1, f2 in [(0., 1.), (0., 0.5), (1., 0.5)]:

        policy_1, policy_2, V_1, V_2 = compute_soft_nash_equilibrium(
                mdp=mdp,
                gamma=0.9,
                lambda_reg=0.05, # hidden entropy regularisation parameter
                f1=f1,
                f2=f2,
                tol=1.,
                damping=0.1,
                max_iters=500
        )

        with open(f"data/cooking_equilibrium_{layout}_{f1}_{f2}.pkl", "wb") as f:
            pickle.dump((policy_1, policy_2), f)


        rollout = mdp.rollout(policy_1, policy_2)

        crollout = [
            (mdp.state_meanings[timestep.state], (timestep.reward_1 + timestep.reward_2)) for timestep in rollout
        ]
        vis = RolloutVisualiser(
            layout=layouts[layout],
            rollout=crollout,
        )
        vis.rollout_visualisation(name=f"data/movie_{layout}_{f1}_{f2}")



