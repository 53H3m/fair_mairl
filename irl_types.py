import os
import pickle
from collections.abc import Callable
from typing import NamedTuple, Tuple, Any, Dict, List

import numpy as np
import tree

from games.mdp import TimeStep

array = np.array
class IRLParams(NamedTuple):
    fairness_levels: Tuple[float, ...]
    reward_func: np.ndarray | Callable

    def to_str(self):
        """Ensure it can be reconstructed from its string representation."""
        return f"IRLParams(fairness_levels={self.fairness_levels}, reward_func={repr(self.reward_func)})"

    @staticmethod
    def from_str(string: str):
        return eval(string)


class Demo(NamedTuple):
    state: Any
    agent_1_idx: int
    agent_2_idx: int
    action_1: Any
    action_2: Any
    done: bool


    @classmethod
    def from_timesteps(
            cls,
            agent_1_idx: int,
            agent_2_idx: int,
            timesteps: TimeStep
    ):

        return cls(
            state=timesteps.state,
            agent_1_idx=np.full(timesteps.action_1.shape, agent_1_idx, dtype=np.int32),
            agent_2_idx=np.full(timesteps.action_1.shape, agent_2_idx, dtype=np.int32),
            action_1=timesteps.action_1,
            action_2=timesteps.action_2,
            done=timesteps.done
        )

    def append(self, other):

        return tree.map_structure(
            lambda *ds : np.concatenate(ds, axis=0),
            self, other
        )

    @staticmethod
    def merge(demo_dict: Dict):
        return tree.map_structure(
            lambda *ds : np.concatenate(ds, axis=0),
            *demo_dict.values()
        )

    def batched(self):
        # TODO, for now assume same length for all episodes
        num_episodes = np.sum(self.done)
        def time_major(d):
            shape = d.shape
            return np.reshape(d, (shape[0] // num_episodes, num_episodes) + shape[1:])

        return tree.map_structure(
            time_major,
            self
        )




def tuple_mean(tuples):
    return tree.map_structure(lambda *v: np.mean(v, axis=0), *tuples)

class GridDomain:
    def __init__(
            self,
            name: str,
            dim: int,
            step_size: float,
            bounds: Tuple[float, float],
            dtype=np.float32,
    ):
        self.name = name
        self.dtype = dtype
        self.dim = dim
        self.step_size = step_size
        self.bounds = bounds

    def walk(self, prev_value):
        step = np.random.choice([-self.step_size, 0, self.step_size], self.dim, p=(0.15, 0.7, 0.15))
        #step = np.random.normal(0, self.step_size, self.dim)
        return self.dtype(np.clip(prev_value + step, *self.bounds).squeeze())

    def pdf(self):
        return 1.

    def prior(self):
        if self.dtype == int:
            return np.random.randint(self.bounds[0], self.bounds[1]+1, self.dim).squeeze()
        d = self.bounds[1] - self.bounds[0]
        n = round(d / self.step_size)
        possible_initial_values = np.linspace(*self.bounds, n+1)
        return np.random.choice(possible_initial_values, self.dim).squeeze()

    def update_step_size(self, percent):
        delta = percent * (self.bounds[1] - self.bounds[0])
        self.step_size = self.dtype(np.clip(self.step_size + delta, *self.bounds))


def get_domains_for(
        params: IRLParams,
        fairness_bounds = (0, 1),
        fairness_step_size = 0.1,
        reward_bounds=(0, 1),
        reward_step_size=0.1,
):

    fairness_domains = tuple(GridDomain(
        name=f"fairness_level_{i}",
        dim=1,
        step_size=fairness_step_size,
        bounds=fairness_bounds,
    ) for i, level in enumerate(params.fairness_levels))

    rewards_domain = GridDomain(
        name=f"reward_func",
        dim=params.reward_func.shape,
        step_size=reward_step_size,
        bounds=reward_bounds,
    )

    return IRLParams(fairness_levels=fairness_domains, reward_func=rewards_domain)

if __name__ == '__main__':

    x = IRLParams(fairness_levels=(0., 0.2, 0.4), reward_func=np.random.random((5,5)))
    rep = x.to_str()

    y = eval(rep)

    print(y)


class RunData(NamedTuple):
    solution: IRLParams
    mg_config: dict
    reward_samples: List[IRLParams]
    metrics: dict

    policy_samples: None | List[np.ndarray] = None


class Config(NamedTuple):
    num_trajectories_per_pairing: int
    pool_size: int

    sampling_algo: str
    sampling_method: str

    seed: None | int = None
    gap_function: None | str = None
    prior: None | str = None

    exp_type: str = "random_mg"

    def run_type(self):
        string = f"{self.sampling_algo}_{self.sampling_method}"
        if self.sampling_method == "PORP":
            string += f"_{self.gap_function}"

        return string

    def run_name(self, include_seed=True):

        if include_seed:
            string = f"{self.seed}_{self.num_trajectories_per_pairing}_{self.pool_size}"
        else:
            string = f"{self.num_trajectories_per_pairing}_{self.pool_size}"

        return string

    def unseeded_path(self):
        return f"data/{self.exp_type}/{self.run_type()}/{self.run_name(include_seed=False)}_eval.pkl"

    def build_dir(self):
        path = f"data/{self.exp_type}/{self.run_type()}"
        if not os.path.exists(path):
            os.mkdir(path)

    def run_exists(self):
        return os.path.exists(f"data/{self.exp_type}/{self.run_type()}/{self.run_name()}.pkl")

    def save_run(self, run_data: RunData):
        self.build_dir()

        path = f"data/{self.exp_type}/{self.run_type()}/{self.run_name()}.pkl"

        with open(path, "wb") as f:
            pickle.dump(run_data, f)

    @staticmethod
    def save_eval(path: str, evaluation: dict):
        with open(path, "wb") as f:
            pickle.dump(evaluation, f)

    def load_eval(self):
        with open(self.unseeded_path(), "rb") as f:
            e = pickle.load(f)
        return e

    def load_run(self) -> RunData:
        try:
            path = f"data/{self.exp_type}/{self.run_type()}/{self.run_name()}.pkl"
            with open(path, "rb") as f:
                run_data = pickle.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Could not load checkpoint for config {self}: {e}")

        return run_data


class CookingConfig(Config):
    num_trajectories_per_pairing: int
    pool_size: int

    sampling_algo: str
    sampling_method: str

    seed: None | int = None
    gap_function: None | str = None

    exp_type: str = "cooking"

    use_prior: bool = False

class MHConfig(NamedTuple):
    num_samples: int
    step_size: float = 0.1
    warmup_frac: float = 0.25
    max_step_size: float = 1.
    min_step_size: float = 0.01
    increment_size: float = 0.01
