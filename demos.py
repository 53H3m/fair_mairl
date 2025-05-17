import itertools
import os.path
import pickle
from collections import defaultdict
from typing import Tuple, Dict, List

import numpy as np

from games.mdp import RandomTwoPlayerMDP, TimeStep
from games.regularised_mdp_solver import compute_soft_nash_equilibrium, compute_best_response, compute_visitation, \
    compute_value
from irl_types import IRLParams, Demo
from games.tf_regularised_mdp_solver import compute_soft_nash_equilibrium_tf


class Demonstrations:

    def __init__(
            self,
            partial_mdp_config: dict,
            gamma: float,
            lambda_reg: float,
            true_mg = None
    ):
        self.partial_mdp_config = partial_mdp_config
        self.gamma = gamma
        self.lambda_reg = lambda_reg

        if true_mg is None:
            self.base_mdp = RandomTwoPlayerMDP(
                **self.partial_mdp_config,
            )
        else:
            self.base_mdp = true_mg

        self.values: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.equilibriums: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self.demonstrations: Dict[Tuple[int, int], Demo] = {}
        self.log_likelihoods = {}
        self.num_combinations = 0
        self.agent_pool_size = 0
        self.combinations: np.ndarray | None = None


    def get_demos(
            self,
            num_trajectories_per_pairing: int,
            truth: IRLParams,
            compute_values=False,
    ):
        self.agent_pool_size = len(truth.fairness_levels)

        combinations = list(itertools.combinations(
            range(self.agent_pool_size), 2
        ))

        true_mdp = RandomTwoPlayerMDP(
            **self.partial_mdp_config,
            **truth._asdict()
        )

        self.num_combinations = len(combinations)
        self.combinations = np.concatenate([combinations, np.arange(len(combinations))[:, np.newaxis]], axis=1, dtype=np.int32)

        for (i, j) in itertools.combinations(
                range(self.agent_pool_size), 2
        ):
            pi_a, pi_b, V_a, V_b = compute_soft_nash_equilibrium(
                mdp=true_mdp,
                gamma=self.gamma,
                lambda_reg=self.lambda_reg,
                f1=truth.fairness_levels[i],
                f2=truth.fairness_levels[j],
            )

            if num_trajectories_per_pairing > 0:
                rollouts = []
                for _ in range(num_trajectories_per_pairing):
                    rollouts.extend(true_mdp.rollout(pi_a, pi_b))
                if (i, j) not in self.demonstrations:
                    self.demonstrations[(i, j)] = Demo.from_timesteps(i, j, TimeStep.stack(rollouts))
                else:
                    self.demonstrations[(i, j)].append(Demo.from_timesteps(i, j, TimeStep.stack(rollouts)))

            self.equilibriums[(i, j)] = pi_a, pi_b

            if compute_values:
                vis = compute_visitation(pi_a, pi_b, true_mdp.transition_matrix, true_mdp.initial_state_dist,
                                         self.gamma)
                V_a = np.sum(vis * V_a)
                V_b = np.sum(vis * V_b)
                self.values[(i, j)] = V_a, V_b

    def get_value_for(
            self,
            truth: IRLParams,
    ):

        true_mdp = RandomTwoPlayerMDP(
            **self.partial_mdp_config,
            **truth._asdict()
        )

        values = {}

        for (i, j) in itertools.combinations(
                range(self.agent_pool_size), 2
        ):
            pi_a, pi_b = self.equilibriums[(i, j)]
            V_a, V_b = compute_value(
                policy_1=pi_a,
                policy_2=pi_b,
                mdp=true_mdp,
                gamma=self.gamma,
                f1=truth.fairness_levels[i],
                f2=truth.fairness_levels[j],
            )

            vis = compute_visitation(pi_a, pi_b, true_mdp.transition_matrix, true_mdp.initial_state_dist,
                                     self.gamma)
            V_a = np.sum(vis * V_a)
            V_b = np.sum(vis * V_b)

            values[(i, j)] = V_a, V_b

        return values

    def compute_loglikelihood(
            self,
            params: IRLParams
    ):
        mdp = RandomTwoPlayerMDP(
            **self.partial_mdp_config,
            **params._asdict()
        )

        log_likelihood = 0.

        for (i, j) in itertools.combinations(
            range(len(params.fairness_levels)), 2
        ):
            pi_a, pi_b = compute_soft_nash_equilibrium(
                mdp=mdp,
                gamma=self.gamma,
                lambda_reg=self.lambda_reg,  # suppose we know that
                f1=params.fairness_levels[i],
                f2=params.fairness_levels[j],
            )

            ts = self.demonstrations[(i, j)]
            pi_a_probs = pi_a[ts.state, ts.action_1]
            pi_b_probs = pi_b[ts.state, ts.action_2]

            log_likelihood += np.sum(np.log(pi_a_probs) + np.log(pi_b_probs))

        self.log_likelihoods[params.to_str()] = log_likelihood

    def log_likelihood(self, params: IRLParams) -> float:
        param_str = params.to_str()
        if param_str not in self.log_likelihoods:
            self.compute_loglikelihood(params)
        return self.log_likelihoods[param_str]


class CompiledDemonstrations(Demonstrations):

    def get_demos(
            self,
            num_trajectories_per_pairing: int,
            truth: IRLParams,
    ):
        self.agent_pool_size = len(truth.fairness_levels)

        combinations = list(itertools.combinations(
            range(self.agent_pool_size), 2
        ))

        self.num_combinations = len(combinations)
        self.combinations = np.concatenate([combinations, np.arange(len(combinations))[:, np.newaxis]], axis=1, dtype=np.int32)
        layout = self.base_mdp.layout
        for (i, j) in itertools.combinations(
                range(self.agent_pool_size), 2
        ):
            f1 = round(float(truth.fairness_levels[i]), 1)
            f2 = round(float(truth.fairness_levels[j]), 1)

            p = f"data/cooking_equilibrium_{layout}_{f1}_{f2}.pkl"
            if os.path.exists(p):
                with open(p, "rb") as f:
                    pi_a, pi_b = pickle.load(f)
            else:
                p = f"data/cooking_equilibrium_{layout}_{f2}_{f1}.pkl"
                with open(p, "rb") as f:
                    pi_b, pi_a = pickle.load(f)

            if num_trajectories_per_pairing > 0:
                rollouts = []
                for _ in range(num_trajectories_per_pairing):
                    rollouts.extend(self.base_mdp.rollout(pi_a, pi_b))
                if (i, j) not in self.demonstrations:
                    self.demonstrations[(i, j)] = Demo.from_timesteps(i, j, TimeStep.stack(rollouts))
                else:
                    self.demonstrations[(i, j)].append(Demo.from_timesteps(i, j, TimeStep.stack(rollouts)))

            self.equilibriums[(i, j)] = pi_a, pi_b


