from abc import ABCMeta
from typing import List, Union
import functools
from itertools import accumulate

import numpy as np

from gcl.agents.mlp_policy import MLPPolicyPG
from gcl.agents.base_agent import BaseAgent
from gcl.scripts.replay_buffer import ReplayBuffer
import gcl.scripts.utils as utils
from gcl.scripts.utils import PathDict


class PGAgent(BaseAgent, metaclass=ABCMeta):
    gamma: float
    reward_to_go: bool

    def __init__(self, env, agent_params: dict):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']

        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.demo_buffer = ReplayBuffer(1000000)
        self.sample_buffer = ReplayBuffer(1000000)

    def train_policy(self, observations: np.ndarray, actions: np.ndarray, rewards_list: Union[np.ndarray, List],
                     next_observations, terminals) -> dict:
        """
        Training a PG agent refers to updating its actor using the given
        observations/actions and the calculated q_vals/advantages that come from
        the seen rewards.
        """
        # step 1: calculate q values of each (s_t, a_t) point, using rewards
        # (r_0, ..., r_t, ..., r_T)
        q_values = self.calculate_q_vals(rewards_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point
        advantages = self.estimate_advantage(observations, q_values)

        # step 3: use all datapoints (s_t, a_t, q_t, adv_t) to update the PG
        # actor/policy
        train_log = self.actor.update(observations, actions, advantages, q_values)

        return train_log

    ##################################################################################################

    def calculate_q_vals(self, rewards_list: List[List[float]]) -> np.ndarray:
        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        if not self.reward_to_go:

            # For each point (s_t, a_t), associate its value as being the discounted sum of rewards over the full traj
            # In other words: value of (s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values = np.concatenate([self._discounted_return(r) for r in rewards_list])

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:

            # For each point (s_t, a_t), associate its value as being the discounted sum of rewards over the full traj
            # In other words: value of (s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = np.concatenate([self._discounted_cumsum(r) for r in rewards_list])

        return q_values

    def estimate_advantage(self, obs: np.ndarray, q_values: np.ndarray):
        """
        Computes advantages by (possibly) subtracting a baseline from the
        estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the baseline
        if self.nn_baseline:
            baselines_unnormalized = self.actor.run_baseline_prediction(obs)

            # ensure that the baseline and q_values have the same dimensionality
            # to prevent silent broadcasting errors
            assert baselines_unnormalized.ndim == q_values.ndim

            # baseline was trained with standardized q_values, so ensure that the predictions
            # have the same mean and standard deviation as the current batch of q_values
            baselines = baselines_unnormalized * np.std(q_values) + np.mean(q_values)

            # compute advantage estimates using q_values and baselines
            advantages = q_values - baselines

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            advantages = utils.normalize(q_values, np.mean(q_values), np.std(q_values))

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths: List[PathDict], demo=False):
        """
        Add paths to demo or sample buffer
        """
        if demo:
            self.demo_buffer.add_rollouts(paths)
        else:
            self.sample_buffer.add_rollouts(paths)

    def sample(self, batch_size: int, demo=False):
        """
        Sample recent transition steps of size batch_size
        """
        assert isinstance(batch_size, int) and batch_size >= 0
        if demo:
            return self.demo_buffer.sample_recent_data(batch_size, concat_rew=False)
        else:
            return self.sample_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    #                  HELPER FUNCTIONS                 #
    #####################################################

    def _discounted_return(self, rewards: List[float]) -> List[float]:
        """
        Helper function
        Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single
        rollout of length T
        Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        discounted_return = functools.reduce(
            lambda ret, reward: ret * self.gamma + reward,
            reversed(rewards),
        )
        return [discounted_return] * len(rewards)

    def _discounted_cumsum(self, rewards: List[float]) -> List[float]:
        """
            Helper function which
            - takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            - and returns a list where the entry in each index t' is
              sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        return list(accumulate(
            reversed(rewards),
            lambda ret, reward: ret * self.gamma + reward,
        ))[::-1]
