from abc import ABCMeta
from typing import Optional, List, Union
from functools import reduce
from itertools import accumulate

import numpy as np

from .mlp_policy import MLPPolicyPG
from .base_agent import BaseAgent
from .mlp_reward import MLPReward
from gcl.scripts.replay_buffer import ReplayBuffer
from gcl.scripts.utils import PathDict, normalize


class GCL_Agent(BaseAgent, metaclass=ABCMeta):
    gamma: float
    standardize_advantages: bool
    nn_baseline: bool
    reward_to_go: bool

    def __init__(self, env, agent_params: dict):
        super(GCL_Agent, self).__init__()

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
        # TODO: Add Guided Policy Search (GPS) policy

        # reward function
        self.reward = MLPReward(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['output_size'],
            learning_rate=self.agent_params['learning_rate']
        )

        # Replay buffers: demo holds expert demonstrations and sample holds policy samples
        self.demo_buffer = ReplayBuffer(1000000)
        self.sample_buffer = ReplayBuffer(1000000)
        self.background_buffer = ReplayBuffer(1000000)

    def train_reward(self, demo_batch: np.ndarray, sample_batch: np.ndarray) -> dict:
        """
        Train the reward function
        :param demo_batch: demo rollouts
        :param sample_batch: sample rollouts
        :return: reward_log
        :type: dict
        """
        # unpack rollouts into obs, act, log_probs
        demo_obs = [demo_path['observation'] for demo_path in demo_batch]
        demo_acs = [demo_path['action'] for demo_path in demo_batch]
        sample_obs = [sample_path['observation'] for sample_path in sample_batch]
        sample_acs = [sample_path['action'] for sample_path in sample_batch]
        sample_log_probs = [sample_path['log_prob'] for sample_path in sample_batch]

        # Estimate gradient loss and update parameters
        reward_log = self.reward.update(demo_obs, demo_acs, sample_obs, sample_acs, sample_log_probs)

        return reward_log

    ##################################################################################################

    def train_policy(self,
                     observations: np.ndarray,
                     actions: np.ndarray,
                     rewards_list: Union[np.ndarray, List],
                     next_observations, terminals) -> dict:
        """
        Training a PG agent refers to updating its actor using the given observations/actions
        and the calculated q_vals/advantages that come from the seen rewards.
        """
        # step 1: calculate q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values = self.calculate_q_vals(rewards_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point
        advantages = self.estimate_advantage(observations, q_values)

        # step 3: use all datapoints (s_t, a_t, q_t, adv_t) to update the PG actor/policy
        train_log = self.actor.update(observations, actions, advantages, q_values)

        return train_log

    def calculate_q_vals(self, rewards_list: List[List[float]]) -> np.ndarray:
        """
        Monte Carlo estimation of the Q function.
        """
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

    def estimate_advantage(self, obs, q_values: np.ndarray) -> np.ndarray:
        """
        Computes advantages by subtracting a baseline from the estimated Q values
        """
        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the baseline
        if self.nn_baseline:
            baselines_unnormalized = self.actor.run_baseline_prediction(obs)

            # ensure that the baseline and q_values have the same dimensionality
            # to prevent silent broadcasting errors
            q_values = q_values.reshape(-1)
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
            advantages = normalize(q_values, np.mean(q_values), np.std(q_values))

        return advantages

    #####################################################
    #####################################################

    def add_to_buffer(self, paths: List[PathDict], demo=False, background=False) -> None:
        """
        Add paths to demo or sample buffer
        """
        if demo:
            self.demo_buffer.add_rollouts(paths)
        elif background:
            self.background_buffer.add_rollouts(paths)
        else:
            self.sample_buffer.add_rollouts(paths)

    #####################################################
    #####################################################

    def sample_rollouts(self, num_rollouts: int, demo=False) -> np.ndarray:
        """
        Randomly sample paths from demo or sample buffer
        :param: num_rollouts
        :param: if demo sample from demo buffer, else sample from sample buffer
        :return: random rollouts (paths) from buffer
        """
        assert num_rollouts > 0
        if demo:
            return self.demo_buffer.sample_random_rollouts(num_rollouts)
        else:
            return self.sample_buffer.sample_random_rollouts(num_rollouts)


    def sample_recent_rollouts(self, num_rollouts: int, demo=False) -> np.ndarray:
        """
        Sample recent paths from demo or sample buffer
        :param: num_rollouts
        :param: if demo sample from demo buffer, else sample from sample buffer
        :return: random rollouts (paths) from buffer
        """
        assert num_rollouts > 0 and isinstance(num_rollouts, int)
        if demo:
            return self.demo_buffer.sample_recent_rollouts(num_rollouts)
        else:
            return self.sample_buffer.sample_recent_rollouts(num_rollouts)


    def sample_background_rollouts(self, batch_size: Optional[int] = 1000,
                                   recent=False, all_rollouts=False) -> np.ndarray:
        assert not (recent and all_rollouts)
        if all_rollouts:
            return self.background_buffer.sample_all_rollouts()
        elif recent:
            assert isinstance(batch_size, int) and batch_size >= 0
            return self.background_buffer.sample_recent_rollouts(batch_size)
        else:
            assert isinstance(batch_size, int) and batch_size >= 0
            return self.background_buffer.sample_random_rollouts(batch_size)

    #####################################################
    #####################################################

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

        discounted_return = reduce(
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
