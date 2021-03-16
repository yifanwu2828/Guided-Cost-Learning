import numpy as np

from gcl.agents.mlp_policy import MLPPolicyPG
from gcl.agents.base_agent import BaseAgent
from gcl.agents.mlp_reward import MLPReward
from gcl.scripts.replay_buffer import ReplayBuffer
import gcl.scripts.utils as utils


class GCL_Agent(BaseAgent):
    def __init__(self, env, agent_params):
        super(GCL_Agent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            learning_rate=self.agent_params['learning_rate']
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
        self.demo_buffer = ReplayBuffer(1000)
        self.sample_buffer = ReplayBuffer(1000)

    def train_reward(self, demo_batch, sample_batch):
        """
        Train the reward function
        """
        demo_obs = np.array([demo['observation'] for demo in demo_batch])
        demo_acs = np.array([demo['action'] for demo in demo_batch])
        sample_obs = np.array([sample['observation'] for sample in sample_batch])
        sample_acs = np.array([sample['action'] for sample in sample_batch])
        sample_log_probs = np.array([sample['log_prob'] for sample in sample_batch])

        reward_log = self.reward.update(demo_obs, demo_acs, sample_obs, sample_acs, sample_log_probs)

        return reward_log

    def train_policy(self, observations, actions, rewards_list, next_observations, terminals):
        """
        Training a PG agent refers to updating its actor using the given observations/actions
        and the calculated qvals/advantages that come from the seen rewards.
        TODO: Add training for GPS policy
        """
        # step 1: calculate q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values = self.calculate_q_vals(rewards_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point
        advantages = self.estimate_advantage(observations, q_values)

        # TODO: step 3: use all datapoints (s_t, a_t, q_t, adv_t) to update the PG actor/policy
        # HINT: `train_log` should be returned by your actor update method
        train_log = self.actor.update(observations, actions, advantages, q_values)

        return train_log

    def calculate_q_vals(self, rewards_list):
        """
        Monte Carlo estimation of the Q function.
        """
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        # For each point (s_t, a_t), associate its value as being the discounted sum of rewards over the full trajectory
        # In other words: value of (s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        q_values = np.concatenate([self._discounted_cumsum(r) for r in rewards_list])

        return q_values

    def estimate_advantage(self, obs, q_values):
        """
        Computes advantages by subtracting a baseline from the estimated Q values
        """
        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the baseline
        baselines_unnormalized = self.actor.run_baseline_prediction(obs)  # V(s)

        # ensure that the baseline and q_values have the same dimensionality
        # to prevent silent broadcasting errors
        assert baselines_unnormalized.ndim == q_values.ndim

        # baseline was trained with standardized q_values, so ensure that the predictions
        # have the same mean and standard deviation as the current batch of q_values
        baselines = baselines_unnormalized * np.std(q_values) + np.mean(q_values)
        # TODO: compute advantage estimates using q_values and baselines
        advantages = q_values - baselines

        # Normalize the resulting advantages
        # TODO: standardize the advantages to have a mean of zero
        # and a standard deviation of one
        # HINT: there is a `normalize` function in `infrastructure.utils`
        advantages = utils.normalize(advantages, np.mean(advantages), np.std(advantages))

        return advantages

    #####################################################
    #####################################################

    def add_to_buffer(self, paths, demo=False):
        """
        Add paths to demo or sample buffer
        """
        if demo:
            self.demo_buffer.add_rollouts(paths)
        else:
            self.sample_buffer.add_rollouts(paths)

    def sample_rollouts(self, batch_size, demo=False):
        """
        Sample paths from demo or sample buffer
        """
        if demo:
            return self.demo_buffer.sample_random_rollouts(batch_size)
        else:
            return self.sample_buffer.sample_random_rollouts(batch_size)

    def sample(self, batch_size):
        """
        Sample transition steps of size batch_size
        """
        return self.demo_buffer.sample_recent_data(batch_size, concat_rew=False)


    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################


    def _discounted_cumsum(self, rewards):
        """
        Helper function which
        -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
        -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        # TODO: create `list_of_discounted_returns`
        # HINT1: note that each entry of the output should now be unique,
        # because the summation happens over [t, T] instead of [0, T]
        # HINT2: it is possible to write a vectorized solution, but a solution
        list_of_discounted_cumsums = [0] * len(rewards)
        s = 0
        for t, r in reversed(list(enumerate(rewards))):
            s += (self.gamma ** t) * r
            list_of_discounted_cumsums[t] = s
        return list_of_discounted_cumsums
