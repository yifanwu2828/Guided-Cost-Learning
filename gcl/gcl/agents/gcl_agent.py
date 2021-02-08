from gcl.agents.mlp_policy import MLPPolicyPG
from gcl.agents.base_agent import BaseAgent 
from gcl.agents.mlp_reward import MLPReward
from gcl.scripts.replay_buffer import ReplayBuffer

class GCL_Agent(BaseAgent):
    def __init__(self, env, agent_params):
        super(GCL_Agent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
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
        self.demo_buffer = ReplayBuffer(1e6)
        self.sample_buffer = ReplayBuffer(1e6)

    def train(self):
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
        ## HINT: `train_log` should be returned by your actor update method
        train_log = self.actor.update(observations, actions, advantages, q_values)

        return train_log


    def add_to_buffer(self, paths, demo=False):
        """
        Add paths to demo or sample buffer
        """
        if demo:
            self.demo_buffer.add_rollouts(paths)
        else:
            self.sample_buffer.add_rollouts(paths)

    def sample(self, batch_size, demo=False):
        """
        Sample paths from demo or sample buffer
        """
        if demo:
            return self.demo_buffer.sample_recent_data(batch_size)
        else:
            return self.sample_buffer.sample_recent_data(batch_size)
