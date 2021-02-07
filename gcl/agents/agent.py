import numpy as np 

from base_agent import BaseAgent 


class IRLAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(IRLAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params



        # actor/policy
        self.actor = MLPPolicy(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'])

        # Replay buffer
        self.replay_buffer = ReplayBuffer(1e6)

    def train(self):
        pass

    def add_ro_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
