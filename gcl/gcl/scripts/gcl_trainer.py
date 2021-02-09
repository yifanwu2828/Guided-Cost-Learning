import os
import sys
import time

import gym
import gym_nav
import numpy as np 
import torch

import pytorch_util as ptu 
import utils

# TODO: set global variables for saved videos


class GCL_Trainer():

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params
        self.params = params
        # TODO: create logger
        # self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)

        # TODO: set up plotting

        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim


        # TODO: set up simulation timestep for saved videos

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy=None,
                          expert_data=None, expert_policy=None):
        """
        :param n_iter:  number of iterations
        :param collect_policy: q_k
        :param expert_data: D_demo
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        
        # add demonstrations to replay buffer
        demo_paths = self.collect_demo_trajectories(expert_data, expert_policy)
        self.agent.add_to_buffer(demo_paths, demo=True)

        for itr in range(n_iter):
            print("\n********** Iteration {} ************".format(itr))

            # TODO: set up logging

            # Generate samples D_traj from current trajectory distribution q_k
            paths, train_video_paths = self.collect_training_trajectories(
                collect_policy, self.params['batch_size']
            )

            # Append samples D_traj to D_samp
            self.agent.add_to_buffer(paths)

            # Use D_{samp} to update cost c_{\theta}
            self.train_reward()

            # Update q_k(\tau) using D_{traj} and use Guided policy search to obtain q_{k+1}(\tau)
            self.train_policy()

    def collect_demo_trajectories(self, expert_data, expert_policy):
        """
        :param expert_data:  relative path to saved 
        :param expert_policy:  relative path to saved expert policy
        :return:
            paths: a list of trajectories
        """
        # Load expert policy or expert demonstrations D_demo
        if expert_data:
            print('\nLoading saved demonstrations...')
            with open(expert_data, 'rb') as f:
                demo_paths = pickle.load(f)
            # TODO: sample self.params['demo_size'] from demo_paths
        elif expert_policy:
            # TODO: make this to accept other expert policies
            from stable_baselines3 import PPO
            expert_policy = PPO.load(expert_policy)
            print('\nRunning expert policy to collect demonstrations...')
            demo_paths = utils.sample_trajectories(
                self.env, 
                expert_policy, 
                batch_size=self.params['demo_size'], 
                expert=True
            )
        else:
            raise ValueError('Please provide either expert demonstrations or expert policy')
        return demo_paths

    def collect_training_trajectories(self, collect_policy, batch_size):
        """
        :param collect_policy:  the current policy which we use to collect data
        :param batch_size:  the number of trajectories to collect
        :return:
            paths: a list trajectories
            train_video_paths: paths which also contain videos for visualization purposes
        """
        print("\nCollecting sample trajectories to be used for training ...")
        paths = utils.sample_trajectories(self.env, collect_policy, batch_size)

        # TODO: add logging and training videos
        train_video_paths = None
        return paths, train_video_paths


    def train_reward(self):
        """
        Algorithm 2: Nonlinear IOC with stochastic gradients 
        """
        print("\nUpdating reward parameters")
        reward_logs = []
        for k in range(self.params['num_reward_train_steps_per_iter']):
            # Sample demonstration batch D^_{demo} \subset D_{demo}
            # Sample background batch D^_{samp} \subset D_{sample}
            demo_batch = self.agent.sample_rollouts(self.params['train_demo_batch_size'], demo=True)
            sample_batch = self.agent.sample_rollouts(self.params['train_sample_batch_size'])

            # Use the sampled data to train the reward function
            reward_log = self.agent.train_reward(demo_batch, sample_batch)
            reward_logs.append(reward_log)
        return reward_logs

    def train_policy(self):
        """
        Guided policy search or PG
        """
        print('\nTraining agent using sampled data from replay buffer...')
        train_logs = []
        for train_step in range(self.params['num_policy_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            train_log = self.agent.train_policy(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            train_logs.append(train_log)
        return train_logs