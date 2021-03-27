import argparse
import os
import time

import numpy as np
import torch
# from torch import multiprocessing
import matplotlib.pyplot as plt
import gym
import gym_nav
from rl_trainer import RL_Trainer
from gcl.agents.pg_agent import PGAgent
from utils import tic, toc


class PG_Trainer(object):

    def __init__(self, params):
        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'output_size': params['output_size'],
            'learning_rate': params['learning_rate'],
        }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not (params['dont_standardize_advantages']),
            'reward_to_go': params['reward_to_go'],
            'nn_baseline': params['nn_baseline'],
        }

        train_args = {
            'num_policy_train_steps_per_iter': params['num_policy_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = PGAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )


if __name__ == '__main__':
    # set overflow warning to error instead
    np.seterr(all='raise')
    torch.autograd.set_detect_anomaly(True)
    # multiprocessing.set_start_method('fork')

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-env', type=str, default='NavEnv-v0')
    parser.add_argument('--exp_name', '-exp', type=str, default='testPG')

    # relative to where you're running this script from
    parser.add_argument('--expert_policy', '-epf', type=str, default='ppo_nav_env')
    parser.add_argument('--expert_data', '-ed', type=str, default='')

    parser.add_argument(
        '--n_iter', '-n', type=int, default=10,
        help='Number of total iterations')
    parser.add_argument(
        '--demo_size', type=int, default=10,
        help='Number of expert rollouts to add to replay buffer'
    )
    parser.add_argument(
        '--batch_size', type=int, default=10,
        help='Number of current policy rollouts to add to replay buffer at each iteration'
    )
    parser.add_argument(
        '--num_reward_train_steps_per_iter', type=int, default=10,
        help='Number of reward updates per iteration'
    )
    parser.add_argument(
        '--train_demo_batch_size', type=int, default=10,
        help='Number of expert rollouts to sample from replay buffer per reward update'
    )
    parser.add_argument(
        '--train_sample_batch_size', type=int, default=10,
        help='Number of policy rollouts to sample from replay buffer per reward update'
    )
    parser.add_argument(
        '--num_policy_train_steps_per_iter', type=int, default=10,
        help='Number of policy updates per iteration')
    parser.add_argument(
        '--train_batch_size', type=int, default=1000,
        help='Number of transition steps to sample from replay buffer per policy update'
    )
    parser.add_argument(
        '--eval_batch_size', type=int, default=100,
        help='Number of transition steps to sample from current policy for evaluation'
    )

    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=20)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)  # -1 not log video
    parser.add_argument('--scalar_log_freq', type=int, default=-1)
    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)


    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Setting param
    params['n_iter'] = 20

    params["num_reward_train_steps_per_iter"] = 100  # K_r
    params["num_policy_train_steps_per_iter"] = 800  # K_p

    params['demo_size'] = 200                # number of rollouts add to demo buffer per itr in outer loop
    params["sample_size"] = 100               # number of rollouts add to sample buffer per itr in outer loop

    params["train_demo_batch_size"] = 100   # number of rollouts sample from demo buffer in train reward
    params["train_sample_batch_size"] = 100  # number of rollouts sample from sample buffer in train reward

    assert params["sample_size"] >= params["train_sample_batch_size"]
    assert params['demo_size'] >= params["train_demo_batch_size"]

    params["batch_size"] = 100*20
    params["train_batch_size"] = params["batch_size"]

    params['discount'] = 0.99
    params["learning_rate"] = 5e-3
    print(params)


    ###################
    ### RUN TRAINING
    ###################
    start_train = tic()
    trainer = PG_Trainer(params)
    trainer.run_training_loop()
    toc(start_train)