import argparse
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import gym_nav

try:
    from icecream import ic
    from icecream import install
    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from gcl.infrastructure.rl_trainer import RL_Trainer
from gcl.agents.pg_agent import PGAgent
from gcl.infrastructure.utils import tic, toc


class PG_Trainer(object):

    def __init__(self, params: dict):
        #####################
        # SET AGENT PARAMS
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
        # RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        policy_log_lst = self.rl_trainer.run_training_loop(
            n_iter=params['n_iter'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )
        return policy_log_lst



if __name__ == '__main__':
    # set overflow warning to error instead
    np.seterr(all='raise')
    torch.autograd.set_detect_anomaly(True)

    torch.backends.cudnn.benchmark = True


    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-env', type=str, default='FetchReach-v1')
    parser.add_argument('--exp_name', '-exp', type=str, default='fetch_env_irl')

    # relative to where you're running this script from
    parser.add_argument('--expert_policy', '-epf', type=str, default='')
    parser.add_argument('--expert_data', '-ed', type=str, default='')

    parser.add_argument('--reward_to_go', '-rtg', action='store_true', default=True)
    parser.add_argument('--nn_baseline', action='store_true', default=True)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true', default=False)

    parser.add_argument('--n_iter', '-n', type=int, default=200)
    parser.add_argument(
        '--num_policy_train_steps_per_iter', type=int, default=1,
        help='Number of policy updates per iteration')

    parser.add_argument('--batch_size', '-b', type=int, default=1000)  # steps collected per train iteration
    parser.add_argument(
        '--train_batch_size', type=int, default=1000,
        help='Number of transition steps to sample from replay buffer per policy update'
    )
    parser.add_argument(
        '--eval_batch_size', type=int, default=100,
        help='Number of transition steps to sample from current policy for evaluation'
    )

    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=100)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    parser.add_argument('--ep_len', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)  # -1 not log video
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--save_params', action='store_true', default=False)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)


    ##################################
    # CREATE DIRECTORY FOR LOGGING
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
    params["no_gpu"] = True

    params['n_iter'] = 200
    params["batch_size"] = 10000
    params["train_batch_size"] = params["batch_size"]
    params['num_policy_train_steps_per_iter'] = 1    # >1 do not work

    params['discount'] = 0.99
    params["learning_rate"] = 5e-3
    ic(params)


    ###################
    # RUN TRAINING
    ###################
    start_train = tic()
    trainer = PG_Trainer(params)
    policy_log_lst = trainer.run_training_loop()
    toc(start_train)

    plt.figure()
    plt.plot(policy_log_lst)
    plt.title("policy_loss")
    plt.show()

    # saving mlp Policy
    SAVE = False
    fname2 = "test_pg_fetch.pth"
    if SAVE:
        policy_model = trainer.rl_trainer.agent.actor
        torch.save(policy_model, fname2)
        del policy_model

    policy_model = torch.load(fname2)
    policy_model.eval()