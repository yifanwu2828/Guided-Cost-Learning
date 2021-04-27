import argparse
import os
import time
from pprint import pprint

import numpy as np
import torch
import matplotlib.pyplot as plt

from gcl.infrastructure.gcl_trainer import GCL_Trainer
from gcl.agents.gcl_agent import GCL_Agent
from gcl.infrastructure.utils import tic, toc


class IRL_Trainer(object):

    def __init__(self, params):
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
        self.params['agent_class'] = GCL_Agent
        self.params['agent_params'] = agent_params

        ################
        # IRL TRAINER
        ################

        self.gcl_trainer = GCL_Trainer(self.params)

    def run_training_loop(self) -> tuple:
        train_log_lst, policy_log_lst = self.gcl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy=self.gcl_trainer.agent.actor,
            eval_policy=self.gcl_trainer.agent.actor,
            expert_data=self.params['expert_data'],
            expert_policy=self.params['expert_policy']
        )
        return train_log_lst, policy_log_lst


def removeOutliers(x, outlierConstant=1.5) -> list:
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

    result = a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

    return result.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-env', type=str, default='NavEnv-v0')
    parser.add_argument('--exp_name', '-exp', type=str, default='nav_env_irl')

    # relative to where you're running this script from
    parser.add_argument('--expert_policy', '-epf', type=str, default='ppo_nav_env')
    parser.add_argument('--expert_data', '-ed', type=str, default='')

    # PG setting
    parser.add_argument('--reward_to_go', '-rtg', action='store_true', default=True)
    parser.add_argument('--nn_baseline', action='store_true', default=True)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true', default=False)

    parser.add_argument(
        '--n_iter', '-n', type=int, default=10,
        help='Number of total iterations in outer training loop (Algorithm 1: Guided cost learning)')
    parser.add_argument(
        '--demo_size', type=int, default=50,
        help='Number of expert rollouts to add to demo replay buffer'
    )
    parser.add_argument(
        '--sample_size', type=int, default=10,
        help='Number of current policy rollouts to add to replay buffer at each iteration'
    )
    parser.add_argument(
        '--num_reward_train_steps_per_iter', type=int, default=10,
        help='Number of reward updates per iteration in Algorithm 2: Nonlinear IOC with stochastic gradients'
    )
    parser.add_argument(
        '--train_demo_batch_size', type=int, default=10,
        help='Number of expert rollouts (subset of D_demo) to sample from replay buffer per reward update'
    )
    parser.add_argument(
        '--train_sample_batch_size', type=int, default=10,
        help='Number of policy rollouts (subset of D_samp) to sample from replay buffer per reward update'
    )
    parser.add_argument(
        '--num_policy_train_steps_per_iter', type=int, default=1,
        help='Number of policy updates per iteration')
    parser.add_argument(
        '--train_batch_size', type=int, default=1000,
        help='Number of transition steps to sample from replay buffer per policy update'
    )   # use at least 5000 for 100 policy update itr
    parser.add_argument(
        '--eval_batch_size', type=int, default=100,
        help='Number of transition steps to sample from current policy for evaluation'
    )

    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=20)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    parser.add_argument('--ep_len', type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--samp_recent', action='store_true', default=False,
                        help='sample random data or recent data from D_samp in train reward'
                        )
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
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    # RUN TRAINING
    ###################
    print("##### PARAM ########")
    # path of pretrain model
    path = os.getcwd()
    params["no_gpu"] = True  # False
    params["expert_policy"] = "../model/sac_nav_env"
    params["ep_len"] = 100
    params['samp_recent'] = False

    '''Outer Training Loop (Algorithm 1: Guided cost learning)'''
    # Number of iteration of outer training loop (Algorithm 1)
    params['n_iter'] = 150  # sweet spot 77
    # Number of expert rollouts to add to demo replay buffer before outer loop
    params['demo_size'] = 200
    # number of current policy rollouts add to sample buffer per itr in outer training loop
    # params["sample_size"] = 100

    ''' Train Reward (Algorithm 2) '''
    # Number of reward updates per iteration in Algorithm 2
    params["num_reward_train_steps_per_iter"] = 10  # 10 K_r
    # Number of expert rollouts to sample from replay buffer per reward update
    params["train_demo_batch_size"] = 100
    # Number of policy rollouts to sample from replay buffer per reward update
    params["train_sample_batch_size"] = 100  # 100


    ''' Train Policy (Policy Gradient) '''
    # Number of policy updates per iteration
    params["num_policy_train_steps_per_iter"] = 1  # K_p
    # Number of transition steps to sample from sample replay buffer per policy update
    # equivalent to number of transition steps collect in outer loop
    params["train_batch_size"] = 10_000  # 10_000



    # size of subset should be less than size of set
    # assert params["sample_size"] >= params["train_sample_batch_size"]
    assert params['demo_size'] >= params["train_demo_batch_size"]
    assert params["train_batch_size"] >= params["train_sample_batch_size"] * params["ep_len"]

    params['discount'] = 0.99
    params["learning_rate"] = 1e-3
    params['reward_to_go'] = True
    params['nn_baseline'] = True
    params['dont_standardize_advantages'] = False
    pprint(params)


    trainer = IRL_Trainer(params)
    start_train = tic()
    train_log_lst, policy_log_lst = trainer.run_training_loop()
    toc(start_train, ftime=True)

    ###################
    # Test
    ###################

    res = removeOutliers(train_log_lst)
    plt.figure()
    plt.plot(res)
    plt.title("train_loss without outlier")
    plt.plot(list(range(len(train_log_lst))), [train_log_lst[0]] * len(train_log_lst))
    plt.show()

    plt.figure()
    plt.plot(train_log_lst)
    plt.title("train_loss")
    plt.plot(list(range(len(train_log_lst))), [train_log_lst[0]] * len(train_log_lst))
    plt.show()

    plt.figure()
    plt.plot(policy_log_lst)
    plt.title("policy_loss")
    plt.show()

    # saving mlp Reward and Policy
    SAVE = True
    if SAVE:
        fname1 = "../model/test_gcl_reward_GPU.pth"
        reward_model = trainer.gcl_trainer.agent.reward
        torch.save(reward_model, fname1)

        fname2 = "../model/test_gcl_policy_GPU.pth"
        policy_model = trainer.gcl_trainer.agent.actor
        torch.save(policy_model, fname2)


if __name__ == '__main__':
    print(torch.__version__)
    # set overflow warning to error instead
    # np.seterr(all='raise')
    main()
    print("Done!")

