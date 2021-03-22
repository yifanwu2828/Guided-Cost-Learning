import argparse
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import gym_nav
from gcl_trainer import GCL_Trainer
from gcl.agents.gcl_agent import GCL_Agent
from utils import tic, toc


class IRL_Trainer():

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
        }

        train_args = {
            'num_policy_train_steps_per_iter': params['num_policy_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = GCL_Agent
        self.params['agent_params'] = agent_params

        ################
        ## IRL TRAINER
        ################

        self.gcl_trainer = GCL_Trainer(self.params)

    def run_training_loop(self):
        train_log_lst, policy_log_lst = self.gcl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy=self.gcl_trainer.agent.actor,
            eval_policy=self.gcl_trainer.agent.actor,
            expert_data=self.params['expert_data'],
            expert_policy=self.params['expert_policy']
        )
        return train_log_lst, policy_log_lst


def removeOutliers(x, outlierConstant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

    result = a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

    return result.tolist()


if __name__ == '__main__':
    # set overflow warning to error instead
    np.seterr(all='raise')
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='NavEnv-v0')
    parser.add_argument('--exp_name', type=str, default='nav_env_irl')
    parser.add_argument('--expert_policy', type=str,
                        default='ppo_nav_env')  # relative to where you're running this script from

    parser.add_argument('--expert_data', type=str, default='')  # relative to where you're running this script from
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

    # change path of pretrain model
    path = os.getcwd()
    # print (os.path.join(path,"tmp", "ppo_nav_env"))

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
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
    ### RUN TRAINING
    ###################
    print("##### PARAM ########")
    # params["expert_policy"] = os.path.join(path, "tmp/demo_agent", params["expert_policy"])
    params["expert_policy"] = os.path.join(path, "tmp/demo_agent", "a2c_nav_env")
    params['n_iter'] = 20
    # Number of expert rollouts to add to replay buffer
    params['demo_size'] = 100
    params['discount'] = 0.99
    params["learning_rate"] = 5e-3
    print(params)

    # Number of current policy rollouts to add to replay buffer at each iteration
    # Number of reward updates per iteration
    # Number of policy updates per iteration
    # Number of expert rollouts to sample from replay buffer per reward update
    # Number of policy rollouts to sample from replay buffer per reward update
    # Number of transition steps to sample from replay buffer per policy update PG

    params["batch_size"] = 20
    params["num_reward_train_steps_per_iter"] = 10  # K_r
    params["num_policy_train_steps_per_iter"] = 10  # K_p
    params["train_demo_batch_size"] = 100
    params["train_sample_batch_size"] = 100
    params["train_batch_size"] = 1000



    trainer = IRL_Trainer(params)
    start_train = tic()
    train_log_lst, policy_log_lst = trainer.run_training_loop()
    toc(start_train)

    ###################
    ### Test
    ###################

    res = removeOutliers(train_log_lst)
    plt.figure()
    plt.plot(res)
    plt.title("train_loss without outlier")
    # plt.ylim(-5000,500)
    plt.plot(list(range(len(train_log_lst))), [train_log_lst[0]] * len(train_log_lst))
    plt.show()

    plt.figure()
    plt.plot(train_log_lst)
    plt.title("train_loss")
    # plt.ylim(-5000,500)
    plt.plot(list(range(len(train_log_lst))), [train_log_lst[0]] * len(train_log_lst))
    plt.show()

    # plt.figure()
    # plt.plot(train_log_lst)
    # plt.title("train_loss_limit")
    # plt.ylim(-1000, 1000)
    # plt.plot(list(range(len(train_log_lst))), [train_log_lst[0]] * len(train_log_lst))
    # plt.show()

    plt.figure()
    plt.plot(policy_log_lst)
    plt.title("policy_loss")
    plt.show()

    # saving mlp Reward
    SAVE = True
    if SAVE:
        fname1 = "test_reward2.pth"
        reward_model = trainer.gcl_trainer.agent.reward
        torch.save(reward_model, fname1)

        fname2 = "test_policy2.pth"
        policy_model = trainer.gcl_trainer.agent.actor
        torch.save(policy_model, fname2)
