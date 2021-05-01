import argparse
import os
import time
from pprint import pprint
from collections import OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt
from icecream import ic

from gym.wrappers import FilterObservation, FlattenObservation
from stable_baselines3 import A2C, SAC, PPO, HER

from gcl.infrastructure.sb3_trainer import GCL_Trainer
from gcl.agents.gcl_agentSB3 import GCL_AgentSB3
from gcl.infrastructure.utils import tic, toc
from gcl.infrastructure.wrapper import FixGoal, LearningReward

ALGO = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "her": HER,
}

WRAPPER = {
    '': None,
    'filter_obs': FilterObservation,
    'flatten_obs': FlattenObservation,
    'fix_goal': [FixGoal, FlattenObservation, FilterObservation],
    'mlp_rew': LearningReward,
}


class IRL_Trainer(object):

    def __init__(self, params):
        #####################
        # SET AGENT PARAMS
        #####################

        train_rew_args = {
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

        train_policy_args = {
            'model_class': params['algo'],
            'additional_algo': params['additional_algo'],

            # A2C
            'normalize_advantage': False,
            # 'policy_lr': params.get("policy_lr", None),

            # PPO
            # 'batch_size': params[batch_size]
            # 'n_steps': ,  # only in a2c or ppo

            # SAC
            # 'train_freq': 10
            # 'gradient_steps': -1

        }

        policy_kwargs = {
            'policy_kwargs': params.get('policy_kwargs', None)
        }

        util_args = {
            "verbose": params['verbose'],
            "seed": params['seed'],
            "create_eval_env": params.get('create_eval_env', False),
            'tensorboard_log': params['runs'],
        }

        agent_params = OrderedDict(
            {
                **train_rew_args,
                **estimate_advantage_args,
                **train_policy_args,
                **policy_kwargs,
                **util_args,
            }
        )
        ic(agent_params)

        self.params = params
        self.params['agent_class'] = GCL_AgentSB3
        self.params['agent_params'] = agent_params

        ################
        # IRL TRAINER
        ################

        self.gcl_trainer = GCL_Trainer(self.params)

    def run_training_loop(self) -> tuple:
        train_log_lst, policy_log_lst = self.gcl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy=self.gcl_trainer.agent.actor,
            eval_policy=None,
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
    # ENV args
    parser.add_argument('--exp_name', '-exp', type=str, default='nav_env_irl')
    parser.add_argument('--env_name', '-env', type=str, default='NavEnv-v0')
    parser.add_argument('--ep_len', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        "-rt", "--rewardType", type=str, default='dense',
        help="Reward type 'sparse' or 'dense' used in non-HER training ",
    )
    parser.add_argument(
        "-wrapper", "--EnvWrapper", type=str, default='',
        help="Apply wrapper to env",
    )
    ##################################################################################
    # Expert Demo args (relative to where you're running this script from)
    parser.add_argument('--expert_policy', '-epf', type=str, default='sac_nav_env')
    parser.add_argument('--expert_data', '-ed', type=str, default='')

    # Policy args
    parser.add_argument("-algo", help="RL Algorithm", type=str, default='ppo')
    parser.add_argument("-add", "--additional_algo", help="RL Algorithm with HER", type=str, default=None)

    # if custom PG setting
    parser.add_argument('--reward_to_go', '-rtg', action='store_true', default=True)
    parser.add_argument('--nn_baseline', action='store_true', default=True)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true', default=False)

    ##################################################################################
    # Train Reward args
    parser.add_argument(
        '--n_iter', '-n', type=int, default=100,
        help='Number of total iterations in outer training loop (Algorithm 1: Guided cost learning)')
    parser.add_argument(
        '--demo_size', type=int, default=200,
        help='Number of expert rollouts to initially add to demo replay buffer'
    )
    parser.add_argument(
        '--num_reward_train_steps_per_iter', type=int, default=10,
        help='Number of reward updates per iteration in Algorithm 2: Nonlinear IOC with stochastic gradients'
    )
    # subset of D_demo
    parser.add_argument(
        '--train_reward_demo_batch_size', type=int, default=100,
        help='Number of expert rollouts (subset of D_demo) to sample from replay buffer per reward update'
    )
    # subset of D_samp
    parser.add_argument(
        '--train_reward_sample_batch_size', type=int, default=100,
        help='Number of policy rollouts (subset of D_samp) to sample from replay buffer per reward update'
    )
    # Sample randomly or only recent data from D_samp
    parser.add_argument(
        '--samp_recent', action='store_true', default=False,
        help='sample random data or recent data from D_samp in train reward'
    )

    ##################################################################################
    # Train Reward Arch
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=20)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    # Utils
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=-1)
    parser.add_argument('--save_params', action='store_true', default=False)
    parser.add_argument("-verb", '--verbose', help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    # CREATE DIRECTORY FOR LOGGING
    ##################################
    path_lst = ['../../data', "../../runs"]
    logdir_lst = []
    for log_path in path_lst:
        if not (os.path.exists(log_path)):
            os.makedirs(log_path)

        logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(log_path, logdir)
        logdir_lst.append(logdir)
        if not (os.path.exists(logdir)):
            os.makedirs(logdir)
        ic(logdir)

    params['logdir'], params['runs'] = logdir_lst

    ###################
    # RUN TRAINING
    ###################
    print("##### PARAM ########")
    # path of pretrain model
    params["no_gpu"] = False  # False
    params["expert_policy"] = "../rl-trained-agents/sac_nav_env"
    # params['algo'] = 'sac'
    params["ep_len"] = 100

    '''Outer Training Loop (Algorithm 1: Guided cost learning)'''
    # Number of iteration of outer training loop (Algorithm 1)
    params['n_iter'] = 101  # converge PPO:20, A2C: 100+
    # Number of expert rollouts to add to demo replay buffer before outer loop
    params['demo_size'] = 200

    ''' Train Reward (Algorithm 2) '''
    # Number of `expert` rollouts to sample from replay buffer per reward update
    # Number of `policy` rollouts to sample from replay buffer per reward update
    params["train_reward_demo_batch_size"] = 100
    params["train_reward_sample_batch_size"] = 100

    ''' Train Policy (PPO, A2C, SAC, SAC+HER) '''
    # Number of transition steps to sample from sample replay buffer per policy update
    params["train_batch_size"] = 10_000  # 10_000

    # size of subset should be less than size of set
    assert params['demo_size'] >= params["train_reward_demo_batch_size"]
    assert params["train_batch_size"] >= params["train_reward_sample_batch_size"] * params["ep_len"]

    params["learning_rate"] = 1e-3
    ic(params)

    ###################
    # Test
    ###################
    trainer = IRL_Trainer(params)
    train_log_lst, policy_log_lst = trainer.run_training_loop()

    ####################
    # Plot
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
        fname1 = "../model/test_sb3_reward_GPU.pth"
        reward_model = trainer.gcl_trainer.agent.reward
        torch.save(reward_model, fname1)

        fname2 = "../model/test_sb3_policy_GPU"
        policy_model = trainer.gcl_trainer.agent.actor
        policy_model.save(fname2)


if __name__ == '__main__':
    print(torch.__version__)
    torch.backends.cudnn.benchmark = True
    main()
    print("Done!")

