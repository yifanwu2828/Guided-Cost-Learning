import argparse
import os
import time
from icecream import ic

import numpy as np
import torch

import gym
from gym.wrappers import FilterObservation, FlattenObservation
from stable_baselines3 import HER, SAC, PPO, A2C, TD3, DDPG
from stable_baselines3.common.monitor import Monitor
from gcl.infrastructure.wrapper import FixGoal, LearningReward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-env", help="environment ID", type=str, default="FetchReach-v1")
    parser.add_argument("-f", help="Log folder", type=str, default="../model/")
    parser.add_argument("-algo", help="RL Algorithm", type=str, required=True)
    parser.add_argument("-a", "--add", help="RL Algorithm with HER", type=str, default=None)
    parser.add_argument("-rt", "--rewardType", type=str, default='dense',
                        help="Reward type 'sparse' or 'dense' used in non-HER training ",
                        )
    parser.add_argument("-seed", help="number of timesteps", default=42, type=int)
    parser.add_argument("-train", action='store_true', default=False,
                        help="train new demo if True, or load existed demo if False ",
                        )
    parser.add_argument("-verb", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument("-nr", "--norender", action="store_true", default=False,
                        help="Do not render the environment (useful for tests)"
                        )
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic action")
    args = parser.parse_args()

    ALGO={
            "ppo": PPO,
            "a2c": A2C,
            "sac": SAC,
            "her": HER,
            "td3": TD3,
            "ddpg": DDPG,
        }
    if args.train:
        if args.algo == 'her':
            env = gym.make(args.env)
            env.seed(args.seed)
            env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
            env = FixGoal(env)
            env = Monitor(env)
            model = ALGO['her'](
                'MlpPolicy',
                env,
                ALGO['sac'],
                n_sampled_goal=4,
                goal_selection_strategy='future',
                online_sampling=True,
                learning_rate=0.001,
                verbose=1,
                max_episode_length=None
            )
        start = time.time()
        model.learn(total_timesteps=200_000)
        end = time.time() - start
        ic(end)