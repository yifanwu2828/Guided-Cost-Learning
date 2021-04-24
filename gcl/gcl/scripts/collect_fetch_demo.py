import argparse
from typing import List
import sys
import os
import time
try:
    from icecream import ic
    from icecream import install
    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

import numpy as np
import gym
from gym.wrappers import FilterObservation, FlattenObservation

from stable_baselines3 import HER, SAC, PPO, A2C, TD3, DDPG
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper




class VecExtractDictObs(VecEnvWrapper):
    """
    A vectorized wrapper for extracting dictionary observations.

    :param venv: The vectorized environment
    :param key_lst: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv, key_lst: List[str]):
        self.key_lst = key_lst
        # self.reward_type = 'sparse'
        self.reward_type = 'dense'
        self.distance_threshold = 0.05
        super().__init__(venv=venv, observation_space=gym.spaces.Box(float('-inf'), float('inf'), (13, )))

    def reset(self) -> np.ndarray:
        obsDict = self.venv.reset()
        obs = np.concatenate([v for k, v in obsDict.items() if k in self.key_lst], axis=None, dtype=np.float32)
        ic(obs.shape)
        return obs


    def step_wait(self) -> VecEnvStepReturn:
        obsDict, reward, done, info = self.venv.step_wait()
        obs = np.concatenate([v for k, v in obsDict.items() if k in self.key_lst], axis=None, dtype=np.float32)

        reward = self.compute_reward(obsDict['achieved_goal'], obsDict['desired_goal'])
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, goal, info=None):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-env", help="environment ID", type=str, default="FetchReach-v1")
    parser.add_argument("-f",  help="Log folder", type=str, default="../model/")
    parser.add_argument("-algo", help="RL Algorithm", type=str, required=True)
    parser.add_argument("-n",  help="number of timesteps", default=200, type=int)
    parser.add_argument("-seed",  help="number of timesteps", default=42, type=int)
    parser.add_argument("-train",  help="train new demo or load existed demo ", action='store_true', default=False)
    parser.add_argument("-verb", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "-nr", "--norender", action="store_true", default=False,
        help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic action")
    args = parser.parse_args()

    params = {
        "env_id": "FetchReach-v1",
        "seed": 42,
        "model_class": SAC,
        "goal_selection_strategy": 'future',
        "online_sampling": True,
        "learning_rate": 0.001,
        "max_episode_length": 1200
    }

    ALGO={
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "her": HER,
        "td3": TD3,
        "ddpg": DDPG,
    }



    save_file = args.algo + '_' + "FetchReach_v1_env"
    fname = os.path.join(args.f, save_file)

    if args.train:
        # Initialize the model
        if args.algo == 'her':
            env = gym.make(args.env)
            env.seed(args.seed)
            # env.reward_type = 'dense' # default sparse
            env = Monitor(env)
            model_class = ALGO["sac"]  # works also with SAC,DQN, DDPG and TD3
            # Available strategies (cf paper): future, final, episode
            goal_selection_strategy = params["goal_selection_strategy"]  # equivalent to GoalSelectionStrategy.FUTURE
            # If True the HER transitions will get sampled online
            online_sampling = params["online_sampling"]
            # Time limit for the episodes
            max_episode_length = params["max_episode_length"]  # 1200
            model = ALGO[args.algo](
                'MlpPolicy',
                env,
                model_class,
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=online_sampling,
                learning_rate=0.001,
                verbose=1,
                max_episode_length=max_episode_length)
            # Train the model
            model.learn(total_timesteps=20_000)
        else:
            # env = make_vec_env(env_id=args.env, n_envs=1, seed=args.seed)
            # env = VecExtractDictObs(env, key_lst=['observation', 'desired_goal'])
            env = gym.make(args.env)
            env.seed(args.seed)
            # env.reward_type = 'dense' # default sparse
            env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
            env = Monitor(env)
            model= ALGO[args.algo]("MlpPolicy", env, learning_rate=3e-4, verbose=1)

            # Train the model
            model.learn(total_timesteps=200_000)
        model.save(fname)
        del model

    ic(fname)
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    if args.algo == 'her':
        env = gym.make(args.env)
        env.seed(args.seed)
        model = ALGO[args.algo].load(fname, env=env)
    else:
        model = ALGO[args.algo].load(fname)

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []

    env = make_vec_env(env_id=args.env, n_envs=1, seed=args.seed)
    obs = env.reset()
    for t in range(args.n):

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if not args.norender:
            env.render("human")
        try:
            time.sleep(env.model.opt.timestep)
        except AttributeError:
            time.sleep(0.03)
            pass
        episode_reward += float(reward)
        ep_len += 1
        # TODO: look into how to apply wrappers
        if done or info[0]["is_success"] == 1:
            print(info)
            print(f"Episode Reward: {episode_reward:.2f}")
            print("Episode Length", ep_len)
            episode_rewards.append(episode_reward)
            episode_lengths.append(ep_len)
            episode_reward = 0.0
            ep_len = 0
            state = None
            obs = env.reset()

        # Reset also when the goal is achieved when using HER
        if done and info[0].get("is_success") == 1:
            print("Success?", info[0].get("is_success", False))

            if info[0].get("is_success") is not None:
                successes.append(info[0].get("is_success", False))
                episode_reward, ep_len = 0.0, 0
    env.close()
    print("Done!!")

    # Investigate env config
    '''
    obs: 
    # gym.spaces.dict.Dict
        Dict(
        achieved_goal: Box(-inf, inf, (3,), float32), shape: 3
        desired_goal: Box(-inf, inf, (3,), float32), shape: 3
        observation: Box(-inf, inf, (10,), float32), shape: 10
    ) 
    '''
    ic(type(env.observation_space))
    ic(env.observation_space)
    ic(env.observation_space.shape)

    '''
    acs:
        Box(-1.0, 1.0, (4,), float32), shape: 4
    '''
    # ic(env.action_space)
    # ic(type(env.action_space))
    # ic(env.action_space.shape)
    # print("\n")

    # ic(obs)
    # ic(reward)
    # ic(done)
    # ic(info)

    # extract all value in Dict and concatenate it into an array 10+3+3
    # input_size = ob_dim = 16
    # output_size = ac_dim = 4

    # action_bound = env.action_space.high
    # ic(action_bound)
    # ic(env._max_episode_steps)
    # ic(obs['achieved_goal'].shape) #1-D array

    # Reward can be either -1 or 0.
    # The episode terminates when: Episode length is greater than 50
