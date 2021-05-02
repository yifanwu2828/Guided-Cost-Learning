import argparse
from typing import List
import sys
import os
import argparse
import time
from icecream import ic

import numpy as np
import matplotlib.pyplot as plt
import mujoco_py
import gym
from gym.wrappers import FilterObservation, FlattenObservation

from stable_baselines3 import HER, SAC, PPO, A2C, TD3, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


class FixGoal(gym.Wrapper):
    def __init__(self, env, pos=(1.3040752, 0.74440193, 0.66095406)):
        super().__init__(env)
        self.env = env
        assert len(pos) == 3
        if not isinstance(pos, np.ndarray):
            pos = np.array(pos, dtype=np.float32)
        self.pos = pos

    def step(self, action):
        observation, _, done, info = self.env.step(action)

        achieved_goal = observation[3:6]
        reward = self.compute_reward(achieved_goal, self.env.goal)

        return observation, reward, done, info

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, goal, info=None):
        d = self.goal_distance(achieved_goal, goal)
        if self.env.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d


    def reset(self):
        obs = self.env.reset()
        self.env.goal[0] = self.pos[0]
        self.env.goal[1] = self.pos[1]
        self.env.goal[2] = self.pos[2]

        # this one do not work
        # self.env.goal = self.pos
        obs[0:3] = self.env.goal.copy()
        return obs


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
    parser.add_argument("-a", "--add", help="RL Algorithm with HER", type=str, default=None)
    parser.add_argument(
        "-rt", "--rewardType", type=str, default='dense',
        help="Reward type 'sparse' or 'dense' used in non-HER training ", )
    parser.add_argument("-n",  help="number of timesteps", default=500, type=int)
    parser.add_argument("-seed",  help="number of timesteps", default=42, type=int)
    parser.add_argument("-train", action='store_true', default=False,
                        help="train new demo if True, or load existed demo if False ",)
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
        "max_episode_length": 1200,
    }

    ALGO={
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "her": HER,
        "td3": TD3,
        "ddpg": DDPG,
    }

    if args.add is not None:
        save_file = args.algo + '_' + args.add + '_' + "FetchReach_v1_env"
    else:
        save_file = args.algo + '_' + "FetchReach_v1_env"
    fname = os.path.join(args.f, save_file)

    if args.train:
        # Initialize the model
        if args.algo == 'her':
            env = gym.make(args.env)
            env.seed(args.seed)
            env.reward_type = 'dense'  # default sparse
            ic(env.reward_type)

            # env = Monitor(env)
            if args.add is not None:
                # model_class = ALGO[args.add]  # works also with SAC,DQN, DDPG and TD3
                model_class = SAC("MlpPolicy", env, verbose=1, device='cpu')
                ic(model_class)
            else:
                model_class = ALGO["sac"]  # works also with SAC,DQN, DDPG and TD3
            # Available strategies (cf paper): future, final, episode
            goal_selection_strategy = params["goal_selection_strategy"]  # equivalent to GoalSelectionStrategy.FUTURE
            # If True the HER transitions will get sampled online
            online_sampling = params["online_sampling"]
            # Time limit for the episodes
            # max_episode_length = params["max_episode_length"]  # 1200
            model = ALGO[args.algo](
                'MlpPolicy',
                env,
                model_class,
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=True,
                learning_rate=0.001,
                verbose=1,
                max_episode_length=None,
                device='cpu',
            )
            # Train the model
            start = time.time()
            total_timesteps = 29600
            ic(total_timesteps)
            model.learn(total_timesteps=total_timesteps)
            end = time.time() - start
            ic(end)

        else:
            # env = make_vec_env(env_id=args.env, n_envs=1, seed=args.seed)
            # env = VecExtractDictObs(env, key_lst=['observation', 'desired_goal'])
            env = gym.make(args.env)
            env.seed(args.seed)
            env.reward_type = args.rewardType  # default sparse
            # env.reward_type = 'dense'  # default sparse
            ic(env.reward_type)
            env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
            env = FixGoal(env)
            env = Monitor(env)
            if args.algo == 'sac':
                total_timesteps = 135_000
                ic(args.algo)
            else:
                total_timesteps = 200_000
            lr= params.get("learning_rate", 3e-4)
            model = ALGO[args.algo]("MlpPolicy", env, learning_rate=lr, verbose=2)
            ic(lr)
            # Train the model
            start = time.time()
            model.learn(total_timesteps=total_timesteps, log_interval=10)
            end = time.time() - start
            ic(end)

        if args.algo == 'her' and args.add is not None:
            model.save(fname)
        else:
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
    demo_log = {
        "acs": [],
        "obs": [],
        # For HER, monitor success rate
        "successes": [],

        "episode_rewards": [],
        "episode_lengths": [],
        "episode_successes": [],

    }
    episode_reward = 0.0
    ep_len = 0

    env = gym.make(args.env)
    env.seed(args.seed)

    if args.algo != 'her':
        env.reward_type = 'dense'
        env = FilterObservation(env, ['observation', 'desired_goal'])
        env = FlattenObservation(env)
        env = FixGoal(env)

    ic(env)
    obs = env.reset()
    for t in range(args.n):

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        ic(info)

        demo_log['obs'].append(obs)
        demo_log['acs'].append(action)
        demo_log['successes'].append(info.get("is_success"))

        if not args.norender:
            env.render("human")
            time.sleep(0.03)
        episode_reward += float(reward)
        ep_len += 1
        # TODO: look into how to apply wrappers
        if done:  # or info["is_success"] == 1:

            # print(f"Episode Reward: {episode_reward:.2f}")
            # print("Episode Length", ep_len)
            demo_log['episode_rewards'].append(episode_reward)
            demo_log['episode_lengths'].append(ep_len)
            episode_reward = 0.0
            ep_len = 0
            state = None
            obs = env.reset()

        # Reset also when the goal is achieved when using HER
        if done:
            print("Success?", info.get("is_success", False))

            if info.get("is_success") is not None:
                demo_log['successes'].append(info.get("is_success", False))
                episode_reward, ep_len = 0.0, 0
    env.close()
    print("Done!!")

    demo_ac0 = []
    demo_ac1 = []
    demo_ac2 = []
    demo_ac3 = []
    for ac in demo_log['acs']:
        demo_ac0.append(ac[0])
        demo_ac1.append(ac[1])
        demo_ac2.append(ac[2])
        demo_ac3.append(ac[3])

    fig, axs = plt.subplots(4)
    fig.suptitle('Action')
    axs[0].plot(demo_ac0, label='demo_pos_ctrl[0]')
    axs[1].plot(demo_ac1, label='demo_pos_ctrl[1]')
    axs[2].plot(demo_ac2, label='demo_pos_ctrl[3]')
    axs[3].plot(demo_ac3, label='demo_gripper_ctrl')
    for i in range(len(axs)):
        axs[i].legend()
    plt.show(block=True)


    '''def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)'''


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
    # ic(type(env.observation_space))
    # ic(env.observation_space)

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
