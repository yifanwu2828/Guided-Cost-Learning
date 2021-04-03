import time
import gym
import gym_nav

import numpy as np

from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


def evaluate(model, num_episodes=100, env_id='NavEnv-v0', render=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :param env_id: env name
    :param render: whether to render
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    eval_env = gym.make(env_id)
    all_episode_rewards = []
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        obs = eval_env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = eval_env.step(action)
            episode_rewards.append(reward)
            if render:
                eval_env.render()
        all_episode_rewards.append(sum(episode_rewards))
    eval_env.close()
    mean_episode_reward = np.mean(all_episode_rewards)
    max_episode_reward = np.max(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    print(f"Mean_reward:{mean_episode_reward:.3f} +/- {std_episode_reward:.3f} in {num_episodes} episodes")
    return mean_episode_reward, std_episode_reward


def A2C_demo():
    print("################# Collecting A2C Demo #################")
    # Parallel environments
    # env = make_vec_env('NavEnv-v0', n_envs=6)
    # model = A2C(MlpPolicy, env, verbose=1,learning_rate=3e-4)
    # start_time = time.time()
    # model.learn(total_timesteps=2e5)
    # print("Finsih in {}".format(time.time() - start_time))

    # Save model
    # model.save("ppo_nav_env")
    model = A2C.load("../model/a2c_nav_env")
    evaluate(model, num_episodes=200, env_id='NavEnv-v0')

    env = gym.make('NavEnv-v0')
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()


def PPO_demo():
    print("################# Collecting PPO Demo #################")
    # Parallel environments
    env = make_vec_env('NavEnv-v0', n_envs=6)
    model = PPO(MlpPolicy, env, verbose=1, learning_rate=3e-3)
    start_time = time.time()
    model.learn(total_timesteps=516_096)
    print(f"Finish in {(time.time() - start_time)}")

    # Save model
    model.save("../model/ppo_nav_env")
    del model

    # Load model
    model = PPO.load("../model/ppo_nav_env")
    evaluate(model, num_episodes=200, env_id='NavEnv-v0')

    env = gym.make('NavEnv-v0')
    obs = env.reset()
    t=-1
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.1)
        if done:
            obs = env.reset()
            print(f"itr:{i}, step:{int(i - t)} -> done :{done}")
            t = i
    env.close()


def SAC_demo():
    print("################# Collecting SAC Demo #################")
    env = gym.make('NavEnv-v0')
    # wrap Monitor to env to visualize ep_len_mean and ep_rew_mean
    env = Monitor(env)
    # model = SAC('MlpPolicy', env, verbose=1,)
    # model.learn(total_timesteps=2e4, log_interval=10)
    # model.save("sac_nav_env")
    model = SAC.load("../model/sac_nav_env")
    evaluate(model, num_episodes=200, env_id='NavEnv-v0')


if __name__ == '__main__':
    PPO_demo()
