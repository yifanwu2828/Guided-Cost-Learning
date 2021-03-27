import os
import time

import gym
import gym_nav

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import A2C, PPO, SAC

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate(model, num_episodes=100, env_id='NavEnv-v0',):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :param env_id
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
            # env.render()
        all_episode_rewards.append(sum(episode_rewards))
    eval_env.close()
    mean_episode_reward = np.mean(all_episode_rewards)
    max_episode_reward = np.max(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    print(f"Mean_reward:{mean_episode_reward:.3f} +/- {std_episode_reward:.3f} in {num_episodes} episodes")
    return mean_episode_reward, std_episode_reward


def main():

    '''
           Recent algorithms (PPO, SAC, TD3) normally require little hyperparameter tuning,
           however, don’t expect the default ones to work on any environment.
           look at the RL zoo (or the original papers) for tuned hyperparameters

           -Continuous Actions - Single Process
               Current State Of The Art (SOTA) algorithms are SAC, TD3 and TQC.
               Please use the hyperparameters in the RL zoo for best results.

           -Continuous Actions - Multiprocessed
               Take a look at PPO, TRPO or A2C.
               Again, don’t forget to take the hyperparameters from the RL zoo
           '''
    # Multiprocess :PPO, SAC,A2C
    # #Remain DDPG, HER, TD3


def vec_env_training():
    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # All actions lie in [-1.0, 1.0]
    # Parallel environments for multiprocessing polocy : PPO SAC, A2C
    env_id = 'NavEnv-v0'
    env = make_vec_env(env_id=env_id, n_envs=4, seed=0)
    # env = gym.make('NavEnv-v0')
    # check_env (env, warn=True, skip_render_check=True)
    print("Observation space:", env.observation_space)
    print("bs_space Shape:", env.observation_space.shape)
    print("Action space:", env.action_space)
    print("Act_space Shape:", env.action_space.shape)

    # A2C lr=1e-3, tts=2e5,normalize_advantage=True
    # PPO lr=5e-3, tts=1e5,
    model = A2C('MlpPolicy', env, learning_rate=1e-3, verbose=1, normalize_advantage=True)
    start_time = time.time()
    # Train the agent and evaluate it
    model.learn(total_timesteps=2e5, log_interval=200)
    print("Finish in {} seconds".format(time.time() - start_time))

    # Create save dir
    save_dir = "./tmp/demo_agent/"
    os.makedirs(save_dir, exist_ok=True)
    zip_name = 'A2C_lr1e-3_tts2e5'
    # The model will be saved under ZIP_NAME.zip
    model.save(save_dir + zip_name)
    del model
    loaded_model = A2C.load(save_dir + zip_name)
    # mean_reward, std_reward = evaluate_policy(loaded_model, model.get_env(), n_eval_episodes=100)
    mean_reward, std_reward = evaluate(loaded_model, num_episodes=200, env_id='NavEnv-v0')


def dummyVecEnv_training():
    env_id = 'NavEnv-v0'
    env = gym.make(env_id)
    check_env (env, warn=True, skip_render_check=True)
    # env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    print("Observation space:", env.observation_space, "\tObs_space Shape:", env.observation_space.shape)
    print("Action space:", env.action_space, "\tAct_space Shape:", env.action_space.shape)

    model = SAC('MlpPolicy', env, verbose=1,)
    start_time = time.time()
    # SAC lr = 5e-4 , tts=1e4
    model.learn(total_timesteps=1e4, log_interval=200)
    print("Finish in {} seconds".format(time.time()-start_time))

    save_dir = "./tmp/demo_agent/"
    os.makedirs(save_dir, exist_ok=True)
    zip_name = 'SAC_lr3e-4_tts1e4'
    model.save(save_dir + zip_name)
    del model
    loaded_model = SAC.load(save_dir + zip_name)
    mean_reward, std_reward = evaluate(loaded_model, num_episodes=200, env_id = 'NavEnv-v0',)


if __name__ == '__main__':

    save_dir = "./tmp/demo_agent/"
    print("%%%%%%%%%%%% A2C %%%%%%%%%%%%")
    loaded_model_A2C = A2C.load(save_dir + "A2C_lr1e-3_tts2e5_normAdv")
    mean_reward, std_reward = evaluate(loaded_model_A2C, num_episodes=1000, env_id='NavEnv-v0', )

    print("%%%%%%%%%%%% PPO %%%%%%%%%%%%")
    loaded_model_PPO = PPO.load(save_dir + "PPO_lr5e-3_tts1e5")
    mean_reward, std_reward = evaluate(loaded_model_PPO, num_episodes=1000, env_id='NavEnv-v0', )

    print("%%%%%%%%%%%% SAC %%%%%%%%%%%%")
    loaded_model_SAC = SAC.load(save_dir + "SAC_lr5e-4_tts1e4")
    mean_reward, std_reward = evaluate(loaded_model_SAC, num_episodes=1000, env_id='NavEnv-v0', )








