import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as preprocessing
import gym
import gym_nav
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import time
from gym_nav.envs.multi_nav_env import MultiNavEnv
from utils import tic, toc

if __name__ == '__main__':
    #######################################################################################
    #######################################################################################
    # Set seed
    SEED = 1
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    #######################################################################################
    # # load model
    start_load = tic("############ Load Model ############")
    demo_model = PPO.load("ppo_nav_env")

    fname2 = "test_gcl_policy_650.pth"
    policy_model = torch.load(fname2)
    policy_model.eval()
    #######################################################################################
    # Init ENV
    env = gym.make("MultiNavEnv-v0")
    env.seed(SEED)
    #######################################################################################
    demo_obs, agent_obs = env.reset()
    n_step = range(5000)
    for t in tqdm(n_step):
        demo_action, _ = demo_model.predict(demo_obs, deterministic=True)
        agent_action, log_prob = policy_model.get_action(agent_obs)
        agent_action = agent_action[0]

        obs, reward, done, info = env.step(demo_action, agent_action)
        demo_obs, agent_obs = obs
        demo_rew, agent_rew = reward
        demo_done, agent_done = done
        env.render()
        done = all([demo_done, agent_done])
        if done:
            demo_obs, agent_obs = env.reset()
            time.sleep(0.1)
    env.close()

    # env = gym.make("NavEnv-v0")
    # env.seed(SEED)
    #
    # demo_obs = env.reset()
    # n_step = range(500)
    # for t in tqdm(n_step):
    #     demo_action, _states = demo_model.predict(demo_obs, deterministic=True)
    #     print(demo_action)
    #     demo_obs, reward, done, info = env.step(demo_action)
    #     env.render()
    #     if done:
    #         demo_obs = env.reset()
    #         time.sleep(0.05)
    # env.close()