import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as preprocessing
import gym
import gym_nav
from stable_baselines3 import PPO
from tqdm import tqdm

from utils import tic, toc

if __name__ == '__main__':
    #######################################################################################
    # Set overflow from warning to raise
    np.seterr(all='raise')
    torch.autograd.set_detect_anomaly(True)
    #######################################################################################
    # Set seed
    SEED = 1
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    #######################################################################################
    # load model
    start_load = tic("############ Load Model ############")
    visual_model = PPO.load("tmp/demo_agent/ppo_nav_env")
    model = PPO.load("ppo_nav_env")
    #######################################################################################
    # Init ENV
    env = gym.make('NavEnv-v0')
    env.seed(SEED)
    #######################################################################################
    obs = env.reset()
    n_step = range(10000)
    for t in tqdm(n_step):
        # action, _states = visual_model.predict(obs, deterministic=True)
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            obs = env.reset()
    env.close()