import argparse
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import torch
import gym
from stable_baselines3 import PPO, SAC
from tqdm import tqdm
from dask.distributed import Client

from gcl.infrastructure.utils import tic, toc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true', default=False)
    parser.add_argument('--video', '-v', action='store_true', default=False)
    parser.add_argument('--videoPath', '-path', type=str, default='test_env.gif')
    args = parser.parse_args()
    params = vars(args)

    RENDER = params["render"]
    VIDEO = params["video"]
    PATH = params["videoPath"]

    RENDER_RGB = False
    if VIDEO:
        RENDER_RGB = VIDEO
    #######################################################################################
    #######################################################################################
    # Set seed
    SEED = 1
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    #######################################################################################
    # load model
    start_load = tic("############ Load Model ############")
    # demo_model = PPO.load("../model/ppo_nav_env")
    demo_model = SAC.load("../model/sac_nav_env")

    # fname2 = "../model/test_gcl_policy_GPU.pth"
    # policy_model = torch.load(fname2)
    # policy_model.eval()
    #######################################################################################
    # Init ENV
    env = gym.make("NavEnv-v0")
    env.seed(SEED)
    #######################################################################################
    np.set_printoptions(threshold=sys.maxsize)

    # map_res = cv2.resize(map, dsize=(64*10, 64*10), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("reward map", map_res)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    images = []
    obs = env.reset()
    t = -1
    for i in tqdm(range(300)):
        action, _states = demo_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if RENDER:
            img_array = env.render(mode='human')
            if VIDEO:
                img = Image.fromarray(img_array, 'RGB')
                img = img.resize((500, 500))
                images.append(img)
            time.sleep(0.1)

        elif RENDER_RGB and VIDEO:
            img_array = env.render(mode='rgb_array')
            img = Image.fromarray(img_array, 'RGB')
            img = img.resize((500, 500))
            images.append(img)
        if done:
            obs = env.reset()
            print(f"itr:{i}, step:{int(i - t)} -> done :{done}")
            t = i
    env.close()


    if VIDEO:
        imageio.mimsave(PATH, images)
    toc(start_load)
