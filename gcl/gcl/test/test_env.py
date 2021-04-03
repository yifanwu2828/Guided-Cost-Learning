import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import torch
import gym
from stable_baselines3 import PPO
from tqdm import tqdm


from gcl.infrastructure.utils import tic

if __name__ == '__main__':
    #######################################################################################
    #######################################################################################
    # Set seed
    SEED = 1
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    #######################################################################################
    # load model
    start_load = tic("############ Load Model ############")
    demo_model = PPO.load("../model/ppo_nav_env")

    # fname2 = "../model/test_gcl_policy_GPU.pth"
    # policy_model = torch.load(fname2)
    # policy_model.eval()
    #######################################################################################
    # Init ENV
    env = gym.make("NavEnv-v0")
    env.seed(SEED)
    #######################################################################################
    np.set_printoptions(threshold=sys.maxsize)
    # print(env.Z)
    # print(env.Z.shape)

    rew_map = env.reward_map.copy()

    a = rew_map[:,:,0]
    b = np.where(a > 70, 1, 0)
    plt.imshow(b,)
    plt.show()

    plt.imshow(rew_map)
    plt.show()


    # map_res = cv2.resize(map, dsize=(64*10, 64*10), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("reward map", map_res)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()
    visual=True
    if visual:
        images = []

        obs = env.reset()
        t = -1
        for i in tqdm(range(300)):
            action, _states = demo_model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
            img_array = env.render(mode='rgb_array')
            img = Image.fromarray(img_array, 'RGB')
            img = img.resize((500, 500))
            images.append(img)
            time.sleep(0.1)
            if done:
                obs = env.reset()
                print(f"itr:{i}, step:{int(i - t)} -> done :{done}")
                t = i
        env.close()
        PATH = 'test_env.gif'
        SAVE=False
        if SAVE:
            imageio.mimsave(PATH, images)