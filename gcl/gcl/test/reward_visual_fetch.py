import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn.preprocessing as preprocessing
import gym
from stable_baselines3 import PPO, SAC, A2C, HER
from icecream import ic
from tqdm import tqdm
import time

from gcl.infrastructure.utils import tic, toc


def get_metrics(reward):
    mean_reward = np.array(reward).mean()
    std_reward = np.array(reward).std()
    return mean_reward, std_reward


def extract_concat(obsDict: dict) -> np.ndarray:
    assert isinstance(obsDict, dict)
    obs = np.concatenate([v for k, v in obsDict.items() if k != 'achieved_goal'], axis=None, dtype=np.float32)
    return obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true', default=False)
    parser.add_argument('--plot', '-plt', action='store_true', default=False)
    parser.add_argument('--video', '-v', action='store_true', default=False)
    parser.add_argument('--videoPath', '-path', type=str, default='test_multimovie.gif')
    parser.add_argument('-device', type=str, default='cuda')
    args = parser.parse_args()
    params = vars(args)
    #######################################################################################
    # Set overflow from warning to raise
    np.seterr(all='raise')
    torch.autograd.set_detect_anomaly(True)
    #######################################################################################
    # Set seed
    SEED = 0
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    #######################################################################################
    # Set global Var
    # VERBOSE = False
    # VISUAL = False
    # POLICY = True

    VERBOSE = True
    VISUAL = True
    POLICY = True

    # #######################################################################################
    # Init ENV
    env = gym.make('FetchReach-v1')
    env.seed(SEED)
    env.reward_type = 'dense'
    ic()
     #######################################################################################
    # load model
    start_load = tic("############ Load Model ############")
    # fname1 = "../model/test_gcl_reward_GPU.pth"
    fname1 = f"../model/test_sb3_reward_her_50.pth"
    reward_model = torch.load(fname1)
    reward_model.eval()

    # fname2 = "../model/test_gcl_policy_GPU.pth"
    fname2 = f"../model/test_sb3_policy_her_50"
    policy_model = HER.load(fname2, env)

    demo_model = HER.load("../rl-trained-agents/her_FetchReach_v1_env", env)
    toc(start_load, "Loading")
    #######################################################################################


    #######################################################################################
    if args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    #######################################################################################
    # Init Param
    reward_log_dict2 = {"act": [], "obs": [], "mlp_reward": [], "true_reward": []}
    #######################################################################################
    ''' TEST LEARNING REWARD'''
    if VISUAL:
        obs = env.reset()
        n_step = range(2000)
        for _ in tqdm(n_step):
            action, _states = demo_model.predict(obs, deterministic=True)
            # ac_tensor = torch.from_numpy(action).float().to('cuda:0')

            obs, reward, done, info = env.step(action)
            obs = extract_concat(obs)

            reward_log_dict2["act"].append(action)
            reward_log_dict2["obs"].append(obs)
            reward_log_dict2["mlp_reward"].append(
                float(
                    reward_model(torch.from_numpy(obs).float().to(device),
                                 torch.from_numpy(action).float().to(device)
                                 ).to('cpu').detach().numpy()
                )
            )
            reward_log_dict2["true_reward"].append(reward)
            # env.render()
            if done or info["is_success"] == 1:
                obs = env.reset()
        env.close()

        mlp_reward = np.array(reward_log_dict2["mlp_reward"])
        true_reward = np.array(reward_log_dict2["true_reward"])
        print(env.reward_range)
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,0))  # (-20, 0)
        scaler.fit(mlp_reward.reshape(-1, 1))
        scaled_reward = scaler.transform(mlp_reward.reshape(-1, 1))

        mean_mlp_reward, std_mlp_reward = get_metrics(scaled_reward)
        mean_true_reward, std_true_reward = get_metrics(true_reward)
        print(f"mean_mlp_reward:{mean_mlp_reward:.4f}, std_mlp_reward:{std_mlp_reward:.4f}")
        print(f"mean_true_reward:{mean_true_reward:.4f}, std_true_reward:{std_true_reward:.4f}")
        print(f"MAE: {mean_absolute_error(true_reward, scaled_reward):.5f}")
        print(f"MSE: {mean_squared_error(true_reward, scaled_reward):.5f}")
        print(f"RMS: {mean_squared_error(true_reward, scaled_reward, squared=False):.5f}")
        # print(f"R2: {r2_score(true_reward, scaled_reward):.5f}")

    #######################################################################################
    if VERBOSE:
        fig, ax2 = plt.subplots(3)
        ax2[0].scatter(range(mlp_reward.size), scaled_reward, label="mlp_reward")
        ax2[0].scatter(range(true_reward.size), true_reward, label="true_reward")
        ax2[0].legend()
        ax2[1].scatter(range(mlp_reward.size), mlp_reward, label="mlp_reward")
        ax2[2].scatter(range(true_reward.size), true_reward, label="true_reward", color='#FF7433')
        ax2[1].legend()
        ax2[2].legend()
        plt.show(block=True)

    #######################################################################################
    all_log = {'agent_rews': [], 'agent_done': [], 'agent_eps_return': [], 'agent_total_return': [], 'agent_ac': [],
               'demo_rews': [], 'demo_done': [], 'demo_eps_return': [], 'demo_total_return': [], 'demo_ac': [],
               'agent_mlp_rews': [], 'demo_mlp_rews': [],
               "winner": []
               }
    images = []

    RENDER = params['render']
    PLOT = params['plot']
    VIDEO = params['video']
    PATH = params['videoPath']

    RENDER_RGB = False
    if VIDEO:
        RENDER_RGB = VIDEO

    env = gym.make('FetchReach-v1')
    env.seed(SEED)
    env.reward_type = 'dense'
    agent_obs = env.reset()
    n_step = range(1000)
    for t in tqdm(n_step):
        agent_action, _ = policy_model.predict(agent_obs, deterministic=True)

        all_log['agent_ac'].append(agent_action)

        obs, reward, done, info = env.step(agent_action)
        if RENDER:
            env.render(mode='human')
            time.sleep(0.05)

        elif RENDER_RGB and VIDEO:
            img_array = env.render(mode='rgb_array')
            img = Image.fromarray(img_array, 'RGB')
            img = img.resize((500, 500))
            images.append(img)

        if done or info["is_success"] == 1:
            agent_obs = env.reset()
    env.close()

