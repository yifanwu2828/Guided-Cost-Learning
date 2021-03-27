import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as preprocessing
import gym
import gym_nav
from stable_baselines3 import PPO
from tqdm import tqdm
import pickle

from utils import tic, toc


def get_metrics(reward):
    mean_reward = np.array(reward).mean()
    std_reward = np.array(reward).std()
    return mean_reward, std_reward


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
    # Set global Var
    # VERBOSE = False
    # VISUAL = False
    # POLICY = True

    VERBOSE = True
    VISUAL = True
    POLICY = True
    #######################################################################################
    # load model
    start_load = tic("############ Load Model ############")
    fname1 = "test_reward2.pth"
    # fname1 = "mlp_reward_nitr30_demo100.pth"
    reward_model = torch.load(fname1)
    reward_model.eval()

    fname2 = "test_policy2.pth"
    policy_model = torch.load(fname2)
    policy_model.eval()

    visual_model = PPO.load("tmp/demo_agent/ppo_nav_env")
    model = PPO.load("ppo_nav_env")
    toc(start_load, "Loading")
    #######################################################################################
    # Init ENV
    env = gym.make('NavEnv-v0')
    env.seed(SEED)
    #######################################################################################
    # Init Param
    reward_log_dict = {"act": [], "obs": [], "mlp_reward": [], "true_reward": [], }
    #######################################################################################
    '''TEST LEARNING REWARD'''
    if VISUAL:
        obs = env.reset()
        n_step = range(500)
        for _ in tqdm(n_step):
            action, _states = visual_model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            reward_log_dict["act"].append(action)
            reward_log_dict["obs"].append(obs)
            reward_log_dict["mlp_reward"].append(float(reward_model(torch.from_numpy(obs).float(),
                                                                    torch.from_numpy(action).float())
                                                       .detach().numpy()))
            reward_log_dict["true_reward"].append(reward)
            # env.render()
            if done:
                obs = env.reset()
        env.close()

        mlp_reward = np.array(reward_log_dict["mlp_reward"])
        true_reward = np.array(reward_log_dict["true_reward"])

        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 0))
        scaler.fit(mlp_reward.reshape(-1, 1))
        scaled_reward = scaler.transform(mlp_reward.reshape(-1, 1))
        f, ax = plt.subplots()
        ax.scatter(range(mlp_reward.size), scaled_reward, label="mlp_reward")
        ax.scatter(range(true_reward.size), true_reward, label="true_reward")
        ax.legend()
        plt.show()
        mean_mlp_reward, std_mlp_reward = get_metrics(scaled_reward)
        mean_true_reward, std_true_reward = get_metrics(true_reward)
        print(f"mean_mlp_reward:{mean_mlp_reward:.4f}, std_mlp_reward:{std_mlp_reward:.4f}")
        print(f"mean_true_reward:{mean_true_reward:.4f}, std_true_reward:{std_true_reward:.4f}")
        print(f"MSE: {mean_squared_error(true_reward, scaled_reward):.5f}")
    #######################################################################################
    if VERBOSE:
        f1, ax1 = plt.subplots()
        ax1.scatter(range(mlp_reward.size), mlp_reward, label="mlp_reward")
        ax1.scatter(range(true_reward.size), true_reward, label="true_reward")
        ax1.legend()
        plt.show()

        f2, ax2 = plt.subplots()
        ax2.scatter(range(mlp_reward.size), mlp_reward, label="mlp_reward")
        ax2.legend()
        plt.show()

        f3, ax2 = plt.subplots()
        ax2.scatter(range(true_reward.size), true_reward, label="true_reward", color='#FF7433')
        ax2.legend()
        plt.show()

    #######################################################################################
    #######################################################################################
    # Init Param
    reward_log_dict2 = {"act": [], "obs": [], "mlp_reward": [], "true_reward": [], }
    #######################################################################################
    '''TEST LEARNING REWARD'''
    if VISUAL:
        obs = env.reset()
        n_step = range(500)
        for _ in tqdm(n_step):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            reward_log_dict2["act"].append(action)
            reward_log_dict2["obs"].append(obs)
            reward_log_dict2["mlp_reward"].append(float(reward_model(torch.from_numpy(obs).float(),
                                                                     torch.from_numpy(action).float())
                                                        .detach().numpy()))
            reward_log_dict2["true_reward"].append(reward)
            # env.render()
            if done:
                obs = env.reset()
        env.close()

        mlp_reward = np.array(reward_log_dict2["mlp_reward"])
        true_reward = np.array(reward_log_dict2["true_reward"])

        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 0))
        scaler.fit(mlp_reward.reshape(-1, 1))
        scaled_reward = scaler.transform(mlp_reward.reshape(-1, 1))
        f, ax = plt.subplots()
        ax.scatter(range(mlp_reward.size), scaled_reward, label="mlp_reward")
        ax.scatter(range(true_reward.size), true_reward, label="true_reward")
        ax.legend()
        plt.show()
        mean_mlp_reward, std_mlp_reward = get_metrics(scaled_reward)
        mean_true_reward, std_true_reward = get_metrics(true_reward)
        print(f"mean_mlp_reward:{mean_mlp_reward:.4f}, std_mlp_reward:{std_mlp_reward:.4f}")
        print(f"mean_true_reward:{mean_true_reward:.4f}, std_true_reward:{std_true_reward:.4f}")
        print(f"MSE: {mean_squared_error(true_reward, scaled_reward):.5f}")
    #######################################################################################
    if VERBOSE:
        f1, ax1 = plt.subplots()
        ax1.scatter(range(mlp_reward.size), mlp_reward, label="mlp_reward")
        ax1.scatter(range(true_reward.size), true_reward, label="true_reward")
        ax1.legend()
        plt.show()

        f2, ax2 = plt.subplots()
        ax2.scatter(range(mlp_reward.size), mlp_reward, label="mlp_reward")
        ax2.legend()
        plt.show()

        f3, ax2 = plt.subplots()
        ax2.scatter(range(true_reward.size), true_reward, label="true_reward", color='#FF7433')
        ax2.legend()
        plt.show()

    #######################################################################################
    ''' Visual Reward'''
    if VISUAL:
        a = np.zeros(2)
        num = 64
        x = np.linspace(-env.size, env.size, num=num)
        y = np.linspace(-env.size, env.size, num=num)

        X, Y = np.meshgrid(x, y)
        Z = np.zeros((num, num))
        rew_lst = []
        for i in tqdm(range(num)):
            for j in range(num):
                obs = np.array([X[i, j], Y[i, j]])
                obs = np.concatenate((obs, np.zeros(env.vel_dim)))
                rew = reward_model(torch.from_numpy(obs).float(), torch.from_numpy(a).float()).detach().numpy()
                rew_lst.append(rew[0])
                Z[i, j] = rew[0]
                # Z[i, j] = rew
                # env.eval_gaussian()
                # print(Z[i, j].shape)
                # print(Z[i, j])

        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 0))
        scaler.fit(Z)
        Z = (scaler.transform(Z) * 255).astype(np.uint8)

        # reward_min, reward_max = np.min(Z), np.max(Z)
        # print(reward_min, reward_max)
        # Z = ((Z - reward_min) / (reward_max - reward_min) * 255).astype(np.uint8)
        # print(Z)
        reward_map = Z  # np.stack((Z, Z, Z), axis=-1)
        plt.imshow(reward_map, cmap='gray', vmin=0, vmax=255)
        plt.title("Learning Reward")
        plt.show()

        plt.imshow(env.reward_map)
        plt.title("True Reward")
        plt.show()
    #######################################################################################
    #######################################################################################
    policy_log_dict = {"act": [], "obs": [], "agent_reward": [], "expert_reward": [], "ep_len": []}
    policy_expert_dict = {"act": [], "obs": [], "expert_reward": [], "done": [], "info": []}
    #######################################################################################
    ''' TEST Policy'''
    env_agent = gym.make('NavEnv-v0')
    env_agent.seed(0)
    collect = False
    if POLICY:
        t = 0
        obs = env_agent.reset()  # set inital position
        policy_expert_dict["obs"].append(obs)
        n_step = range(1000)
        for i in tqdm(n_step, leave=False):
            action, _states = model.predict(obs, deterministic=True)
            if collect:
                obs, reward, done, _ = env.step(action)
                policy_expert_dict["act"].append(action)
                policy_expert_dict["obs"].append(obs)
                policy_expert_dict["expert_reward"].append(reward)
                policy_expert_dict["done"].append(done)

            action, _ = policy_model.get_action(obs)
            action = action.reshape(-1)

            obs, reward, done, _ = env.step(action)
            policy_log_dict["act"].append(action)
            policy_log_dict["obs"].append(obs)
            policy_log_dict["agent_reward"].append(reward)
            # if i <5:
            #     env.render()
            env.render()
            if done:
                ep_len = int(i - t)
                # policy_expert_dict["info"].append(ep_len)
                policy_log_dict["ep_len"].append(ep_len)
                print(env.pos)
                print(f"itr:{i}, step:{ep_len} -> done :{done}")
                obs = env.reset()
                t = i
        env.close()
