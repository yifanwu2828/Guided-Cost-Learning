import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as preprocessing
import gym
import gym_nav
from stable_baselines3 import PPO
from tqdm import tqdm


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
    np.random.seed(0)
    torch.random.manual_seed(0)
    #######################################################################################
    # Set global Var
    # VERBOSE = False # True
    # VISUAL = False #True
    # POLICY = True

    VERBOSE = True  # True
    VISUAL = True  # True
    POLICY = True
    #######################################################################################
    # load model
    fname1 = "mlp_reward_nitr15_demo50.pth"
    # fname1 = "mlp_reward_nitr30_demo100.pth"
    reward_model = torch.load(fname1)
    reward_model.eval()

    fname2 = "policy_nitr15_demo50.pth"
    policy_model = torch.load(fname2)
    policy_model.eval()

    model = PPO.load("tmp/demo_agent/ppo_nav_env")
    #######################################################################################
    # Init ENV
    env = gym.make('NavEnv-v0')
    env.seed(0)
    #######################################################################################
    # Init Param
    reward_log_dict = {"act": [], "obs": [], "mlp_reward": [], "true_reward": [], }
    #######################################################################################
    '''TEST LEARNING REWARD'''
    obs = env.reset()
    n_step = range(1000)
    for _ in tqdm(n_step):
        action, _states = model.predict(obs, deterministic=True)
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

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1),)
    mlp_reward = min_max_scaler.fit_transform(-mlp_reward.reshape(-1, 1) )
    #######################################################################################
    if VERBOSE:
        mean_mlp_reward, std_mlp_reward = get_metrics(mlp_reward)
        mean_true_reward, std_true_reward = get_metrics(true_reward)
        print(f"mean_mlp_reward:{mean_mlp_reward:.4f}, std_mlp_reward:{std_mlp_reward:.4f}")
        print(f"mean_true_reward:{mean_true_reward:.4f}, std_true_reward:{std_true_reward:.4f}")
        print(f"MSE: {mean_squared_error(true_reward, mlp_reward):.5f}")

        f1, ax = plt.subplots()
        ax.scatter(range(mlp_reward.size), mlp_reward, label="mlp_reward")
        ax.scatter(range(true_reward.size), true_reward, label="true_reward")
        ax.legend()
        plt.show()

        f2, ax = plt.subplots()
        ax.scatter(range(mlp_reward.size), mlp_reward, label="mlp_reward")
        ax.legend()
        plt.show()

        f3, ax = plt.subplots()
        ax.scatter(range(true_reward.size), true_reward, label="true_reward", color='#FF7433')
        ax.legend()
        plt.show()

    #######################################################################################
    ''' Visual Reward'''
    if VISUAL:
        a = np.zeros(2)
        num = env.obs_dim
        x = np.linspace(-env.size, env.size, num=num)
        y = np.linspace(-env.size, env.size, num=num)

        X, Y = np.meshgrid(x, y)
        Z = np.zeros((num, num))

        for i in tqdm(range(num)):
            for j in range(num):
                obs = np.array([X[i, j], Y[i, j]])
                obs = np.concatenate((obs, np.zeros(env.vel_dim)))
                rew = float(reward_model(torch.from_numpy(obs).float(), torch.from_numpy(a).float()).detach().numpy())
                rew = rew*-1
                Z[i, j] = rew
                # env.eval_gaussian()
                # print(Z[i, j].shape)
                # print(Z[i, j])
        reward_min, reward_max = np.min(Z), np.max(Z)
        Z = ((Z - reward_min) / (reward_max - reward_min) * 255).astype(np.uint8)
        reward_map = np.stack((Z, Z, Z), axis=-1)
        plt.imshow(reward_map)
        plt.title("Learning Reward")
        plt.show()

        plt.imshow(env.reward_map)
        plt.title("True Reward")
        plt.show()
    #######################################################################################
    #######################################################################################
    policy_log_dict = {"act": [], "obs": [], "agent_reward": [], "expert_reward": [], }
    #######################################################################################
    ''' TEST Policy'''
    env_agent = gym.make('NavEnv-v0')
    env_agent.seed(0)
    if POLICY:
        obs = env_agent.reset()
        n_step = range(1000)
        for _ in tqdm(n_step):
            action, _states = model.predict(obs, deterministic=True)
            # print(action)
            # print(action.shape)
            action, _ = policy_model.get_action(obs)
            action= action.reshape(-1)
            obs, reward, done, info = env.step(action)
            policy_log_dict["act"].append(action)
            policy_log_dict["obs"].append(obs)
            policy_log_dict["agent_reward"].append(reward)
            env.render()
            if done:
                obs = env.reset()
        env.close()




