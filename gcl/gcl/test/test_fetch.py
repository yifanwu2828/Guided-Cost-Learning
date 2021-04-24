import argparse
import sys
import os
import time

try:
    from icecream import ic
    from icecream import install

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from stable_baselines3 import HER, SAC, PPO, A2C


def get_metrics(reward):
    mean_reward = np.array(reward).mean()
    std_reward = np.array(reward).std()
    return mean_reward, std_reward


def extract_concat(a: dict):
    # assert isinstance(a, dict)
    temp = np.array([], dtype=np.float32)
    for value in a.values():
        temp = np.concatenate((temp, value), axis=None, dtype=np.float32)
    return temp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-env", help="environment ID", type=str, default="FetchReach-v1")
    parser.add_argument("-f", help="Log folder", type=str, default="../model/")
    parser.add_argument("-algo", help="RL Algorithm", default="her", type=str, required=False)
    parser.add_argument("-n", help="number of timesteps", default=300, type=int)
    parser.add_argument("-seed", help="number of timesteps", default=42, type=int)
    parser.add_argument("-train", help="train new demo or load existed demo ", default=False)
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
        "learning_rate": 0.001,
        "max_episode_length": 1200
    }

    ALGO = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "her": HER,

    }
    model_class = SAC  # works also with SAC,DQN, DDPG and TD3

    env = gym.make(args.env)
    env.seed(args.seed)

    # Available strategies (cf paper): future, final, episode
    goal_selection_strategy = params["goal_selection_strategy"]  # equivalent to GoalSelectionStrategy.FUTURE

    # If True the HER transitions will get sampled online
    online_sampling = True  # config["goal_selection_strategy"]
    # Time limit for the episodes
    max_episode_length = params["max_episode_length"]  # 1200

    save_file = "her_FetchReach_v1_env"
    fname = os.path.join(args.f, save_file)

    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    model = HER.load(fname, env=env)

    # #######################################################################################
    # load model
    fname1 = "../model/test_fetch_gcl_reward_GPU.pth"
    reward_model = torch.load(fname1)
    reward_model.eval()

    fname2 = "../model/test_fetch_gcl_policy_GPU.pth"
    policy_model = torch.load(fname2)
    policy_model.eval()

    demo_state = None
    demo_episode_reward = 0.0
    demo_episode_rewards, demo_episode_lengths = [], []
    demo_ep_len = 0
    # For HER, monitor success rate
    demo_successes = []

    agent_state = None
    agent_episode_reward = 0.0
    agent_episode_rewards, agent_episode_lengths = [], []
    # For HER, monitor success rate
    agent_successes = []
    reward_dict = {"act": [], "obs": [], "mlp_reward": [], "true_reward": []}

    obs = env.reset()
    for t in range(args.n):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        demo_successes.append(info.get("is_success"))

        temp = extract_concat(obs)
        reward_dict["mlp_reward"].append(float(reward_model(torch.from_numpy(temp).float(),
                                                            torch.from_numpy(action).float())
                                               .detach().numpy()))
        reward_dict["true_reward"].append(reward)

        if not args.norender:
            env.render("human")
        try:
            time.sleep(env.model.opt.timestep)
        except AttributeError:
            pass

        demo_episode_reward += reward
        demo_ep_len += 1
        if done and info["is_success"] == 1:
            print(f"Episode Reward: {demo_episode_reward:.2f}")
            print("Episode Length", demo_ep_len)
            demo_episode_rewards.append(demo_episode_reward)
            demo_episode_lengths.append(demo_ep_len)

            demo_episode_reward = 0.0
            demo_ep_len = 0
            demo_state = None
            obs = env.reset()

    # env.close()
    # print("Done!!")

    # ic(env.reward_range)

    mlp_reward = np.array(reward_dict["mlp_reward"])
    true_reward = np.array(reward_dict["true_reward"])

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 0))  # (-20, 0)
    scaler.fit(mlp_reward.reshape(-1, 1))
    scaled_reward = scaler.transform(mlp_reward.reshape(-1, 1))

    mean_mlp_reward, std_mlp_reward = get_metrics(mlp_reward)
    mean_true_reward, std_true_reward = get_metrics(true_reward)
    print(f"mean_mlp_reward:{mean_mlp_reward:.4f}, std_mlp_reward:{std_mlp_reward:.4f}")
    print(f"mean_true_reward:{mean_true_reward:.4f}, std_true_reward:{std_true_reward:.4f}")

    fig, ax2 = plt.subplots(3)
    ax2[0].scatter(range(mlp_reward.size), scaled_reward, label="mlp_reward")
    ax2[0].scatter(range(true_reward.size), true_reward, label="true_reward")
    ax2[0].legend()
    ax2[1].scatter(range(mlp_reward.size), mlp_reward, label="mlp_reward")
    ax2[2].scatter(range(true_reward.size), true_reward, label="true_reward", color='#FF7433')
    ax2[1].legend()
    ax2[2].legend()
    plt.show(block=True)

    try:
        obs = env.reset()
        for t in range(args.n):
            agent_obs = extract_concat(obs)
            agent_action, log_prob = policy_model.get_action(agent_obs)
            ic(agent_action)
            ic(agent_action.shape)
            agent_action = agent_action.reshape(-1)
            # import ipdb; ipdb.set_trace()
            env.render("human")
            obs, reward, done, info = env.step(agent_action)

            if done and info["is_success"] == 1:
                obs = env.reset()
    except KeyboardInterrupt:
        pass
    env.close()