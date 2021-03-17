import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error
import gym
import gym_nav
from stable_baselines3 import PPO
from tqdm import tqdm


def get_metrics(reward):
    mean_reward = np.array(reward).mean()
    std_reward = np.array(reward).std()
    return mean_reward, std_reward


def main():
    pass


if __name__ == '__main__':
    np.random.seed(0)
    torch.random.manual_seed(0)

    fname = "mlp_reward_nitr15_demo50.pth"
    # fname = "mlp_reward_nitr30_demo100.pth"
    reward_model = torch.load(fname)
    reward_model.eval()

    model = PPO.load("tmp/demo_agent/ppo_nav_env")
    # evaluate(model, num_episodes=200, env_id='NavEnv-v0')

    env = gym.make('NavEnv-v0')
    env.seed(0)
    obs = env.reset()

    log_dict = {"act": [], "obs": [], "mlp_reward": [], "true_reward": [], }
    n_step = range(1000)
    for _ in tqdm(n_step):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        log_dict["act"].append(action)
        log_dict["obs"].append(obs)
        log_dict["mlp_reward"].append(float(reward_model(torch.from_numpy(obs).float(),
                                                         torch.from_numpy(action).float())
                                            .detach().numpy()))
        log_dict["true_reward"].append(reward)
        # env.render()
        if done:
            obs = env.reset()
    env.close()

    mlp_reward = np.array(log_dict["mlp_reward"]) + 1  # TODO: original plot becomes similar after adding 1
    true_reward = np.array(log_dict["true_reward"])

    mean_mlp_reward, std_mlp_reward = get_metrics(mlp_reward)
    mean_true_reward, std_true_reward = get_metrics(true_reward)
    print(f"mean_mlp_reward:{mean_mlp_reward:.4f}, std_mlp_reward:{std_mlp_reward:.4f}")
    print(f"mean_true_reward:{mean_true_reward:.4f}, std_true_reward:{std_true_reward:.4f}")
    print(f"MSE: {mean_squared_error(true_reward, mlp_reward):.5f}")

    f, ax = plt.subplots()
    ax.scatter(range(mlp_reward.size), mlp_reward, label="mlp_reward")
    ax.scatter(range(true_reward.size), true_reward, label="true_reward")
    ax.legend()
    plt.show()

    f, ax = plt.subplots()
    ax.scatter(range(mlp_reward.size), mlp_reward - 1, label="mlp_reward")
    ax.scatter(range(true_reward.size), true_reward, label="true_reward")
    ax.legend()
    plt.show()
