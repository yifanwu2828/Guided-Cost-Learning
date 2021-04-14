import argparse
import sys
import os
import time

import gym
from stable_baselines3 import HER, SAC, PPO, A2C
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-env", help="environment ID", type=str, default="FetchReach-v1")
    parser.add_argument("-f",  help="Log folder", type=str, default="../model/")
    parser.add_argument("-algo", help="RL Algorithm", default="her", type=str, required=False)
    parser.add_argument("-n",  help="number of timesteps", default=200, type=int)
    parser.add_argument("-seed",  help="number of timesteps", default=42, type=int)
    parser.add_argument("-train",  help="train new demo or load existed demo ", default=False)
    parser.add_argument("-verb", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "-nr", "--no-render", action="store_true", default=False,
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

    ALGO={
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "her": HER,

    }

    model_class = SAC   # works also with SAC,DQN, DDPG and TD3

    env=gym.make(args.env)
    env.seed(args.seed)

    # Available strategies (cf paper): future, final, episode
    goal_selection_strategy = params["goal_selection_strategy"]  # equivalent to GoalSelectionStrategy.FUTURE

    # If True the HER transitions will get sampled online
    online_sampling = True  # config["goal_selection_strategy"]
    # Time limit for the episodes
    max_episode_length = params["max_episode_length"]  # 1200

    save_file = "her_FetchReach_v1_env"
    fname = os.path.join(args.f, save_file)

    if args.train:
        # Initialize the model
        model = HER(
            'MlpPolicy',
            env,
            model_class,
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy,
            online_sampling=online_sampling,
            learning_rate=0.001,
            verbose=1,
            max_episode_length=max_episode_length)

        # Train the model
        model.learn(total_timesteps=20_000)
        model.save(fname)
        del model

    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    model = HER.load(fname, env=env)

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []

    #
    obs = env.reset()
    for t in range(args.n):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        env.render("human")
        try:
            time.sleep(env.model.opt.timestep)
        except AttributeError:
            pass
        episode_reward += reward
        ep_len += 1
        # TODO: look into how to apply wrappers
        if done or info["is_success"] == 1:
            print(info)
            print(f"Episode Reward: {episode_reward:.2f}")
            print("Episode Length", ep_len)
            episode_rewards.append(episode_reward)
            episode_lengths.append(ep_len)
            episode_reward = 0.0
            ep_len = 0
            state = None
            obs = env.reset()

        # Reset also when the goal is achieved when using HER
        if done and info[0].get("is_success") is not None:
            print("Success?", info[0].get("is_success", False))

            if info[0].get("is_success") is not None:
                successes.append(info[0].get("is_success", False))
                episode_reward, ep_len = 0.0, 0
    env.close()
    print("Done!!")
