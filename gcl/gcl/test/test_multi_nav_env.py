import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import torch
import gym
from stable_baselines3 import PPO, SAC, A2C
from tqdm import tqdm
import time

from gcl.infrastructure.utils import tic, toc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default="MultiNavEnv-v0")
    parser.add_argument('--render', '-r', action='store_true', default=False)
    parser.add_argument('--plot', '-plt', action='store_true', default=False)
    parser.add_argument('--video', '-v', action='store_true', default=False)
    parser.add_argument('--videoPath', '-path', type=str, default='test_multimovie.gif')
    args = parser.parse_args()
    params = vars(args)
    #######################################################################################
    #######################################################################################
    # Set seed
    SEED = 1
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    #######################################################################################
    # # load model
    start_load = tic("############ Load Model ############")
    demo_model = SAC.load("../model/sac_nav_env")

    fname1 = "../model/test_gcl_reward_GPU.pth"
    reward_model = torch.load(fname1)
    reward_model.eval()

    fname2 = "../model/test_gcl_policy_GPU.pth"
    policy_model = torch.load(fname2)
    policy_model.eval()
    #######################################################################################
    # Init ENV
    if params['env'] == "MultiNavEnv-v0":
        env = gym.make("MultiNavEnv-v0")
    else:
        env = gym.make("NavEnv-v0")
    env.seed(SEED)
    #######################################################################################
    np.set_printoptions(threshold=sys.maxsize)
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

    a_min, a_max = [-1, 1]
    demo_obs, agent_obs = env.reset()
    n_step = range(300)
    for t in tqdm(n_step):
        demo_action, _ = demo_model.predict(demo_obs, deterministic=True)
        agent_action, log_prob = policy_model.get_action(agent_obs)
        agent_action = agent_action[0]

        demo_mlp_rew = float(reward_model(torch.from_numpy(demo_obs).float(),
                                          torch.from_numpy(demo_action).float())
                             .detach().numpy()
                             )

        agent_mlp_rew = float(reward_model(torch.from_numpy(agent_obs).float(),
                                           torch.from_numpy(agent_action).float())
                              .detach().numpy()
                              )

        all_log['demo_mlp_rews'].append(demo_mlp_rew)
        all_log['agent_mlp_rews'].append(agent_mlp_rew)




        all_log['demo_ac'].append(np.clip(demo_action, a_min, a_max))
        all_log['agent_ac'].append(np.clip(agent_action, a_min, a_max))

        obs, reward, done, info = env.step(demo_action, agent_action)
        demo_obs, agent_obs = obs
        demo_rew, agent_rew = reward

        demo_done, agent_done = done
        if not demo_done:
            all_log['demo_rews'].append(demo_rew)
            all_log['agent_rews'].append(0)

        if not agent_done:
            all_log['demo_rews'].append(0)
            all_log['agent_rews'].append(agent_rew)

        if RENDER:
            img_array = env.render(mode='human')
            img = Image.fromarray(img_array, 'RGB')
            images.append(img)
            time.sleep(0.1)

        elif RENDER_RGB and VIDEO:
            img_array = env.render(mode='rgb_array')
            img = Image.fromarray(img_array, 'RGB')
            img = img.resize((500, 500))
            images.append(img)

        # Logging
        if demo_done and not agent_done:
            # print("demo_done")
            all_log['winner'].append(0)
        if agent_done and not demo_done:
            # print("agent_done")
            all_log['winner'].append(1)

        done = all([demo_done, agent_done])
        if done:
            if PLOT:
                plt.figure()
                plt.plot(np.zeros_like(all_log['demo_rews']))
                plt.plot(all_log['demo_rews'], label='demo_rews')
                plt.plot(all_log['agent_rews'], label='agent_rews')
                plt.legend()
                plt.show()

            all_log['agent_eps_return'].append(sum(all_log['agent_rews']))
            all_log['demo_eps_return'].append(sum(all_log['demo_rews']))

            all_log['agent_total_return'].extend(all_log['agent_rews'])
            all_log['demo_total_return'].extend(all_log['demo_rews'])

            all_log['demo_rews'] = []
            all_log['agent_rews'] = []

            time.sleep(2)
            demo_obs, agent_obs = env.reset()
    env.close()

    # save a Gif
    if VIDEO:
        start_save = tic("##### Saving Gif  #####")
        imageio.mimsave(PATH, images)
        toc(start_save, "Finishing saving GIF")

    plt.figure()
    plt.plot(all_log['winner'])
    plt.title("Winner of each round, 0: demo done first, 1: agent done first")
    plt.show(block=True)

    demo_ac0 = [ac[0] for ac in all_log['demo_ac']]
    demo_ac1 = [ac[1] for ac in all_log['demo_ac']]
    agent_ac0 = [ac[0] for ac in all_log['agent_ac']]
    agent_ac1 = [ac[1] for ac in all_log['agent_ac']]

    fig, axs = plt.subplots(2)
    fig.suptitle('Action')
    axs[0].plot(demo_ac0, label='demo')
    axs[0].plot(agent_ac0, label='agent')
    axs[1].plot(demo_ac1, label='demo')
    axs[1].plot(agent_ac1, label='agent')

    axs[0].legend()
    axs[1].legend()
    plt.show(block=True)


    plt.figure()
    plt.plot(all_log['demo_mlp_rews'], label='demo_mlp_rews')
    plt.plot(all_log['agent_mlp_rews'], label='agent_mlp_rews')
    plt.legend()
    plt.title("Learning Reward")
    plt.show(block=True)

    plt.figure()
    plt.plot(all_log['demo_eps_return'], label='demo_eps_Return')
    plt.plot(all_log['agent_eps_return'], label='agent_eps_Return')
    plt.legend()
    plt.title("Return of each episode")
    plt.show(block=True)

    plt.figure()
    plt.plot(all_log['demo_total_return'], label='demoReturn')
    plt.plot(all_log['agent_total_return'], label='agentReturn')
    plt.title("Total Reward")
    plt.legend()
    plt.show(block=True)

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
