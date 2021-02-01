import numpy as np
import gym
from gym import wrappers
from pg import Agent


import pybullet_envs

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    seed = 0
    env.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    
    lr = 5e-3
    gamma = 0.99
    nn_baseline =False

    agent_params = {'act_dim':act_dim,
                    'obs_dim':obs_dim,
                    'n_layers':2,
                    'size':64,
                    'discrete':discrete,
                    'learning_rate':lr,
                    'nn_baseline':nn_baseline

                    }

    agent =Agent(env,agent_params)

    score_history=[]
    score = 0
    n_episodes = 1000

    env = wrappers.Monitor(env, 'tmp/CartPole-v0', video_callable=lambda episodeid: True, force=True)

    for i in range(n_episodes):
        
        print('episode:',i, 'score %.3f' %score)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)
            agent.store_rewards(reward)
            observation = observation
            score += reward
        score_history.append(score)
        agent.train()