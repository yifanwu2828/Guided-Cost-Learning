import time
import gym 
import gym_nav 
from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.ppo import MlpPolicy
#from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

def main():
    # Parallel environments
    env = make_vec_env('NavEnv-v0', n_envs=16)
    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=1e6) 

    # Save model
    model.save("ppo_nav_env")
        
    env = gym.make('NavEnv-v0')
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

#def main():
#    env = gym.make('NavEnv-v0')
#    model = SAC(MlpPolicy, env, verbose=1)
#    model.learn(total_timesteps=10000, log_interval=10)#

#    obs = env.reset()
#    while True:
#        action, _states = model.predict(obs, deterministic=True)
#        obs, reward, done, info = env.step(action)
#        env.render()
#        if done:
#            obs = env.reset()


if __name__ == '__main__':
    main()
