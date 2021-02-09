import gym
import gym_nav
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

def custom_env_check(envName):
    """
    check custom_env and try a random agent on custom environment
    :param envName
    :type str
    """
    assert isinstance(envName,str)
    assert len(envName) > 0
    env = gym.make(envName)
    check_env (env, warn=True, skip_render_check=True)
    obs = env.reset()
    n_steps = 1000
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            obs = env.reset()



def main():
    pass

if __name__ == '__main__':
    pass
    # custo,env check
    custom_env_check(envName='NavEnv-v0')
    '''
           Recent algorithms (PPO, SAC, TD3) normally require little hyperparameter tuning,
           however, don’t expect the default ones to work on any environment.
           look at the RL zoo (or the original papers) for tuned hyperparameters

           -Continuous Actions - Single Process
               Current State Of The Art (SOTA) algorithms are SAC, TD3 and TQC.
               Please use the hyperparameters in the RL zoo for best results.

           -Continuous Actions - Multiprocessed
               Take a look at PPO, TRPO or A2C.
               Again, don’t forget to take the hyperparameters from the RL zoo
           '''
    # tried PPO, SAC. Remain A2C, DDPG, HER, TD3
    env = gym.make('CartPole-v1')
    env.seed(0)

    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    # model.save()
    # del model
    # model = A2C.load()

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    obs = env.reset()
    for i in range(1000):
        '''
        As some policy are stochastic by default (e.g. A2C or PPO), you should also try to set deterministic=True
        when calling the .predict() method, this frequently leads to better performance.
        '''
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()