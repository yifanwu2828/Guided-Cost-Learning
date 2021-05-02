from typing_extensions import TypedDict
from typing import Tuple, List, Union, Optional, Dict
import time
import warnings
import random

import numpy as np
import torch
import gym
from gym import spaces
from stable_baselines3 import A2C, SAC, PPO, HER
from icecream import ic
try:
    import gym_nav
except ImportError:
    pass

from gcl.agents.base_agent import BaseAgent


class PathDict(TypedDict):
    observation: np.ndarray
    image_obs: np.ndarray
    action: np.ndarray
    log_prob: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminal: np.ndarray


############################################
############################################
def tic(message: Optional[str] = None) -> float:
    """ Timing Function """
    if message:
        print(message)
    else:
        print("############ Time Start ############")
    return time.time()


############################################
############################################
def toc(t_start: float, name: Optional[str] = "Operation", ftime=False) -> None:
    """ Timing Function """
    assert isinstance(t_start, float)
    sec: float = time.time() - t_start
    if ftime:
        duration = time.strftime("%H:%M:%S", time.gmtime(sec))
        print(f'\n############ {name} took: {str(duration)} ############\n')
    else:
        print(f'\n############ {name} took: {sec:.4f} sec. ############\n')


############################################
############################################

def set_random_seed(seed: int = 42, deterministic: bool = False):
    """
    Seed the different random generators.

    :param seed:
    :param deterministic:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if deterministic:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        # Fix use on algorithm instead of randomly searching best one
        torch.backends.cudnn.benchmark = False
        raise UserWarning("WARNING: Deterministic operations for CuDNN, it may impact performances")


########################################################################################


########################################################################################
def evaluate_model(eval_env_id, model, num_episodes=1000, render=False):
    """
    Evaluate a RL agent in the given ENV
    :param: name of eval_env
    :param: model: (BaseRLModel object) the RL Agent
    :param: num_episodes: (int) number of episodes to evaluate it
    :param: render
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    eval_env = gym.make(eval_env_id)
    all_episode_rewards = []
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        obs = eval_env.reset()
        while not done:
            # stable-baselines3 implementation
            action, _states = model.predict(obs, deterministic=True)

            # our implementation
            # action, log_prob = model.get_action(obs)
            # action = action[0]

            # here, action, rewards and dones are arrays
            obs, reward, done, info = eval_env.step(action)
            episode_rewards.append(reward)
            if render:
                eval_env.render()
        all_episode_rewards.append(sum(episode_rewards))
    eval_env.close()
    mean_episode_reward = np.mean(all_episode_rewards)
    max_episode_reward = np.max(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    print(f"Mean_reward:{mean_episode_reward:.3f} +/- {std_episode_reward:.3f} in {num_episodes} episodes")
    print(f"Max_reward:{max_episode_reward:.3f} in {num_episodes} episodes")
    return mean_episode_reward, std_episode_reward


########################################################################################
########################################################################################
def sample_trajectory(env,
                      policy,
                      agent,
                      max_path_length: int,
                      render: bool = False, render_mode: str = 'rgb_array',
                      expert: bool = False, evaluate: bool = False, device='cpu',
                      deterministic: bool = True, sb3: bool = False,
                      ) -> PathDict:
    """
    Sample a single trajectory and returns infos
    :param env: simulation environment
    :param policy: current policy or expert policy
    :param agent:
    :param max_path_length: max_path_length should equal to env.max_steps
    :param render: visualize trajectory if render is True
    :param render_mode: 'human' or 'rgb_array'
    :param expert: sample from expert policy if True
    :param evaluate:
    :param device: 'cpu' or 'cuda'
    :param deterministic: whether or not to return deterministic actions, else stochastic
    :param sb3: use sb3 model
    :return: PathDict
    """
    assert isinstance(max_path_length, int)
    assert max_path_length >= 0

    # initialize env for the beginning of a new rollout
    ob: Union[Dict, np.ndarray] = env.reset()
    last_states = None

    # init vars
    obs: List[np.ndarray] = []
    acs: List[np.ndarray] = []
    log_probs: List[np.ndarray] = []
    rewards: List[np.ndarray] = []
    next_obs: List[np.ndarray] = []
    terminals: List[int] = []
    image_obs: List[np.ndarray] = []

    steps = 0
    while True:
        # render image of the simulated env
        if render:
            if render_mode == 'rgb_array':
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            elif render_mode == 'human':
                env.render(mode=render_mode)
                if hasattr(env, 'model'):
                    try:
                        time.sleep(env.model.opt.timestep)
                    except AttributeError:
                        warnings.warn(f"No attribute name 'model' found in {env.__str__}", UserWarning, stacklevel=2)
                else:
                    time.sleep(0.1)

        # use the most recent ob to decide what to do
        if isinstance(ob, dict):
            ob = extract_concat(ob)
        obs.append(ob)

        # expert policy
        if expert:
            # stable_baselines3 implementation
            ac, _state = policy.predict(ob, state=last_states, deterministic=deterministic)

            # The last states (can be None, used in recurrent policies)
            last_states = _state

            # expert demonstrations assume prob = 1 -> log_prob = 0, convert to np array to keep consistency
            log_prob = np.zeros(1, dtype=np.float32)

        # collected policy
        else:
            # use sb3 implementation to obtain action and log_probs
            if sb3:
                algo_name = agent.agent_params["model_class"]

                ac, _state = policy.predict(ob, state=last_states, deterministic=deterministic)

                # sb3 take action as tensor and output logprob tensor
                ac_tensor = torch.from_numpy(ac).float().to(device)

                if algo_name == 'ppo' or algo_name == 'a2c':
                    log_prob = policy.policy.action_dist.log_prob(ac_tensor)

                elif algo_name == 'sac' or algo_name == 'her':
                    log_prob = policy.actor.action_dist.log_prob(torch.unsqueeze(ac_tensor, -1))
                    log_prob = log_prob.sum(-1, keepdim=True)
                else:
                    raise NotImplementedError("Policy Algo Not Implemented!")
                log_prob = log_prob.to('cpu').detach().numpy()

            # query the policy's get_action function
            else:
                ac, log_prob = policy.get_action(ob)
                # unpack ac to remove unwanted type and dim
                ac = ac[0]
        # Record actions and log_prob
        acs.append(ac)
        log_probs.append(log_prob)

        # take that action and record results
        ob, rew, done, info = env.step(ac)
        if isinstance(ob, dict):
            ob = extract_concat(ob)

        # record result(obs) of taking that action
        steps += 1
        next_obs.append(ob)

        # Append True Reward(In GCL true rewards will not be used)
        if expert or evaluate:
            rewards.append(rew)
        else:
            # Append MLP Reward
            rewards.append(

                agent.reward(
                    observation=torch.from_numpy(ob).float().to(device),
                    action=torch.from_numpy(ac).float().to(device),
                ).to('cpu').detach().numpy()

            )

        # end the rollout if (rollout can end due to done, or due to max_path_length, or success in GoalEnv)
        rollout_done = 0
        if done or steps >= max_path_length: #or info.get("is_success", 0.0) == 1:
            rollout_done = 1  # HINT: this is either 0 or 1
        terminals.append(rollout_done)

        # End while loop
        if rollout_done:
            break

    return Path(obs, image_obs, acs, log_probs, rewards, next_obs, terminals)


########################################################################################

def sample_trajectories(
        env,
        policy,
        agent: BaseAgent,
        min_timesteps_per_batch: int,
        max_path_length: int,
        render=False,
        render_mode: str = 'rgb_array',
        expert=False,
        evaluate=False,
        device='cpu',
        deterministic: bool = True,
        sb3: bool = False,
) -> Tuple[List[PathDict], int]:
    """
    Sample rollouts until we have collected batch_size trajectories
    :param env: simulation environment
    :param policy: current policy or expert policy
    :param agent:
    :param min_timesteps_per_batch:
    :param max_path_length: max_path_length should equal to env.max_steps
    :param render: visualize trajectory if render is True
    :param render_mode: 'human' or 'rgb_array'
    :param expert: sample from expert policy if True
    :param evaluate
    :param device: 'cpu' or 'cuda'
    :param deterministic: whether or not to return deterministic actions, else stochastic
    :param sb3: use sb3 model
    :return: List[PathDict], timesteps_this_batch
    """
    assert isinstance(min_timesteps_per_batch, int) and isinstance(max_path_length, int)
    assert min_timesteps_per_batch > 0 and max_path_length > 0

    timesteps_this_batch: int = 0
    paths: List[PathDict] = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path: PathDict = sample_trajectory(
            env,
            policy,
            agent,
            max_path_length=max_path_length,
            render=render,
            render_mode=render_mode,
            expert=expert,
            evaluate=evaluate,
            device=device,
            deterministic=deterministic,
            sb3=sb3,
        )
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
    return paths, timesteps_this_batch


########################################################################################

def sample_n_trajectories(
        env,
        policy,
        agent: BaseAgent,
        ntrajs: int,
        max_path_length: int,
        render=False,
        render_mode: str = 'rgb_array',
        expert=False,
        evaluate=False,
        device='cpu',
        deterministic: bool = True,
        sb3: bool = False
) -> List[PathDict]:
    """
    :param env: simulation environment
    :param policy: current policy or expert policy
    :param agent:
    :param ntrajs: number of trajectories need to collect
    :param max_path_length: max_path_length should equal to env.max_steps
    :param render: visualize trajectory if render is True
    :param render_mode: 'human' or 'rgb_array'
    :param expert: sample from expert policy if True
    :param evaluate
    :param device: 'cpu' or 'cuda'
    :param deterministic: whether or not to return deterministic actions, else stochastic
    :param sb3: use sb3 model
    :return: List[PathDict]
    """
    assert isinstance(ntrajs, int) and isinstance(max_path_length, int)
    assert ntrajs > 0 and max_path_length > 0
    ntraj_paths: List[PathDict] = [
        sample_trajectory(
            env,
            policy, agent,
            max_path_length,
            render=render, render_mode=render_mode,
            expert=expert, evaluate=evaluate,
            device=device,
            deterministic=deterministic,
            sb3=sb3,
        ) for _ in range(ntrajs)
    ]
    return ntraj_paths


############################################
############################################

def Path(
        obs: List[np.ndarray],
        image_obs: Union[List[np.ndarray], List],
        acs: List[np.ndarray],
        log_probs: List[np.ndarray],
        rewards: List[np.ndarray],
        next_obs: List[np.ndarray],
        terminals: List[int]
) -> PathDict:
    """
    Take info (separate arrays) from a single rollout and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "action": np.array(acs, dtype=np.float32),
        "log_prob": np.array(log_probs, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32)
    }


############################################
############################################

def convert_listofrollouts(
        paths: List[PathDict]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations, actions, log_probs = [], [], []
    next_observations, terminals, unconcatenated_rewards = [], [], []
    for path in paths:
        observations.append(path["observation"])
        actions.append(path["action"])
        log_probs.append(path["log_prob"])
        next_observations.append(path["next_observation"])
        terminals.append(path["terminal"])
        unconcatenated_rewards.append(path["reward"])

    observations = np.concatenate(observations, dtype=np.float32)
    actions = np.concatenate(actions, dtype=np.float32)
    log_probs = np.concatenate(log_probs, dtype=np.float32)
    next_observations = np.concatenate(next_observations, dtype=np.float32)
    terminals = np.concatenate(terminals, dtype=np.float32)
    concatenated_rewards = np.concatenate(unconcatenated_rewards, dtype=np.float32)

    return (observations, actions, log_probs,
            next_observations, terminals,
            concatenated_rewards, unconcatenated_rewards)


############################################
############################################

def get_pathlength(path: PathDict) -> int:
    """get number of steps in path"""
    return len(path["reward"])


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def mean_squared_error(a, b):
    return np.mean((a - b) ** 2)


def extract_concat(obsDict: dict) -> np.ndarray:
    assert isinstance(obsDict, dict)
    obs = np.concatenate([v for k, v in obsDict.items() if k != 'achieved_goal'], axis=None, dtype=np.float32)
    return obs


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def get_obs_shape(observation_space: spaces.Space) -> Tuple[int, ...]:
    """
    Get the shape of the observation (useful for the buffers).
    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
