from typing_extensions import TypedDict
from typing import Tuple, List, Union
import time
import numpy as np
import torch
import gym
import gym_nav

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
def tic(message=None):
    """ Timing Function """
    if message:
        print(message)
    else:
        print("############ Time Start ############")
    return time.time()


############################################
############################################
def toc(t_start, name="Operation") -> None:
    """ Timing Function """
    print(f'\n############ {name} took: {(time.time() - t_start):.4f} sec. ############\n')


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
            # # our implementation
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
                      render=False, render_mode: str = 'rgb_array',
                      expert=False, evaluate=False
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
    :return: PathDict
    """
    assert isinstance(max_path_length, int)
    assert max_path_length >= 0
    # initialize env for the beginning of a new rollout
    ob = env.reset()

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
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                # TODO: implement this in NAV_ENV
                # time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        obs.append(ob)
        if expert:
            # stable_baselines3 implementation may need to change this
            # --- check this in every env
            ac, _ = policy.predict(ob, deterministic=True)

            # expert demonstrations assume log_prob = 0, convert to np array to keep consistency
            log_prob = np.zeros(1, dtype=np.float32)

        else:  # collected policy
            # query the policy's get_action function
            ac, log_prob = policy.get_action(ob)
            # unpack ac to remove unwanted type and dim --- check this in every env
            ac = ac[0]
        acs.append(ac)
        log_probs.append(log_prob)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)
        # record result of taking that action
        steps += 1
        next_obs.append(ob)

        if expert or evaluate:  # should expert using true reward?
            rewards.append(rew)
        else:
            # not running on gpu which is slow
            rewards.append(agent.reward.forward(torch.from_numpy(ob).float(),
                                                torch.from_numpy(ac).float()).detach().numpy())

        # end the rollout if (rollout can end due to done, or due to max_path_length)
        rollout_done = 0
        if done or steps >= max_path_length:  # Assume max_path_length == env.max_steps
            rollout_done = 1  # HINT: this is either 0 or 1
        terminals.append(rollout_done)

        if rollout_done:
            break
    # In GCL true rewards will not be used
    return Path(obs, image_obs, acs, log_probs, rewards, next_obs, terminals)


########################################################################################

def sample_trajectories(env, policy, agent: BaseAgent,
                        min_timesteps_per_batch: int, max_path_length: int,
                        render=False, render_mode: str = 'rgb_array',
                        expert=False, evaluate=False
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
            evaluate=evaluate
        )
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
    return paths, timesteps_this_batch


########################################################################################

def sample_n_trajectories(env, policy, agent: BaseAgent,
                          ntrajs: int, max_path_length: int,
                          render=False, render_mode: str = 'rgb_array',
                          expert=False, evaluate=False
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
    :return: List[PathDict]
    """
    assert isinstance(ntrajs, int) and isinstance(max_path_length, int)
    assert ntrajs > 0 and max_path_length > 0
    ntraj_paths: List[PathDict] = [sample_trajectory(env, policy, agent,
                                                     max_path_length,
                                                     render=render, render_mode=render_mode,
                                                     expert=expert, evaluate=evaluate
                                                     ) for _ in range(ntrajs)
                                   ]
    return ntraj_paths


############################################
############################################

def Path(obs: List[np.ndarray], image_obs: Union[List[np.ndarray], List],
         acs: List[np.ndarray], log_probs: List[np.ndarray],
         rewards: List[np.ndarray], next_obs: List[np.ndarray],
         terminals: List[int]
         ) -> PathDict:
    """
    Take info (separate arrays) from a single rollout and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation": np.array(obs, dtype=np.float32),
            "image_obs": np.array(image_obs, dtype=np.uint8),
            "action": np.array(acs, dtype=np.float32),
            "log_prob": np.array(log_probs, dtype=np.float32),
            "reward": np.array(rewards, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)
            }


############################################
############################################

def convert_listofrollouts(paths: List[PathDict]
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    log_probs = np.concatenate([path["log_prob"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards: List = [path["reward"] for path in paths]
    return observations, actions, log_probs, next_observations, terminals, concatenated_rewards, unconcatenated_rewards


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
