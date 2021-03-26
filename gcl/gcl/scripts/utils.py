import numpy as np
import time
import torch
import gym
import gym_nav

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
def toc(t_start, name="Operation"):
    """ Timing Function """
    print(f'\n############ {name} took: {(time.time() - t_start):.4f} sec. ############\n')


########################################################################################

def sample_trajectory(env, policy, agent, max_path_length, render=False, render_mode=('rgb_array'), expert=False):
    """
    Sample one trajectory
    :param env: simulation environment
    :param policy: current policy or expert policy
    :param agent:
    :param max_path_length: max_path_length should equal to env.max_steps
    :param render: visualize trajectory if render is True
    :param render_mode: 'human' or 'rgb_array'
    :param expert: sample from expert policy if True
    """
    # initialize env for the beginning of a new rollout
    ob = env.reset()

    # init vars
    obs, acs, log_probs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], [], []
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
                # time.sleep(0.1)

        # use the most recent ob to decide what to do
        obs.append(ob)
        if expert:
            # stable_baselines3 implementation may need to change this
            # --- check this in every env
            ac, _ = policy.predict(ob, deterministic=True)

            # expert demonstrations assume log_prob = 0
            log_prob = 0
        else:
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

        if expert:  # should expert using true reward?
            rewards.append(rew)
        else:
            # not running on gpu which is slow
            rewards.append(agent.reward.forward(torch.from_numpy(ob).float(),
                                                torch.from_numpy(ac).float()).detach().numpy())

        # end the rollout if (rollout can end due to done, or due to max_path_length)
        rollout_done = 0
        if done or steps >= max_path_length:  # max_path_length == env.max_steps
            rollout_done = 1  # HINT: this is either 0 or 1
            if render:
                print(steps)
                print(env.pos)
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, log_probs, rewards, next_obs, terminals)


########################################################################################

def sample_trajectories(env, policy, agent,
                        min_timesteps_per_batch, max_path_length,
                        render=False, render_mode=('rgb_array'),
                        expert=False):
    """
    Sample rollouts until we have collected batch_size trajectories
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(
            env,
            policy,
            agent,
            max_path_length,
            render=render,
            render_mode=render_mode,
            expert=expert
        )
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
    return paths, timesteps_this_batch


########################################################################################

def sample_n_trajectories(env, policy, agent, ntrajs, max_path_length,
                          render=False, render_mode=('rgb_array'),
                          expert=False):
    """
    Collect ntraj rollouts.
        use sample_trajectory to get each path (i.e. rollout) that goes into paths
        collect n trajectories for video recording
    """
    ntraj_paths = [sample_trajectory(env, policy, agent,
                                     max_path_length,
                                     render=render, render_mode=render_mode,
                                     expert=expert) for _ in range(ntrajs)
                   ]
    return ntraj_paths


############################################
############################################

def Path(obs, image_obs, acs, log_probs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation": np.array(obs, dtype=np.float32),
            "image_obs": np.array(image_obs, dtype=np.uint8),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),
            "log_prob": np.array(log_probs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)
            }


############################################
############################################

def convert_listofrollouts(paths):
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
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, log_probs, next_observations, terminals, concatenated_rewards, unconcatenated_rewards


############################################
############################################

def get_pathlength(path):
    return len(path["reward"])


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean
