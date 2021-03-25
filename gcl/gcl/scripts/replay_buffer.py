import utils
import numpy as np 


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):
        default_size = 1000000
        if max_size> default_size:
            print(f"Exceed default_size: {default_size}")
        self.max_size = max_size
        # store each rollout
        self.paths = []
        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.log_probs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if self.obs:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths):
        # add new rollouts into our list of rollouts
        self.paths.extend(paths)
        self.paths = self.paths[-self.max_size:]

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, log_probs, next_observations, terminals, concatenated_rews, unconcatenated_rews = utils.convert_listofrollouts(paths)

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.log_probs = log_probs[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.concatenated_rews = concatenated_rews[-self.max_size:]
            self.unconcatenated_rews = unconcatenated_rews[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.log_probs = np.concatenate([self.log_probs, log_probs])[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]
            self.concatenated_rews = np.concatenate(
                [self.concatenated_rews, concatenated_rews]
            )[-self.max_size:]
            if isinstance(unconcatenated_rews, list):
                self.unconcatenated_rews += unconcatenated_rews  # TODO keep only latest max_size around
            else:
                self.unconcatenated_rews.append(unconcatenated_rews)  # TODO keep only latest max_size around

    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return np.array(self.paths)[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return np.array(self.paths)[-num_rollouts:].tolist()

    def sample_all_rollouts(self):
        return np.array(self.paths)

    ########################################
    ########################################
    def sample_random_data(self, batch_size):
        assert (self.obs.shape[0] == self.acs.shape[0]
                == self.concatenated_rews.shape[0]
                == self.next_obs.shape[0] == self.terminals.shape[0])
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        if concat_rew:
            return self.obs[-batch_size:], self.acs[-batch_size:], self.concatenated_rews[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:]
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -=1
                num_recent_rollouts_to_return +=1
                num_datapoints_so_far += utils.get_pathlength(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            observations, actions, log_probs, next_observations, terminals, concatenated_rews, unconcatenated_rews = utils.convert_listofrollouts(rollouts_to_return)
            return observations, actions, unconcatenated_rews, next_observations, terminals

    def flush(self):
        """ Clear Replay Buffer """
        self.paths = []
        self.obs = None
        self.acs = None
        self.log_probs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None
