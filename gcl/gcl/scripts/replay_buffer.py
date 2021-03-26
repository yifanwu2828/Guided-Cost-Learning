import utils
import numpy as np 
from typing import List


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):
        default_size = 1000000
        if max_size > default_size:
            print("###########################")
            print(f"Exceed default_size: {default_size}")
        self.max_size = max_size
        # store each rollout
        self.paths = []
        self.num_paths = 0
        self.num_data = 0
        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.log_probs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        return len(self.paths)

    def add_rollouts(self, paths):
        """ Add new rollouts into our list of rollouts """
        self.paths.extend(paths)
        # if size exceed buffer's max size, drop old rollouts and keep new on instead
        self.paths = self.paths[-self.max_size:]
        self.num_paths =len(self.paths)

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
        self.num_data = self.obs.shape[0]
    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts) -> np.ndarray:
        """ Randomly select Rollouts """
        assert len(self.paths) >= num_rollouts, "Rollouts in Buffer is less than rollouts acquired "
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return np.array(self.paths)[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1)->List:
        """ Select Recent Rollouts """
        assert len(self.paths)>=num_rollouts, "Rollouts in Buffer is less than rollouts acquired "
        # return np.array(self.paths)[-num_rollouts:]
        return np.array(self.paths)[-num_rollouts:].tolist()

    def sample_all_rollouts(self) -> np.ndarray:
        """ Return All Rollouts from Buffer """
        assert len(self.paths)!= 0, "No rollouts in Buffer"
        return np.array(self.paths)

    ########################################
    ########################################
    def sample_random_data(self, batch_size):
        """
        Sample random transition steps of size batch_size
        """
        assert (self.obs.shape[0] == self.acs.shape[0]
                == self.concatenated_rews.shape[0]
                == self.next_obs.shape[0] == self.terminals.shape[0])
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return (self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices],
                self.next_obs[rand_indices], self.terminals[rand_indices])

    def sample_recent_data(self, batch_size=1, concat_rew=True):
        """
        Sample recent transition steps of size batch_size
        """
        assert len(self.paths) != 0, "No Data in Buffer"
        assert self.num_data >= batch_size, "Data in Buffer is less than data acquired "
        if concat_rew:
            return (self.obs[-batch_size:], self.acs[-batch_size:],
                    self.concatenated_rews[-batch_size:], self.next_obs[-batch_size:],
                    self.terminals[-batch_size:])
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
            (observations, actions, log_probs, next_observations,
             terminals, concatenated_rews, unconcatenated_rews) = utils.convert_listofrollouts(rollouts_to_return)
            return observations, actions, unconcatenated_rews, next_observations, terminals

    def flush(self):
        """ Reset Replay Buffer """
        self.paths = []
        self.obs = None
        self.acs = None
        self.log_probs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None
        self.num_paths = 0
        self.num_data = 0
