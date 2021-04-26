from typing import List, Tuple, Union

import numpy as np

import gcl.infrastructure.utils as utils
from gcl.infrastructure.utils import PathDict


class ReplayBuffer(object):
    """ Buffer to store environment transitions """
    def __init__(self, max_size=1_000_000):
        assert isinstance(max_size, int)
        self._max_size: int = max_size

        # store each rollout
        self._paths: List[PathDict] = []
        self._num_paths: int = 0
        self._num_data: int = 0

        # record size and length of new add paths
        self._new_path_len: int = 0
        self._new_data_len: int = 0

        # store (concatenated) component arrays from each rollout
        self._obs = None
        self._acs = None
        self._log_probs = None
        self._concatenated_rews = None
        self._unconcatenated_rews = None
        self._next_obs = None
        self._terminals = None

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def paths(self) -> List[PathDict]:
        return self._paths

    @paths.setter
    def paths(self, value):
        assert isinstance(value, np.ndarray) or isinstance(value, list)
        self._paths = value

    @property
    def num_paths(self) -> int:
        return self._num_paths

    @num_paths.setter
    def num_paths(self, new_len):
        assert isinstance(new_len, int)
        self._num_paths = new_len

    @property
    def num_data(self) -> int:
        return self._num_data

    @num_data.setter
    def num_data(self, new_size):
        assert isinstance(new_size, int)
        self._num_data = new_size

    ##################################
    # New Paths
    @property
    def new_path_len(self) -> int:
        return self._new_path_len

    @new_path_len.setter
    def new_path_len(self, value):
        assert isinstance(value, int)
        self._new_path_len = value

    @property
    def new_data_len(self) -> int:
        return self._new_data_len

    @new_data_len.setter
    def new_data_len(self, value):
        assert isinstance(value, int)
        self._new_data_len = value

    ##################################
    # Path info
    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, value):
        assert isinstance(value, np.ndarray) or isinstance(value, list)
        self._obs = value

    @property
    def acs(self):
        return self._acs

    @acs.setter
    def acs(self, value):
        assert isinstance(value, np.ndarray) or isinstance(value, list)
        self._acs = value

    @property
    def log_probs(self):
        return self._log_probs

    @log_probs.setter
    def log_probs(self, value):
        assert isinstance(value, np.ndarray) or isinstance(value, list)
        self._log_probs = value

    @property
    def concatenated_rews(self):
        return self._concatenated_rews

    @concatenated_rews.setter
    def concatenated_rews(self, value):
        assert isinstance(value, np.ndarray) or isinstance(value, list)
        self._concatenated_rews = value

    @property
    def unconcatenated_rews(self):
        return self._unconcatenated_rews

    @unconcatenated_rews.setter
    def unconcatenated_rews(self, value):
        assert isinstance(value, np.ndarray) or isinstance(value, list)
        self._unconcatenated_rews = value

    @property
    def next_obs(self):
        return self._next_obs

    @next_obs.setter
    def next_obs(self, value):
        assert isinstance(value, np.ndarray) or isinstance(value, list)
        self._next_obs = value

    @property
    def terminals(self):
        return self._terminals

    @terminals.setter
    def terminals(self, value):
        assert isinstance(value, np.ndarray) or isinstance(value, list)
        self._terminals = value

    ##################################
    ##################################

    def __len__(self) -> int:
        return len(self.paths)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: ({self._num_paths}, {self._num_data})"

    ##################################
    ##################################

    def add_rollouts(self, paths: List[PathDict]) -> None:
        """ Add new rollouts into our list of rollouts """
        assert len(paths) > 0, "Adding empty rollout"

        self.new_path_len = len(paths)
        self.new_data_len = int(np.sum([path['observation'].shape[0] for path in paths]))

        self.paths.extend(paths)
        if len(self.paths) > self._max_size:
            print("###########################")
            print(f"Exceed buffer max_size {self._max_size} by {len(self.paths) - self._max_size}, old path will be "
                  f"dropped")

        # if size exceed buffer's max size, drop old rollouts and keep new on instead
        self.paths = self.paths[-self._max_size:]
        self._num_paths = len(self.paths)

        # convert new rollouts into their component arrays, and append them onto our arrays
        (observations, actions, log_probs,
         next_observations, terminals,
         concatenated_rews, unconcatenated_rews) = utils.convert_listofrollouts(paths)

        # self.paths is empty, init elements with max_size
        if self._obs is None:
            self._obs = observations[-self._max_size:]
            self._acs = actions[-self._max_size:]
            self._log_probs = log_probs[-self._max_size:]
            self._next_obs = next_observations[-self._max_size:]
            self._terminals = terminals[-self._max_size:]
            self._concatenated_rews = concatenated_rews[-self._max_size:]
            self._unconcatenated_rews = unconcatenated_rews[-self._max_size:]
        # Append elements in new paths to paths and keep only latest max_size around
        else:
            self._obs = np.concatenate([self._obs, observations])[-self._max_size:]
            self._acs = np.concatenate([self._acs, actions])[-self._max_size:]
            self._log_probs = np.concatenate([self._log_probs, log_probs])[-self._max_size:]
            self._next_obs = np.concatenate([self._next_obs, next_observations])[-self._max_size:]
            self._terminals = np.concatenate([self._terminals, terminals])[-self._max_size:]
            self._concatenated_rews = np.concatenate([self._concatenated_rews, concatenated_rews])[-self._max_size:]
            if isinstance(unconcatenated_rews, list):
                self._unconcatenated_rews += unconcatenated_rews
            else:
                self._unconcatenated_rews.append(unconcatenated_rews)
        # update total num of data in buffer
        self._num_data += self._obs.shape[0]

    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts) -> np.ndarray:
        """
        Randomly select Rollouts
        :param: num_rollouts
        :type: int
        :return: random rollouts
        :type: np.array(List[PathDict])
        """
        assert isinstance(num_rollouts, int)
        assert len(self.paths) != 0, "No rollouts in Buffer"
        assert len(self.paths) >= num_rollouts, "Rollouts in Buffer is less than rollouts acquired "
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return np.array(self.paths)[rand_indices]

    def sample_recent_rollouts(self, num_rollouts: int = 1) -> np.ndarray:
        """
        Select Recent Rollouts
        :param: num_rollouts
        :type: int
        :return: recent n rollouts
        :type: List[PathDict]
        """
        assert isinstance(num_rollouts, int)
        assert len(self.paths) != 0, "No rollouts in Buffer"
        assert len(self.paths) >= num_rollouts, "Rollouts in Buffer is less than rollouts acquired "
        return np.array(self.paths)[-num_rollouts:]

    def sample_all_rollouts(self) -> np.ndarray:
        """
        Return All Rollouts from Buffer
        :return: random rollouts
        :type: np.array(List[PathDict])
        """
        assert len(self.paths) != 0, "No rollouts in Buffer"
        return np.array(self.paths)

    ########################################
    ########################################

    def sample_random_data(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample random transition steps of size batch_size
        """
        assert isinstance(batch_size, int)
        assert len(self.paths) != 0, "No path in Buffer"
        assert 0 <= batch_size <= self._obs.shape[0], "No enough transition steps in buffer"
        assert (self._obs.shape[0] == self._acs.shape[0]
                == self._concatenated_rews.shape[0]
                == self._next_obs.shape[0] == self._terminals.shape[0]), "num of data do not match"

        rand_indices = np.random.permutation(self._obs.shape[0])[:batch_size]
        return (self._obs[rand_indices], self._acs[rand_indices], self._concatenated_rews[rand_indices],
                self._next_obs[rand_indices], self._terminals[rand_indices])

    def sample_recent_data(self,
                           batch_size: int = 1,
                           concat_rew=True
                           ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, List], np.ndarray, np.ndarray]:
        """
        Sample recent transition steps of size batch_size
        :param batch_size: num of recent transition steps
        :param concat_rew:
        :return: observations, actions, unconcatenated_rews, next_observations, terminals
        """
        assert isinstance(batch_size, int) and batch_size >= 0
        assert len(self.paths) != 0, "No recent Data in Buffer"
        assert self._num_data >= batch_size, "Data in Buffer is less than data acquired "
        if concat_rew:
            return (self._obs[-batch_size:], self._acs[-batch_size:],
                    self._concatenated_rews[-batch_size:], self._next_obs[-batch_size:],
                    self._terminals[-batch_size:])
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -= 1
                num_recent_rollouts_to_return += 1
                num_datapoints_so_far += utils.get_pathlength(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]

            (observations, actions, log_probs, next_observations,
             terminals, concatenated_rews, unconcatenated_rews) = utils.convert_listofrollouts(rollouts_to_return)

            return observations, actions, unconcatenated_rews, next_observations, terminals

    def flush(self) -> None:
        """ Reset Replay Buffer """
        self._paths = []
        self._obs = None
        self._acs = None
        self._log_probs = None
        self._concatenated_rews = None
        self._unconcatenated_rews = None
        self._next_obs = None
        self._terminals = None
        self._num_paths = 0
        self._num_data = 0
        self._new_path_len = 0
        self._new_data_len = 0
