import abc
import numpy as np


class BasePolicy(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, obs: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, *args, **kwargs) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError
