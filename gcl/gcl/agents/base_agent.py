import abc


class BaseAgent(object, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError
