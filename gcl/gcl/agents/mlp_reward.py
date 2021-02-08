import itertools
import torch
from torch import nn
from torch import optim

from gcl.scripts import pytorch_util as ptu

class MLPReward(nn.Module):
    """
    Defines a reward function given the current observation and action
    """
    def __init__(self, ac_dim, ob_dim, n_layers, size, output_size, learning_rate):
        super().__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.output_size = output_size
        self.learning_rate = learning_rate


        self.mlp = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu'
        )
        self.A = nn.Parameter(
            torch.zeros(self.output_size, self.output_size, dtype=torch.float32, device=ptu.device)
        )
        self.b = nn.Parameter(
            torch.zeros(self.output_size, dtype=torch.float32, device=ptu.device)
        )
        # GCL says this is not learnable but I did not find its value
        self.w = nn.Parameter(
            torch.zeros(1, dtype=torch.float32, device=ptu.device)
        )

        self.optimizer = optim.Adam(
            itertools.chain([self.A, self.b, self.w], self.mlp.parameters()),
            self.learning_rate
        )



    def forward(self, observation: torch.FloatTensor, action: torch.FloatTensor):
        """
        Returns the reward for the current state and action
        """
        y = self.mlp(observation)
        z = torch.mm(self.A, y) + self.b
        c = torch.dot(z, z) + self.w * torch.dot(action, action)

        return -c

    def update(self, observations, actions):
        """
        Computes the loss and updates the reward parameters
        Objective is to maximize sum of demo rewards and minimize sum of sample rewards
        Use importance sampling for policy samples
        """

        reward = self(observations, actions)

        raise NotImplementedError