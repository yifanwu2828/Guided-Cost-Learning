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
        c(x, u) = |Ay + b|^2 + w |u|^2 where y = mlp(x)
        """
        y = self.mlp(observation)
        z = torch.matmul(y, self.A) + self.b
        r = -(z * z).sum(-1) - self.w * (action * action).sum(-1)
        return r

    def update(self, demo_obs, demo_acs, sample_obs, sample_acs):
        """
        Computes the loss and updates the reward parameters
        Objective is to maximize sum of demo rewards and minimize sum of sample rewards
        Use importance sampling for policy samples
        """
        demo_obs = ptu.from_numpy(demo_obs)
        demo_acs = ptu.from_numpy(demo_acs)
        sample_obs = ptu.from_numpy(sample_obs)
        sample_acs = ptu.from_numpy(sample_acs)

        demo_return = self(demo_obs, demo_acs).sum(-1)
        sample_return = self(sample_obs, sample_acs).sum(-1)

        # TODO: Use importance sampling to estimate sample return 
        loss = -torch.mean(demo_return) + torch.mean(sample_return)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_reward_log = {
            "Training reward loss": ptu.to_numpy(loss)
        }
        return train_reward_log