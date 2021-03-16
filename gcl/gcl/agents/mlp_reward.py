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

    def update(self, demo_obs, demo_acs, sample_obs, sample_acs, log_probs):
        """
        Computes the loss and updates the reward parameters
        Objective is to maximize sum of demo rewards and minimize sum of sample rewards
        Use importance sampling for policy samples
        Recall that the MLE objective to maximize is:
            1/N sum_{i=1}^N return(tau_i) - log Z
          = 1/N sum_{i=1}^N return(tau_i) - log E_{tau ~ p(tau)} [exp(return(tau))]
          = 1/N sum_{i=1}^N return(tau_i) - log E_{tau ~ q(tau)} [p(tau) / q(tau) * exp(return(tau))]
          = 1/N sum_{i=1}^N return(tau_i) - log (sum_j exp(return(tau_j)) * w(tau_j) / sum_j w(tau_j))
        where w(tau) = p(tau) / q(tau) = 1 / prod_t pi(a_t|s_t) 
        """
        demo_obs = ptu.from_numpy(demo_obs)
        demo_acs = ptu.from_numpy(demo_acs)
        sample_obs = ptu.from_numpy(sample_obs)
        sample_acs = ptu.from_numpy(sample_acs)
        log_probs = torch.squeeze(ptu.from_numpy(log_probs), dim=-1)

        demo_return = self(demo_obs, demo_acs).sum(-1)
        sample_return = self(sample_obs, sample_acs).sum(-1)
        # using 1/N sum_{i=1}^N return(tau_i) - log 1/M (sum_j exp(return(tau_j)) / prod_t pi(a_t|s_t) )
        w = sample_return - torch.sum(log_probs, dim=1)
        w_max = torch.max(w)

        # TODO: Use importance sampling to estimate sample return 
        # trick to avoid overflow: log(exp(x1) + exp(x2)) = x + log(exp(x1-x) + exp(x2-x)) where x = max(x1, x2)
        loss = -torch.mean(demo_return) + torch.log(torch.sum(torch.exp(w - w_max))) + w_max

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_reward_log = {"Training reward loss": ptu.to_numpy(loss)}
        return train_reward_log
