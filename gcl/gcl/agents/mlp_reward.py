from typing import List
import warnings

import numpy as np
import torch
from torch import nn
from torch import optim
from dask import delayed
from gcl.infrastructure import pytorch_util as ptu
warnings.filterwarnings('always')


class MLPReward(nn.Module):
    """
    Defines a reward function given the current observation and action
    """

    def __init__(self, ac_dim, ob_dim, n_layers, size, output_size, learning_rate):
        super().__init__()

        self.ac_dim: int = ac_dim
        self.ob_dim: int = ob_dim
        self.n_layers: int = n_layers
        self.size: int = size
        self.output_size: int = output_size
        self.learning_rate: float = learning_rate

        self.mlp: nn.Module = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            activation='identity',
            output_activation='relu'
        )
        self.initialize_weights()

        self.A = nn.Parameter(
            torch.ones(self.output_size, self.output_size, dtype=torch.float32, device=ptu.device)
        )
        self.b = nn.Parameter(
            torch.ones(self.output_size, dtype=torch.float32, device=ptu.device)
        )
        # GCL says this is not learnable but I did not find its value
        self.w = nn.Parameter(
            torch.ones(1, dtype=torch.float32, device=ptu.device)
        )

        self.optimizer = optim.Adam(
            [
                {'params': self.A, 'lr': 5e-3},
                {'params': self.b, 'lr': 5e-3},
                {'params': self.w, 'lr': 5e-3},
                {'params': self.mlp.parameters()}
            ],
            lr=self.learning_rate
        )
        print("MLP REW", ptu.device)
        # self.mlp.to(ptu.device)

    def initialize_weights(self):
        """weight initialization"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    #####################################################
    #####################################################

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def save(self, PATH):
        assert isinstance(PATH, str)
        torch.save(
            {
                "MLPReward": self.state_dict(),
                "mlp_state_dict": self.mlp.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                "A": self.self.A,
                "b": self.b,
                "w": self.w,
            }, PATH
        )
    #####################################################
    #####################################################

    def forward(self, observation: torch.FloatTensor, action: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns the reward for the current state and action
        Cost function: c(x, u) = |Ay + b|^2 + w |u|^2 where y = mlp(x)
        """
        y = self.mlp(observation)
        z = torch.matmul(y, self.A) + self.b
        cost = (z * z).sum(-1) + self.w * (action * action).sum(-1)
        reward = - cost

        return reward

    #####################################################
    #####################################################
    def update(self, demo_obs: List[np.ndarray], demo_acs: List[np.ndarray],
               sample_obs: List[np.ndarray], sample_acs: List[np.ndarray],
               sample_log_probs: List[np.ndarray]) -> dict:
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
        assert len(demo_obs) == len(demo_acs), "Length of Demo trajs do not match"
        assert len(sample_obs) == len(sample_acs) == len(sample_log_probs), "Length of Sample trajs do not match"

        # Demo Return
        demo_rollouts_return = []
        for demo_ob, demo_ac in zip(demo_obs, demo_acs):
            demo_ob, demo_ac = np.array(demo_ob, dtype=np.float32), np.array(demo_ac, dtype=np.float32)
            demo_rew = self(ptu.from_numpy(demo_ob), ptu.from_numpy(demo_ac))
            # Append total reward along that trajectory
            demo_rollouts_return.append(demo_rew.sum(-1))

        # Sample Return
        sample_rollouts_return = []
        sample_rollouts_logprob = []
        for sample_ob, sample_ac, sample_log_prob in zip(sample_obs, sample_acs, sample_log_probs):
            # Sample returns
            sample_ob, sample_ac = np.array(sample_ob, dtype=np.float32), np.array(sample_ac, dtype=np.float32)
            sample_rew = self(ptu.from_numpy(sample_ob), ptu.from_numpy(sample_ac))
            sample_rollouts_return.append(sample_rew.sum(-1))
            # Sample log_probs
            assert isinstance(sample_log_prob, np.ndarray)
            sample_log_prob = torch.squeeze(ptu.from_numpy(sample_log_prob), dim=-1)
            sum_log_prob = sample_log_prob.sum(-1)
            sample_rollouts_logprob.append(sum_log_prob)

        demo_return = torch.stack(demo_rollouts_return)
        sample_return = torch.stack(sample_rollouts_return)
        sum_log_probs = torch.stack(sample_rollouts_logprob)

        '''
        Directly compute the importance ratio 
        wj  = exp(sum(r)) / prod(exp(log_prob))
            = exp(sum(r)) / exp(sum(log_prob))
            = exp(sum(r) - sum(log_prob))   Let sum(r) - sum(log_prob) be x = [x1, ...xj]
        ->
        importance weights = wj/sum(wj)
        '''
        x = sample_return - sum_log_probs
        weights = torch.exp(x - torch.logsumexp(x, -1))
        # weights should sum to 1
        w = weights.sum(-1).item()
        if abs(w - 1) > 1e-2:
            warnings.warn(f'Sum of Weights larger than one:{w}')

        demo_loss = torch.mean(demo_return)
        sample_loss = torch.sum(weights * sample_return)
        loss = -demo_loss + sample_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_reward_log = {"Training_Reward_Loss": ptu.to_numpy(loss)}
        return train_reward_log
