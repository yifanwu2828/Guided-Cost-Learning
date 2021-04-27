from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions

import gcl.infrastructure.pytorch_util as ptu

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, ac_dim, ob_dim, n_layers, size, act_limit=1):
        super().__init__()
        self.mlp = ptu.build_mlp(
            input_size=ob_dim,
            output_size=n_layers,
            n_layers=n_layers,
            size=size,
            activation='tanh',  # default relu
            output_activation='identity',  # default relu
        )
        self.mean_layer = nn.Linear(size, ac_dim)
        self.logstd_layer = nn.Linear(size, ac_dim)
        self.act_limit = act_limit

    def get_action(
            self,
            obs: np.ndarray,
            deterministic=False,
            with_logprob=True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Query the policy with observation(s) to get selected action(s)
        and the corresponding log probability
        :param obs: observation
        :param deterministic: output deterministic action
        :param with_logprob: whether to output log_prob or not
        :return: action, log_prob
        """
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation.astype(np.float32))

        pi_dist = self(observation)
        if deterministic:
            # deterministic Only used for evaluating policy at test time.
            net_out = self.mlp(observation)
            mean = self.mean_layer(net_out)
            pi_action = mean
        else:
            pi_action = pi_dist.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            pi_logprob = pi_dist.log_prob(pi_action).sum(axis=-1)
            pi_logprob -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            pi_logprob = None

        pi_action = torch.tanh(pi_action)
        # if the env has actions space [-a, a],
        # unnormalize the action by multiplying a to action sample from tanh(Gaussian).
        pi_action = self.act_limit * pi_action

        return ptu.to_numpy(pi_action), ptu.to_numpy(pi_logprob)

    def forward(
            self,
            observation: torch.FloatTensor,
    ) -> distributions.Distribution:

        net_out = self.mlp(observation)
        mean = self.mean_layer(net_out)
        logstd = self.logstd_layer(net_out)
        logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
        std = logstd.exp()

        # Pre-squash distribution
        pi_dist = distributions.normal.Normal(mean, std)
        # pi_dist = distributions.MultivariateNormal(mean, torch.diag(std))
        return pi_dist



class MLPQFunction(nn.Module):
    """ Q Function """
    def __init__(self, ac_dim, ob_dim, n_layers, size):
        super().__init__()
        self.q = ptu.build_mlp(
            input_size=ac_dim+ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
            activation='tanh',
            output_activation='identity',
        )

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        # Ensure q has right shape.
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):

    def __init__(self,
                 ob_dim: int,
                 ac_dim: int,
                 n_layers: int = 2,
                 hidden_size: int = 64,  # 256
                 act_limit: int = 1
                 ):
        super().__init__()

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(ac_dim, ob_dim, hidden_size,  act_limit)
        self.q1 = MLPQFunction(ac_dim, ob_dim, n_layers, hidden_size)
        self.q2 = MLPQFunction(ac_dim, ob_dim, n_layers, hidden_size)

    def act(self, obs, deterministic=False, with_logprob=True):
        with torch.no_grad():
            act, log_prob = self.pi(obs, deterministic, with_logprob)
            return act, log_prob
