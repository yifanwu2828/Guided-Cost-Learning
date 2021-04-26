import abc
import itertools
from typing import Tuple
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from gcl.infrastructure import pytorch_util as ptu
from gcl.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            # use in continuous action
            self.mean_net = None
            self.logstd = None

            # init weight
            self.logits_na.apply(ptu.initialize_weights)

            self.logits_na.to(ptu.device)
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            # use in discrete
            self.logits_na = None

            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            # init weight
            self.mean_net.apply(ptu.initialize_weights)

            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)

            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation.astype(np.float32))
        action_dist = self(observation)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return ptu.to_numpy(action), ptu.to_numpy(log_prob)

    # update/train this policy
    def update(self, observations, actions, **kwargs) -> dict:
        raise NotImplementedError

    def forward(self, observation: torch.FloatTensor) -> distributions.Distribution:
        if self.discrete:
            return distributions.Categorical(logits=self.logits_na(observation))
        else:
            mean = self.mean_net(observation)
            std = self.logstd.exp()
            action_dist = distributions.MultivariateNormal(mean, torch.diag(std))
            return action_dist


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, obs: np.ndarray, acs: np.ndarray, **kwargs,
    ) -> dict:
        self.optimizer.zero_grad()
        observations = ptu.from_numpy(obs.astype(np.float32))
        actions = ptu.from_numpy(acs.astype(np.float32))
        action_dist = self(observations)

        loss = -action_dist.log_prob(actions).mean()
        loss.backward()
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
