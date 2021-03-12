import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from gcl.scripts import utils
from gcl.scripts import pytorch_util as ptu
from gcl.agents.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Policy for predicting a Gaussian action distribution
    """
    def __init__(self, ac_dim, ob_dim, n_layers, size,
                 learning_rate=1e-4, training=True, 
                 nn_baseline=False, **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # Only consider continuous action space
        self.logits_na = None
        self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                  output_size=self.ac_dim,
                                  n_layers=self.n_layers, size=self.size)
        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
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

    # query the policy with observation(s) to get selected action(s) 
    # and the corresponding log probability
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        # Return the action that the policy prescribes
        observation = ptu.from_numpy(observation.astype(np.float32))
        mean, std = self(observation)
        dist = distributions.MultivariateNormal(mean, torch.diag(std))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return ptu.to_numpy(action), ptu.to_numpy(log_prob)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # Returns both the action distribution
        mean = self.mean_net(observation)
        std = self.logstd.exp()
        return mean, std


#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    """
    Policy that uses policy gradient to update parameters
    """
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        
        self.baseline = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.baseline.to(ptu.device)
        self.baseline_optimizer = optim.Adam(
            self.baseline.parameters(),
            self.learning_rate,
        )

        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss
        mean, std = self(observations)
        dist = distributions.MultivariateNormal(mean, torch.diag(std))
        log_prob = dist.log_prob(actions)
        loss = -torch.mean(log_prob * advantages)

        # TODO: optimize `loss` using `self.optimizer`

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ## TODO: normalize the q_values to have a mean of zero and a standard deviation of one
        targets = utils.normalize(q_values, np.mean(q_values), np.std(q_values))
        targets = ptu.from_numpy(targets)

        ## TODO: use the `forward` method of `self.baseline` to get baseline predictions
        baseline_predictions = torch.squeeze(self.baseline(observations))
        
        ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
        ## [ N ] versus shape [ N x 1 ]
        ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
        assert baseline_predictions.shape == targets.shape
        
        # TODO: compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
        # HINT: use `F.mse_loss`
        baseline_loss = F.mse_loss(targets, baseline_predictions)

        # TODO: optimize `baseline_loss` using `self.baseline_optimizer`
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()

        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }
        return train_log

    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]

class MLPPolicyGPS(MLPPolicy):
    """
    Policy that uses guided policy search to update parameters
    """
    def __init__():
        raise NotImplementedError