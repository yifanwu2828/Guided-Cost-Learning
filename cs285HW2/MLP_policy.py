
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import  torch.distributions as distributions

import  pytorch_util as ptu

class MLPPolicyPG( nn.Module):

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
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
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
                chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1, #since value function
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        
        # Samples an action from the distribution in forward function
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = ptu.from_numpy(observation.astype(np.float32))
        m = self(observation)
        action = m.sample()
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, advantages, q_values=None, **kwargs):
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
        # Compute log prob of the selected actions
        m = self(observations)
        log_prob = m.log_prob(actions)

        # Independet Gaussian in each dimension, log(product of joint prob) = sum (log independent prob)
        if not self.discrete:
            log_prob = torch.sum(log_prob,dim=1)

        assert log_prob.size() == advantages.size()
        loss = -1* torch.sum(log_prob * advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline and q_values is not None:
            eps = 1e-8
            targets = (q_values - np.mean(q_values)) / (np.std(q_values)+ eps)
            targets = ptu.from_numpy(targets)
        
            ## TODO: use the `forward` method of `self.baseline` to get baseline predictions
            baseline_predictions = self.baseline(observations)
            # baseline_prediction -> value function
            baseline_predictions  = torch.squeeze(baseline_predictions, dim=1)
            assert  baseline_predictions.shape == targets.shape

            baseline_loss = F.mse_loss(baseline_predictions, targets)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_loss = ptu.to_numpy(loss)
        train_log = {'Training Loss': train_loss,}

        return train_log

    # This function defines the forward pass of the network.
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            return distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(observation)
            std = torch.exp(self.logstd)
            return distributions.Normal(mean, std)
    
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








    