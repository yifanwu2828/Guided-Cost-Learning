from abc import ABCMeta
from typing import Optional, List, Union

import numpy as np
import torch
from icecream import ic
from stable_baselines3 import A2C, SAC, PPO, HER

from .base_agent import BaseAgent
from .mlp_reward import MLPReward
from gcl.infrastructure.replay_buffer import ReplayBuffer
from gcl.infrastructure.wrapper import LearningReward
from gcl.infrastructure.utils import PathDict
import gcl.infrastructure.pytorch_util as ptu


ALGO = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "her": HER,
    }


class GCL_AgentSB3(BaseAgent, metaclass=ABCMeta):
    """ GCL Agent migrate with SB3"""
    def __init__(self, env, agent_params: dict):
        super(GCL_AgentSB3, self).__init__()

        # Init vars
        self.env = env
        self.agent_params = agent_params
        self.model_class = ALGO[agent_params["model_class"]]

        # Init reward function
        self.reward = MLPReward(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['output_size'],
            learning_rate=self.agent_params['learning_rate']
        )
        # set mode to train
        self.reward.train()

        # Apply REW wrapper to env
        self.env = LearningReward(self.env, self.reward, device=ptu.device)
        ic(env)

        # actor/policy
        if agent_params["model_class"] == 'ppo':
            self.actor = self.model_class(
                # key
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.agent_params.get('policy_lr', 3e-4),
                n_steps=self.agent_params.get('n_steps', 2048),
                batch_size=self.agent_params.get('batch_size', 64),
                n_epochs=self.agent_params.get('n_epochs', 10),
                gamma=self.agent_params.get('gae_lambda', 0.95),
                clip_range=self.agent_params.get('clip_range', 0.2),
                clip_range_vf=self.agent_params.get('clip_range_vf', None),

                # utils
                tensorboard_log=self.agent_params.get('tensorboard_log', None),
                # (Only available when passing string for the environment)
                create_eval_env=self.agent_params.get('create_eval_env', False),
                policy_kwargs=self.agent_params.get('policy_kwargs', None),
                verbose=self.agent_params.get('verbose', 1),
                seed=self.agent_params.get('seed', 42),
                device=ptu.device  # "auto"
            )
            self.log_interval = 1

        elif agent_params["model_class"] == 'a2c':
            self.actor = self.model_class(
                # key
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.agent_params.get('policy_lr', 3e-4),
                n_steps=self.agent_params.get('n_steps', 2000),  # -- diff
                gamma=self.agent_params.get('gae_lambda', 0.99),
                gae_lambda=self.agent_params.get('n_steps', 1.0),
                normalize_advantage=self.agent_params.get('normalize_advantage', False),

                # utils
                tensorboard_log=self.agent_params.get('tensorboard_log', None),
                create_eval_env=self.agent_params.get('create_eval_env', False),
                policy_kwargs=self.agent_params.get('policy_kwargs', None),
                verbose=self.agent_params.get('verbose', 1),
                seed=self.agent_params.get('seed', 42),
                device=ptu.device  # "auto"
            )
            self.log_interval = 4

        elif agent_params["model_class"] == 'sac':
            self.actor = self.model_class(
                # key
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.agent_params.get('policy_lr', 3e-4),
                buffer_size=self.agent_params.get('buffer_size', 1_000_000),
                learning_starts=100,
                batch_size=self.agent_params.get('batch_size', 256),
                tau=self.agent_params.get('tau', 0.005),
                gamma=self.agent_params.get('gae_lambda', 0.99),
                train_freq=self.agent_params.get('train_freq', 1),  # SEE DOC
                gradient_steps=self.agent_params.get('gradient_steps', 1),
                action_noise=self.agent_params.get('action_noise', None),
                optimize_memory_usage=self.agent_params.get('optimize_memory_usage', False),
                # update the target network every ``target_network_update_freq``
                # gradient steps.
                target_update_interval=self.agent_params.get('target_update_interval', 1),

                # utils
                tensorboard_log=self.agent_params.get('tensorboard_log', None),
                create_eval_env=self.agent_params.get('create_eval_env', False),
                policy_kwargs=self.agent_params.get('policy_kwargs', None),
                verbose=self.agent_params.get('verbose', 1),  # verbosity level: 0 no output, 1 info, 2 debug
                seed=self.agent_params.get('seed', 42),
                device=ptu.device  # "auto"
            )
            self.log_interval = 4

        elif agent_params["model_class"] == 'her':
            pass
            self.log_interval = 4

        else:
            raise NotImplementedError("Please Provide Valid Policy")

        # Replay buffers: demo holds expert demonstrations and sample holds policy samples
        self.demo_buffer = ReplayBuffer(1_000_000)
        self.sample_buffer = ReplayBuffer(1_000_000)
        self.background_buffer = ReplayBuffer(1_000_000)

        print(f"Agent\nagent device: {ptu.device}")
        ic(self.agent_params.get('policy_lr'))
    #####################################################
    #####################################################

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    #####################################################
    #####################################################

    def save(self, PATH) -> None:
        assert isinstance(PATH, str)
        torch.save(
            {
                "reward": self.reward,
            }, PATH
        )

    #####################################################
    #####################################################

    def train_reward(self, demo_batch: np.ndarray, sample_batch: np.ndarray) -> dict:
        """
        Train the reward function
        :param demo_batch: demo rollouts
        :param sample_batch: sample rollouts
        :return: reward_log
        :type: dict
        """
        # unpack rollouts into obs, act, log_probs
        demo_obs, demo_acs = [], []
        for demo_path in demo_batch:
            demo_obs.append(demo_path['observation'])
            demo_acs.append(demo_path['action'])

        sample_obs, sample_acs, sample_log_probs = [], [], []
        for sample_path in sample_batch:
            sample_obs.append(sample_path['observation'])
            sample_acs.append(sample_path['action'])
            sample_log_probs.append(sample_path['log_prob'])

        # Estimate gradient loss and update parameters
        reward_log = self.reward.update(demo_obs, demo_acs, sample_obs, sample_acs, sample_log_probs)

        return reward_log

    ##################################################################################################
    def train_policy(self, total_timesteps):
        self.actor.learn(
            total_timesteps=total_timesteps,
            callback=None,
            log_interval=self.log_interval,     # deafult: PPO=1, A2C=100, SAC=4, HER=4,
        )
    #####################################################
    #####################################################

    def add_to_buffer(self, paths: List[PathDict], demo=False, background=False) -> None:
        """
        Add paths to demo or sample buffer
        """
        if demo:
            self.demo_buffer.add_rollouts(paths)
        elif background:
            self.background_buffer.add_rollouts(paths)
        else:
            self.sample_buffer.add_rollouts(paths)

    #####################################################
    #####################################################

    def sample_rollouts(self, num_rollouts: int, demo=False) -> np.ndarray:
        """
        Randomly sample paths from demo or sample buffer
        :param: num_rollouts
        :param: if demo sample from demo buffer, else sample from sample buffer
        :return: random rollouts (paths) from buffer
        """
        assert num_rollouts > 0
        if demo:
            return self.demo_buffer.sample_random_rollouts(num_rollouts)
        else:
            return self.sample_buffer.sample_random_rollouts(num_rollouts)

    def sample_recent_rollouts(self, num_rollouts: int, demo=False) -> np.ndarray:
        """
        Sample recent paths from demo or sample buffer
        :param: num_rollouts
        :param: if demo sample from demo buffer, else sample from sample buffer
        :return: random rollouts (paths) from buffer
        """
        assert num_rollouts > 0 and isinstance(num_rollouts, int)
        if demo:
            return self.demo_buffer.sample_recent_rollouts(num_rollouts)
        else:
            return self.sample_buffer.sample_recent_rollouts(num_rollouts)

    def sample_background_rollouts(self, batch_size: Optional[int] = 1000,
                                   recent=False, all_rollouts=False) -> np.ndarray:
        assert not (recent and all_rollouts)
        if all_rollouts:
            return self.background_buffer.sample_all_rollouts()
        elif recent:
            assert isinstance(batch_size, int) and batch_size >= 0
            return self.background_buffer.sample_recent_rollouts(batch_size)
        else:
            assert isinstance(batch_size, int) and batch_size >= 0
            return self.background_buffer.sample_random_rollouts(batch_size)

    #####################################################
    #####################################################

    def sample(self, batch_size: int, demo=False):
        """
        Sample recent transition steps of size batch_size
        """
        assert isinstance(batch_size, int) and batch_size >= 0
        if demo:
            return self.demo_buffer.sample_recent_data(batch_size, concat_rew=False)
        else:
            return self.sample_buffer.sample_recent_data(batch_size, concat_rew=False)

