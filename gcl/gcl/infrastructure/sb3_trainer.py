import time
from collections import OrderedDict
from typing import List, Dict, Tuple, Sequence, Optional

import gym
import numpy as np
import copy
import torch
import pickle
from tqdm import tqdm
from icecream import ic


from gcl.infrastructure import pytorch_util as ptu, utils
from gcl.infrastructure.utils import PathDict
from gcl.infrastructure.logger import Logger
from gcl.policies.base_policy import BasePolicy


class GCL_Trainer(object):
    """ GCL_Trainer"""

    def __init__(self, params):

        #############
        # INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # init gpu
        self.params['use_gpu'] = not self.params['no_gpu']
        ptu.init_gpu(
            use_gpu=self.params['use_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        # ENV
        #############

        # Make the gym environment
        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = params.get('ep_len')
        if self.params['ep_len'] is None:
            print("Episodes length is not specified, Using env.max_episode_steps")
            try:
                self.params['ep_len']: int = self.env.spec.max_episode_steps
            except AttributeError:
                self.params['ep_len']: int = self.env._max_episode_steps  # Access to a protected member
        ic(self.params['ep_len'])

        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        ########################################
        ########################################
        # Observation Dimension
        # Are the observations images?
        is_img: bool = len(self.env.observation_space.shape) > 2
        ob_dim: int = self.env.observation_space.shape if is_img else self.env.observation_space.shape[0]
        self.params['agent_params']['ob_dim'] = ob_dim
        ########################################
        # Action Dimension
        # Is this env continuous, or discrete?
        discrete: bool = isinstance(self.env.action_space, gym.spaces.Discrete)
        ac_dim: int = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['discrete'] = discrete
        self.params['agent_params']['ac_dim'] = ac_dim

        # Action limit for clamping:
        # critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]
        ########################################
        ########################################

        # simulation timestep, will be used for video saving
        # Frame Rate
        # if 'model' in dir(self.env):
        #     self.fps = 1 / self.env.model.opt.timestep
        # elif 'env_wrappers' in self.params:
        #     self.fps = 30  # This is not actually used when using the Monitor wrapper
        # elif 'video.frames_per_second' in self.env.env.metadata.keys():
        #     self.fps = self.env.env.metadata['video.frames_per_second']
        # else:
        #     self.fps = 10
        self.fps = 10
        ########################################
        ########################################

        # Init total ENV steps and initial_return
        self.total_envsteps = None
        self.initial_return = None

        # Timer
        self.start_time = None

        # Logging Flag
        self.log_video = None
        self.log_metrics = None

        #############
        # AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    ##################################

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    ##################################

    ############################################################################

    def run_training_loop(self,
                          n_iter: int,
                          collect_policy: BasePolicy,
                          eval_policy: BasePolicy):
        """
        :param n_iter:  number of () iterations
        :param collect_policy:
        :param eval_policy:
        :return
        """
        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        train_logs_lst = []

        n_iter_loop = tqdm(range(n_iter), desc="Policy Gradient", leave=False)
        for itr in n_iter_loop:
            print(f"\n********** Iteration {itr} ************")

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.log_metrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            with torch.no_grad():
                training_returns = self.collect_training_trajectories(collect_policy, self.params['batch_size'])

            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)
            self.buffer_status()

            # train agent (using sampled data from replay buffer)
            train_logs = self.train_policy()
            train_logs_out = train_logs[0]['Training_Policy_Loss']
            train_logs_lst.append(train_logs_out)

            # log/save
            if self.log_video or self.log_metrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths, train_logs)

                if self.params['save_params']:
                    self.agent.save(f"{self.params['logdir']}/agent_itr_{itr}.pt")
            n_iter_loop.set_postfix(Training_Loss=train_logs_out)

        return train_logs_lst

    ####################################
    ####################################

    def collect_training_trajectories(self,
                                      collect_policy: BasePolicy,
                                      batch_size: int,
                                      ) -> Tuple[List[PathDict], int, Optional[List[PathDict]]]:
        """
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # Init var
        train_video_paths = None
        paths: List[PathDict]

        print("\nCollecting data to be used for training...")

        paths, envsteps_this_batch = utils.sample_trajectories(
            env=self.env,
            policy=collect_policy,
            agent=self.agent,
            min_timesteps_per_batch=batch_size,
            max_path_length=self.params['ep_len'],
            evaluate=True
        )
        print(f"\n--envsteps_this_batch: {envsteps_this_batch}")

        return paths, envsteps_this_batch, train_video_paths

    ########################################################################################

    def train_policy(self) -> List[Sequence[Dict[str, np.ndarray]]]:
        """
        Policy Gradient
        """
        print('\nTraining agent using sampled data from replay buffer...')
        train_policy_logs = []
        for train_step in range(self.params['num_policy_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['train_batch_size'], demo=False)
            train_log = self.agent.train_policy(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            train_policy_logs.append(train_log)

        return train_policy_logs

    ########################################################################################

    def buffer_status(self) -> None:
        """ Show length and size of buffers"""
        samp_paths_len = len(self.agent.sample_buffer)
        samp_data_len = self.agent.sample_buffer.num_data
        print(f"Sample_buffer_size: {samp_paths_len}, {samp_data_len}"
              f"\tAverage ep_len: {samp_data_len / samp_paths_len :.3f}")
        print("##########################################################################")
