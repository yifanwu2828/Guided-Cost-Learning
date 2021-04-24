import time
from collections import OrderedDict
from typing import List, Dict, Tuple, Sequence, Optional

import gym
import numpy as np
import copy
import torch
import pickle
from tqdm import tqdm

from gcl.infrastructure import pytorch_util as ptu, utils
from gcl.infrastructure.utils import PathDict
from gcl.infrastructure.logger import Logger
from gcl.policies.base_policy import BasePolicy

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):
    """ PG_Trainer"""

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
        try:
            self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        except AttributeError:
            self.params['ep_len']: int = self.env._max_episode_steps  # Access to a protected member

        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Observation Dimension
        # Are the observations in Dict?
        if isinstance(self.env.observation_space, gym.spaces.dict.Dict):
            # TODO adjust ob_dim for goal_env
            ob_dim: int = np.sum([self.env.observation_space[key].shape[0] for key in self.env.observation_space])

        # Are the observation continuous?
        elif isinstance(self.env.observation_space, gym.spaces.box.Box):
            # Are the observations images?
            is_img: bool = len(self.env.observation_space.shape) > 2
            ob_dim: int = self.env.observation_space.shape if is_img else self.env.observation_space.shape[0]

        # Are the observation discrete?
        elif isinstance(self.env.observation_space, gym.spaces.Discrete):
            ob_dim: int = self.env.observation_space.n
        else:
            raise ValueError("env.observation_space type not found")

        ########################################
        ########################################
        # Action Dimension
        # Is this env continuous, or discrete?
        discrete: bool = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete

        ac_dim: int = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        # -3 for goal env
        self.params['agent_params']['ob_dim'] = ob_dim-3

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
        return f"RL_Trainer"

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
            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False
            self.log_video = self.log_video

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

    def perform_logging(self, itr: int, paths: List[PathDict],
                        eval_policy: BasePolicy,
                        train_video_paths: List[PathDict], all_logs: List[Sequence[Dict]]
                        ) -> None:
        """Log metrics and Record Video"""
        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        with torch.no_grad():
            eval_returns = utils.sample_trajectories(self.env,
                                                     eval_policy, self.agent,
                                                     min_timesteps_per_batch=self.params['eval_batch_size'],
                                                     max_path_length=self.params['ep_len'],
                                                     evaluate=True
                                                     )
        eval_paths, eval_envsteps_this_batch = eval_returns

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths is not None:
            print('\nCollecting video rollouts eval')
            with torch.no_grad():
                eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, self.agent,
                                                               MAX_NVIDEO, MAX_VIDEO_LEN, True, evaluate=True)

            # save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            print("\n---------------------------------------------------")
            for key, value in logs.items():
                print(f'|\t{key} : {value:.3f}')
                self.logger.log_scalar(value, key, itr)
            print("---------------------------------------------------")
            print('Done logging...\n\n')

            self.logger.flush()

    def buffer_status(self) -> None:
        """ Show length and size of buffers"""
        samp_paths_len = len(self.agent.sample_buffer)
        samp_data_len = self.agent.sample_buffer.num_data
        print(f"Sample_buffer_size: {samp_paths_len}, {samp_data_len}"
              f"\tAverage ep_len: {samp_data_len / samp_paths_len :.3f}")
        print("##########################################################################")
