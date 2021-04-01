import pickle
import time
from functools import lru_cache
from collections import OrderedDict
import itertools
from typing import List, Optional, Tuple, Dict, Sequence, Any
import copy

import gym
import gym_nav
import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

import pytorch_util as ptu
import utils
from utils import PathDict
from gcl.agents.base_policy import BasePolicy
from logger import Logger

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class GCL_Trainer(object):
    """ GCL_Trainer """

    def __init__(self, params: dict):

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
        print("Trainer", ptu.device)
        #############
        # ENV
        #############

        # Make the gym environment
        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len']: int = self.env.max_steps  # TODO: may need to change this for different ENV
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete: bool = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete

        # Are the observations images?
        is_img: bool = len(self.env.observation_space.shape) > 2

        # Observation and action sizes
        ob_dim: int = self.env.observation_space.shape if is_img else self.env.observation_space.shape[0]
        ac_dim: int = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        # Frame Rate
        if 'model' in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30  # This is not actually used when using the Monitor wrapper
        # elif 'video.frames_per_second' in self.env.env.metadata.keys():
        #     self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10

        # Init total ENV steps and initial_return
        self.total_envsteps: int = 0
        self.initial_return: float = 0

        # Timer
        self.start_time = None

        # Logging Flag
        self.log_video: bool = False
        self.log_metrics: bool = False

        #############
        # AGENT
        #############
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])
        self.test_agent = None

    ##################################

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    ##################################

    ############################################################################################
    def run_training_loop(self, n_iter: int,
                          collect_policy, eval_policy,
                          expert_data=None, expert_policy=None
                          ) -> Tuple[Any, Any]:
        """
        Perform Algorithm 1 Guided cost learning
        :param n_iter: number of iterations
        :param collect_policy: q_k
        :param eval_policy: q_k at t
        :param expert_data: D_demo
        :param expert_policy: pi*
        """
        # Init vars at beginning of training
        self.total_envsteps: int = 0
        self.start_time: float = time.time()

        train_log_lst, policy_log_lst = [], []
        demo_paths: List[PathDict]
        samp_paths: List[PathDict]

        #####################################################################
        # 1. Add demonstrations to replay buffer
        with torch.no_grad():
            demo_paths, _, _ = self.collect_demo_trajectories(expert_data, expert_policy,
                                                              ntrajs=self.params['demo_size'],
                                                              render=False, verbose=True)
        self.agent.add_to_buffer(demo_paths, demo=True)
        print(f'\nNum of Demo rollouts collected:{self.agent.demo_buffer.num_paths}')
        print(f'Num of Demo transition steps collected:{self.agent.demo_buffer.num_data}')
        utils.toc(self.start_time, "Finish Loading Expert Demonstrations", ftime=True)

        #####################################################################
        # 2.
        n_iter_loop = tqdm(range(n_iter), desc="Guided Cost Learning", leave=False)
        for itr in n_iter_loop:
            print(f"\n********** Iteration {itr} ************")

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.log_metrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # 3. Generate fresh samples D_traj from current trajectory distribution q_k (collect_policy)
            # collect trajectories, to be used for training
            # On-policy PG need to collect new trajectories at *every* iteration
            samp_start_time = time.time()
            with torch.no_grad():
                training_returns = self.collect_training_trajectories(
                    collect_policy=collect_policy,
                    batch_size=self.params["train_batch_size"]
                )
            # utils.toc(samp_start_time, "collect training trajs")
            samp_paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            del training_returns

            # 4. Append samples D_traj to D_samp
            self.agent.add_to_buffer(samp_paths)
            # show status
            self.buffer_status(demo=True, samp=True)

            # # testing
            # if itr == 240:
            #     self.test_agent = copy.deepcopy(self.agent)
            #     for idx in tqdm(range(200)):
            #         with torch.no_grad():
            #             test_returns = self.collect_training_trajectories(
            #                 collect_policy=self.test_agent.actor,
            #                 batch_size=self.params["train_batch_size"]
            #             )
            #         test_paths, envsteps_this_batch, _ = test_returns
            #         self.test_agent.test_buffer.add_rollouts(test_paths)
            #         policy_logs = self.perform_pg2opt()
            #         self.show_logs(idx, envsteps_this_batch,
            #                        test_paths,
            #                        [0], policy_logs,
            #                        logging=False
            #                        )

            # 5. Use D_{samp} to update cost c_{\theta}
            rew_start_time = time.time()
            reward_logs = self.train_reward()  # Algorithm 2
            # utils.toc(rew_start_time, "Update Reward")

            # 6. Update q_k(\tau) using D_{traj} and using GPS or PG
            ply_start_time = time.time()
            policy_logs = self.train_policy()
            # utils.toc(rew_start_time, "Update Policy")

            # log
            # print('\nBeginning logging procedure...')
            self.show_logs(itr, self.total_envsteps,
                           samp_paths,
                           reward_logs, policy_logs,
                           logging=True
                           )

            # # log/save
            # if self.log_video or self.log_metrics:
            #     # perform logging
            #     print('\nBeginning logging procedure...')
            #     self.perform_logging(itr, samp_paths, eval_policy,
            #                          train_video_paths, reward_logs, policy_logs)
            #     # save_params
            #     if self.params['save_params']:
            #         self.agent.save(f"{self.params['logdir']}/agent_itr_{itr}.pt")

            for r, p in itertools.zip_longest(reward_logs, policy_logs):
                if r:
                    reward_loss = float(r['Training_Reward_Loss'])
                    train_log_lst.append(reward_loss)
                if p:
                    policy_loss = float(p["Training_Policy_Loss"])
                    policy_log_lst.append(policy_loss)

            save_itr = [500, 550, 600, 650]
            if itr in save_itr:
                fname1 = f"test_gcl_reward_{itr}.pth"
                reward_model = self.agent.reward
                torch.save(reward_model, fname1)
                fname2 = f"test_gcl_policy_{itr}.pth"
                policy_model = self.agent.actor
                torch.save(policy_model, fname2)
            # update progress bar
            n_iter_loop.set_postfix()
        return train_log_lst, policy_log_lst

    ############################################################################################
    @lru_cache(maxsize=3)
    def collect_demo_trajectories(self,
                                  expert_data: Optional[str] = None, expert_policy: Optional[str] = None,
                                  ntrajs: int = 100, demo_batch_size: int = 1000,
                                  render=False, verbose=False
                                  ) -> Tuple[List[PathDict], int, Optional[List[PathDict]]]:
        """
        :param: expert_data:  relative path to saved
        :param: expert_policy:  relative path to saved expert policy
        :param: render: show video of demo trajs
        :param: verbose: evaluate expert policy and print metrics
        :return:
            paths: a list of trajectories with len = self.params['demo_size']
                    each trajectory is a dict {obs, image_obs, acs, log_probs, rewards, next_obs, terminals}
        """
        assert not (expert_data and expert_policy), "Choose either expert_data or expert_policy"
        # Init var
        render_mode: str = 'human' if render else 'rgb_array'
        demo_paths: List[PathDict]
        envsteps_this_batch: int = 0
        demo_video_paths: Optional[List[PathDict]] = None

        # Load expert policy or expert demonstrations D_demo
        if expert_data:
            print('\nLoading saved demonstrations...')
            with open(expert_data, 'rb') as f:
                # TODO: load data may not through pickle
                demo_paths = pickle.load(f)
            # TODO: sample self.params['demo_size'] from demo_paths
            return demo_paths[: ntrajs], 0, None

        elif expert_policy:
            expert_policy_model = PPO.load(expert_policy)
            print('\nRunning expert policy to collect demonstrations...')

            demo_paths = utils.sample_n_trajectories(
                self.env,
                policy=expert_policy_model,
                agent=self.agent,
                ntrajs=self.params['demo_size'],
                max_path_length=self.params['ep_len'],
                render=render,
                render_mode=render_mode,
                expert=True
            )

            if verbose:
                utils.evaluate_model(self.params['env_name'], expert_policy_model, num_episodes=100)
        else:
            raise ValueError('Please provide either expert demonstrations or expert policy')
        return demo_paths, envsteps_this_batch, demo_video_paths

    ############################################################################################

    def collect_training_trajectories(self,
                                      collect_policy: BasePolicy,
                                      batch_size: int
                                      ) -> Tuple[List[PathDict], int, Optional[List[PathDict]]]:
        """
        :param collect_policy:  the current policy which we use to collect data
        :param batch_size:  the number of transition steps or trajectories to collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # Init var
        paths: List[PathDict]
        envsteps_this_batch: int
        train_video_paths: Optional[List[PathDict]] = None

        print("\nCollecting sample trajectories to be used for training...")

        paths, envsteps_this_batch = utils.sample_trajectories(
            env=self.env,
            policy=collect_policy,
            agent=self.agent,
            min_timesteps_per_batch=batch_size,
            max_path_length=self.params['ep_len'],
        )
        # print(f"\n--envsteps_this_batch: {envsteps_this_batch}")

        # if self.log_video:
        #     print('\nCollecting train rollouts to be used for saving videos...')
        #     # TODO look in utils and
        #     pass
        #
        # # TODO: add logging
        # if self.log_metrics:
        #     # TODO:# what should be log in this function
        #     pass

        return paths, envsteps_this_batch, train_video_paths

    ############################################################################################
    def train_reward(self) -> List[Dict]:
        """
        Algorithm 2: Nonlinear IOC with stochastic gradients 
        """
        print("\nUpdating reward parameters...")
        reward_logs = []

        # 1.
        K_train_reward_loop = range(self.params['num_reward_train_steps_per_iter'])
        for k_rew in K_train_reward_loop:
            # 2. Sample demonstration batch D^_{demo} \subset D_{demo}
            demo_batch = self.agent.sample_rollouts(self.params['train_demo_batch_size'], demo=True)
            # 3. Sample background batch D^_{samp} \subset D_{sample}
            sample_batch = self.agent.sample_recent_rollouts(self.params['train_sample_batch_size'])

            # reshape rollouts' elements to match the dimension in Replay buffer
            for num_rollout, _ in enumerate(demo_batch):
                demo_batch[num_rollout]["log_prob"] = demo_batch[num_rollout]["log_prob"].reshape(-1, 1)
                demo_batch[num_rollout]["reward"] = demo_batch[num_rollout]["reward"].reshape(-1, 1)

            # 4. Append \hat{D}_demo and \hat{D}_samp to background
            self.agent.add_to_buffer(demo_batch, background=True)
            self.agent.add_to_buffer(sample_batch, background=True)

            # use all samples from background batch
            background_batch = self.agent.sample_background_rollouts(all_rollouts=True)

            # 5&6. Estimate gradient loss and update parameters
            reward_log = self.agent.train_reward(demo_batch, background_batch)
            reward_logs.append(reward_log)

            # clear background batch
            self.agent.background_buffer.flush()

        return reward_logs

    ############################################################################################
    def train_policy(self) -> List[Sequence[Dict[str, np.ndarray]]]:
        """
        Guided policy search or Policy Gradient
        """
        print('\nTraining agent using sampled data from replay buffer...')
        train_policy_logs = []

        K_train_policy_loop = range(self.params['num_policy_train_steps_per_iter'])
        for k in K_train_policy_loop:
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['train_batch_size'],
                demo=False
            )
            policy_loss: dict = self.agent.train_policy(ob_batch, ac_batch, re_batch,
                                                        next_ob_batch, terminal_batch)
            train_policy_logs.append(policy_loss)

        return train_policy_logs

    ############################################################################################

    def perform_logging(self, itr: int, train_paths: List[PathDict],
                        eval_policy: BasePolicy,
                        train_video_paths: List[PathDict],
                        reward_logs: list, policy_logs: list,
                        verbose=True) -> None:
        """
        Log metrics and Record Video
        :param itr:
        :param train_paths: training trajs or step
        :param eval_policy: policy used to evaluate
        :param train_video_paths: trajs to record as video
        :param reward_logs: list of train logs  containing training_reward_loss
        :param policy_logs: list of policy logs containing training_policy_loss
        :param verbose: show metrics
        """

        last_reward_log = reward_logs[-1]
        last_policy_log = policy_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        # with torch.no_grad():
        eval_returns = utils.sample_trajectories(self.env,
                                                 eval_policy, self.agent,
                                                 min_timesteps_per_batch=self.params['eval_batch_size'],
                                                 max_path_length=self.params['ep_len'],
                                                 evaluate=True,
                                                 )
        eval_paths, eval_envsteps_this_batch = eval_returns

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths is not None:
            print('\nCollecting video rollouts eval')
            # with torch.no_grad():
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, self.agent,
                                                           ntrajs=MAX_NVIDEO,
                                                           max_path_length=MAX_VIDEO_LEN,
                                                           render=True, expert=False
                                                           )
            # save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')
        ##############################################################################################################

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in train_paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in train_paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            '''Here eval are viewing True Reward, train viewing MLP Reward'''
            eval_logs = OrderedDict()
            eval_logs["Eval_AverageReturn"] = np.mean(eval_returns)
            eval_logs["Eval_StdReturn"] = np.std(eval_returns)
            eval_logs["Eval_MaxReturn"] = np.max(eval_returns)
            eval_logs["Eval_MinReturn"] = np.min(eval_returns)
            eval_logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            training_logs = OrderedDict()
            training_logs["Train_AverageReturn"] = np.mean(train_returns)
            training_logs["Train_StdReturn"] = np.std(train_returns)
            training_logs["Train_MaxReturn"] = np.max(train_returns)
            training_logs["Train_MinReturn"] = np.min(train_returns)
            training_logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            training_logs["Train_EnvstepsSoFar"] = self.total_envsteps
            training_logs["TimeSinceStart"] = time.time() - self.start_time
            training_logs.update(last_policy_log)
            training_logs.update(last_reward_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            training_logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging

            print("\n---------------------------------------------------")
            '''eval_logs'''
            for key, value in eval_logs.items():
                if verbose:
                    if isinstance(value, str):
                        print(f'|\t{key:<20} | {value:>10} |')
                    else:
                        print(f'|\t{key:<20} | {value:>10.3f} |')
                self.logger.log_scalar(value, key, itr)
            '''training_logs'''
            print("|------------------------|")
            for key, value in training_logs.items():
                if verbose:
                    if isinstance(value, str):
                        print(f'|\t{key:<20} | {value:>10} |')
                    else:
                        print(f'|\t{key:<20} | {value:>10.3f} |')
                self.logger.log_scalar(value, key, itr)
            print("---------------------------------------------------")

            self.logger.flush()

    ##########################################################################
    def buffer_status(self, demo=False, samp=False, background=False) -> None:
        """ Show length and size of buffers"""
        assert any([demo, samp, background])
        if demo:
            demo_paths_len = len(self.agent.demo_buffer)
            demo_data_len = self.agent.demo_buffer.num_data
            print(f"{'Demo_buffer_size:': <20} {demo_paths_len}, {demo_data_len}"
                  f"\t{'-> Average Demo ep_len:': ^25} {demo_data_len / demo_paths_len:>10.2f}")
        if samp:
            samp_paths_len = len(self.agent.sample_buffer)
            samp_data_len = self.agent.sample_buffer.num_data
            samp_new_paths_len = self.agent.sample_buffer.new_path_len
            samp_new_data_len = self.agent.sample_buffer.new_data_len
            print(f"{'Sample_buffer_size:': <20} {samp_data_len}, {samp_paths_len}, "
                  f"\t-> Average Samp_new_rollouts_ep_len: {samp_new_data_len / samp_new_paths_len :>10.2f}")
        if background:
            back_paths_len = len(self.agent.background_buffer)
            back_data_len = self.agent.background_buffer.num_data
            if back_paths_len == back_data_len == 0:
                print(f"Back_buffer_size: {len(self.agent.background_buffer)}, {self.agent.background_buffer.num_data}")
            else:
                print(f"Back_buffer_size: {self.agent.background_buffer.num_data} / {len(self.agent.background_buffer)}"
                      f"\tAverage ep_len: {back_data_len / back_paths_len :.2f} ")
        print("##########################################################################")

    ##########################################################################
    def show_logs(self, itr: int, total_envsteps: int,
                  train_paths: List[PathDict],
                  reward_logs: list, policy_logs: list,
                  verbose=True, logging=False
                  ) -> None:
        # last_reward_log = reward_logs[-1]
        last_policy_log = policy_logs[-1]

        # episode lengths, for logging
        train_returns = [path["reward"].sum() for path in train_paths]
        train_ep_lens = [len(path["reward"]) for path in train_paths]

        training_logs = OrderedDict()
        training_logs["Train_AverageReturn"] = np.mean(train_returns)
        training_logs["Train_StdReturn"] = np.std(train_returns)
        training_logs["Train_MaxReturn"] = np.max(train_returns)
        training_logs["Train_MinReturn"] = np.min(train_returns)
        training_logs["Train_AverageEpLen"] = np.mean(train_ep_lens)
        training_logs["Train_EnvstepsSoFar"] = total_envsteps
        training_logs.update(last_policy_log)
        # training_logs.update(last_reward_log)
        '''training_logs'''
        print("|------------------------|")
        for key, value in training_logs.items():
            if verbose:
                if isinstance(value, str):
                    print(f'|\t{key:<20} | {value:>10} |')
                else:
                    print(f'|\t{key:<20} | {value:>10.3f} |')
            if logging:
                pass
                # self.logger.log_scalar(value, key, itr)
        print("---------------------------------------------------")
        self.logger.flush()

    def perform_pg2opt(self):
        print('\nTraining agent using sampled data from replay buffer...')
        train_policy_logs = []

        K_test_policy_loop = range(self.params['num_policy_train_steps_per_iter'])
        for k in K_test_policy_loop:
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.test_agent.test_buffer. \
                sample_recent_data(self.params['train_batch_size'], concat_rew=False)

            policy_loss: dict = self.test_agent.train_policy(ob_batch, ac_batch, re_batch,
                                                             next_ob_batch, terminal_batch)
            train_policy_logs.append(policy_loss)
        return train_policy_logs
