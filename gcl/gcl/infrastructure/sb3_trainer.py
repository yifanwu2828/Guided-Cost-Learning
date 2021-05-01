import time
import re
import pickle
import itertools as it
from collections import OrderedDict
from functools import lru_cache
from typing import List, Dict, Tuple, Sequence, Optional, Any

import numpy as np
import torch

from tqdm import tqdm
from icecream import ic

import gym
from gym.wrappers import FilterObservation, FlattenObservation

from stable_baselines3 import A2C, SAC, PPO, HER
from stable_baselines3.common.monitor import Monitor

from gcl.infrastructure import pytorch_util as ptu
from gcl.infrastructure import utils
from gcl.infrastructure.utils import PathDict
from gcl.infrastructure.wrapper import FixGoal, LearningReward


WRAPPER = {
    '': None,
    'filter_obs': FilterObservation,
    'flatten_obs': FlattenObservation,
    'fix_goal': ['filter_obs', 'flatten_obs', FixGoal],
    'mlp_rew': LearningReward,
}


class GCL_Trainer(object):
    """ GCL_Trainer"""

    def __init__(self, params):

        #############
        # INIT
        #############

        # Get params
        self.params = params

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
        # self.env.reward_type = 'dense'
        self.env.reward_type = self.params['rewardType']

        self.env.seed(seed)

        # Get ENV wrapper
        self.wrapper = self.params.get('EnvWrapper', '')
        if self.wrapper != '':
            self.wrapper = WRAPPER[self.wrapper]
            if isinstance(self.wrapper, list):
                for wrapper in self.wrapper:
                    # apply wrapper wrt WRAPPER Dict
                    if isinstance(wrapper, str):
                        if wrapper == 'filter_obs':
                            self.env = WRAPPER[wrapper](self.env, ['observation', 'desired_goal'])
                        else:
                            self.env = WRAPPER[wrapper](self.env)
                    else:
                        self.env = wrapper(self.env)
            else:
                self.env = self.wrapper(self.env)
        if 'Fetch' in self.params['env_name']:
            self.env = Monitor(self.env, info_keywords=("is_success",))
        else:
            self.env = Monitor(self.env)
        # Maximum length for episodes
        self.params['ep_len'] = params.get('ep_len', None)
        if self.params['ep_len'] is None:
            print("Episodes length is not specified, Using env.max_episode_steps")
            try:
                self.params['ep_len']: int = self.env.spec.max_episode_steps
            except AttributeError:
                self.params['ep_len']: int = self.env._max_episode_steps  # Access to a protected member

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

        print()
        ic("--------- GCL_Trainer ---------")
        ic(self.wrapper)
        ic(self.env)
        ic(self.env.reward_type)
        ic(self.params['ep_len'])

        #############
        # AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])
        self.samp_recent = self.params['samp_recent']

    ##################################

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    ##################################
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
                                                              render=False,
                                                              verbose=True)
        self.agent.add_to_buffer(demo_paths, demo=True)
        print(f'\nNum of Demo rollouts collected: {self.agent.demo_buffer.num_paths}')
        print(f'Num of Demo transition steps collected: {self.agent.demo_buffer.num_data}')
        utils.toc(self.start_time, "Finish Loading Expert Demonstrations", ftime=True)

        #####################################################################
        # 2.
        n_iter_loop = tqdm(range(n_iter), desc="Guided Cost Learning", leave=False)
        for itr in n_iter_loop:
            print(f"\n********** Iteration {itr} ************")
            # set reward mode to train
            self.agent.reward.train()

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
            with torch.no_grad():
                training_returns = self.collect_training_trajectories(
                    collect_policy=collect_policy,
                    batch_size=self.params["train_reward_sample_batch_size"]
                )

            samp_paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            del training_returns

            # 4. Append samples D_traj to D_samp
            self.agent.add_to_buffer(samp_paths)
            # show status
            self.buffer_status(demo=True, samp=True)

            # 5. Use D_{samp} to update cost c_{\theta}
            reward_logs = self.train_reward(recent=self.samp_recent)  # Algorithm 2

            # 6. Update q_k(\tau) using D_{traj} and using GPS or PG
            policy_logs = self.train_policy()

            # log
            print('\nBeginning logging procedure...')
            self.show_logs(
                itr,
                self.total_envsteps,
                samp_paths,
                reward_logs, policy_logs,
                logging=True
            )

            for r, p in it.zip_longest(reward_logs, policy_logs):
                if r:
                    reward_loss = float(r['Training_Reward_Loss'])
                    train_log_lst.append(reward_loss)
                if p:
                    policy_loss = float(p["Training_Policy_Loss"])
                    policy_log_lst.append(policy_loss)

            save_itr = [50, 77, 99, 125-1, 150-1, 200-1, 250-1, 300, 350, 400]
            if itr in save_itr:
                # Torch
                fname1 = f"../model/test_sb3_reward_{self.params['agent_params']['model_class']}_{itr}.pth"
                reward_model = self.agent.reward
                torch.save(reward_model, fname1)
                # SB3
                fname2 = f"../model/test_sb3_policy_{self.params['agent_params']['model_class']}_{itr}"
                policy_model = self.agent.actor
                policy_model.save(fname2)

            # update progress bar
            n_iter_loop.set_postfix()

        return train_log_lst, policy_log_lst

    ############################################################################

    @lru_cache(maxsize=10)
    def collect_demo_trajectories(
            self,
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

            fname = re.split('/', expert_policy)[-1]
            assert isinstance(expert_policy, str)
            if fname.startswith('ppo'):
                demo_model = PPO.load(path=expert_policy, env=None)
            elif fname.startswith('a2c'):
                demo_model = A2C.load(path=expert_policy, env=None)
            elif fname.startswith('sac'):
                demo_model = SAC.load(path=expert_policy, env=None)
            elif fname.startswith('her'):
                demo_model = HER.load(path=expert_policy, env=self.env)
            else:
                raise ValueError('Please provide valid expert policy')

            expert_policy_model = demo_model

            print('\nRunning expert policy to collect demonstrations...')

            demo_paths = utils.sample_n_trajectories(
                self.env,
                policy=expert_policy_model,
                agent=self.agent,
                ntrajs=self.params['demo_size'],
                max_path_length=self.params['ep_len'],
                render=render,
                render_mode=render_mode,
                expert=True,
                deterministic=True,
            )
        else:
            raise ValueError('Please provide either expert demonstrations or expert policy')
        return demo_paths, envsteps_this_batch, demo_video_paths

    ############################################################################################

    def collect_training_trajectories(
            self,
            collect_policy,
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
        envsteps_this_batch: Optional[int] = 0
        train_video_paths: Optional[List[PathDict]] = None

        print(f"\nCollecting {batch_size} sample trajectories to be used for training...")
        paths = utils.sample_n_trajectories(
            env=self.env,
            policy=collect_policy,
            agent=self.agent,
            ntrajs=batch_size,
            max_path_length=self.params['ep_len'],
            device=ptu.device,
            deterministic=False,
            sb3=True,
        )

        return paths, envsteps_this_batch, train_video_paths

    ############################################################################################
    def train_reward(self, recent=False) -> List[Dict]:
        """
        Algorithm 2: Nonlinear IOC with stochastic gradients
        """
        print("\nUpdating reward parameters...")

        if recent:
            print('\n Sampling recent rollouts from sample Replay buffer')

        reward_logs = []
        # 1.
        K_train_reward_loop = range(self.params['num_reward_train_steps_per_iter'])
        for _ in K_train_reward_loop:
            # 2. Sample demonstration batch D^_{demo} \subset D_{demo}
            demo_batch = self.agent.sample_rollouts(self.params['train_reward_demo_batch_size'], demo=True)
            # 3. Sample background batch D^_{samp} \subset D_{sample}
            if not recent:
                # random sampling
                sample_batch = self.agent.sample_rollouts(self.params['train_reward_sample_batch_size'])
            else:
                # sample recent data
                sample_batch = self.agent.sample_recent_rollouts(self.params['train_reward_sample_batch_size'])

            # reshape rollouts' elements to match the dimension in Replay buffer
            for num_rollout, _ in enumerate(demo_batch):
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
        # Freeze learning reward model
        self.agent.reward.eval()

        # training policy with specified RL algo
        print('\nTraining agent using sampled data from replay buffer...')
        train_policy_logs = []
        # total_timesteps=self.params["train_batch_size"]
        total_timesteps= 10_000
        self.agent.train_policy(total_timesteps=total_timesteps)

        # Unfreeze learning reward model
        self.agent.reward.train()
        return train_policy_logs

    ##########################################################################
    def buffer_status(self, demo=False, samp=False, background=False) -> None:
        """ Show length and size of buffers"""
        assert any([demo, samp, background])
        if demo:
            demo_paths_len = len(self.agent.demo_buffer)
            demo_data_len = self.agent.demo_buffer.num_data
            print(f"{'Demo_buffer_size:': <20} {demo_data_len}, {demo_paths_len}"
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
        last_reward_log = reward_logs[-1]

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
        training_logs.update(last_reward_log)
        '''training_logs'''
        print("|----------------------------|")
        for key, value in training_logs.items():
            if verbose:
                if isinstance(value, str):
                    print(f'|\t{key:<20} | {value:>10} |')
                else:
                    print(f'|\t{key:<20} | {value:>10.3f} |')
            if logging:
                pass
        print("---------------------------------------------------")
