import pickle
import time
from collections import OrderedDict

import gym
import numpy as np
import pytorch_util as ptu
import torch
import utils
from logger import Logger
from tqdm import tqdm


# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class GCL_Trainer():
    """ GCL_Trainer """

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

        # init gpu
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        # ENV
        #############

        # Make the gym environment
        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.env.max_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Are the observations images?
        is_img = len(self.env.observation_space.shape) > 2

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape if is_img else self.env.observation_space.shape[0]
        # assume continuous action space
        ac_dim = self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        #        if 'model' in dir(self.env):
        #            self.fps = 1/self.env.model.opt.timestep
        #        elif 'env_wrappers' in self.params:
        #            self.fps = 30 # This is not actually used when using the Monitor wrapper
        #        elif 'video.frames_per_second' in self.env.env.metadata.keys():
        #            self.fps = self.env.env.metadata['video.frames_per_second']
        #        else:
        #            self.fps = 10
        self.fps = 10

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          expert_data=None, expert_policy=None):
        """
        :param n_iter: number of iterations
        :param collect_policy: q_k
        :param expert_data: D_demo
        :param expert_policy:
        """
        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        '''D_demo: self.agent.demo_buffer.add_rollouts(paths)
           D_samp: self.agent.sample_buffer.add_rollouts(paths)
        '''
        # add demonstrations to replay buffer
        demo_paths = self.collect_demo_trajectories(expert_data, expert_policy)
        self.agent.add_to_buffer(demo_paths, demo=True)

        train_log_lst, policy_log_lst = [], []

        # 2.
        n_iter_loop = tqdm(range(n_iter), desc="Guided cost learning", leave=False)
        for itr in n_iter_loop:
            print("\n")
            print("********** Iteration {} ************".format(itr))
            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # 3. Generate samples D_traj from current trajectory distribution q_k (collect_policy)
            paths, envsteps_this_batch, train_video_paths = self.collect_training_trajectories(
                collect_policy, self.params['batch_size']
            )
            self.total_envsteps += envsteps_this_batch

            # 4. Append samples D_traj to D_samp
            self.agent.add_to_buffer(paths)

            # 5. Use D_{samp} to update cost c_{\theta}
            reward_logs = self.train_reward()  # Algorithm 2

            # 6. Update q_k(\tau) using D_{traj} and using GPS or PG
            policy_logs = self.train_policy()

            # log/save
            if self.log_video or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                # self.perform_logging(itr, paths, eval_policy, train_video_paths, reward_logs, policy_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

            for i, j in zip(reward_logs, policy_logs):
                reward_loss = float(i['Training reward loss'])
                train_log_lst.append(reward_loss)
                policy_loss = float(j['Training Loss'])
                policy_log_lst.append(policy_loss)

            # update progress bar
            # n_iter_loop.set_postfix(train_log=train_log_mean,
            #                         policy_log=policy_log_mean,
            #                         w=float(self.agent.reward.w),
            #                         )

        return train_log_lst, policy_log_lst

    def collect_demo_trajectories(self, expert_data, expert_policy):
        """
        :param expert_data:  relative path to saved 
        :param expert_policy:  relative path to saved expert policy
        :return:
            paths: a list of trajectories with len = self.params['demo_size']
                    each trajectory is a dict {obs, image_obs, acs, log_probs, rewards, next_obs, terminals}
        """
        # Load expert policy or expert demonstrations D_demo
        if expert_data:
            # if expert_data != ''
            print('\nLoading saved demonstrations...')

            with open(expert_data, 'rb') as f:
                # TODO: load data may not through pickle
                demo_paths = pickle.load(f)
            # TODO: sample self.params['demo_size'] from demo_paths -- implemented
            return demo_paths[: self.params['demo_size']]

        elif expert_policy:
            # TODO: make this to accept other expert policies
            # TODO: two kind of policy (multiprocess and single process ) -- implemented
            '''use a dict with key = policy_name and value = policy_class or an indicator
                then import based on the indicator
                Do 3 example first, train and save the parameter
            '''
            from stable_baselines3 import PPO
            expert_policy = PPO.load(expert_policy)
            print('\nRunning expert policy to collect demonstrations...')
            demo_paths, _ = utils.sample_trajectories(
                self.env,
                expert_policy,
                agent=self.agent,
                batch_size=self.params['demo_size'],
                expert=True
            )
        else:
            raise ValueError('Please provide either expert demonstrations or expert policy')
        return demo_paths

    def collect_training_trajectories(self, collect_policy, batch_size):
        """
        :param collect_policy:  the current policy which we use to collect data
        :param batch_size:  the number of trajectories to collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        print("\nCollecting sample trajectories to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, batch_size,
                                                               agent=self.agent)

        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            # TODO look in utils and implement sample_n_trajectories -- implemented
            pass

        # TODO: add logging
        if self.logmetrics:
            # TODO:# what should be log in this function
            pass

        return paths, envsteps_this_batch, train_video_paths

    def train_reward(self):
        """
        Algorithm 2: Nonlinear IOC with stochastic gradients 
        """
        print("\nUpdating reward parameters...")
        reward_logs = []
        K_train_reward_loop = tqdm(range(self.params['num_reward_train_steps_per_iter']),
                                   desc="reward_update",
                                   leave=False)
        for k_rew in K_train_reward_loop:
            # Sample demonstration batch D^_{demo} \subset D_{demo}
            demo_batch = self.agent.sample_rollouts(self.params['train_demo_batch_size'], demo=True)
            # Sample background batch D^_{samp} \subset D_{sample}
            sample_batch = self.agent.sample_rollouts(self.params['train_sample_batch_size'])

            # Use the sampled data to train the reward function
            # Estimate dL_{ioc}/dθ (θ) using batch D^_{demo} and D_{demo}
            # Update parameters θ using gradient dL_{ioc}/dθ (θ)
            reward_log = self.agent.train_reward(demo_batch, sample_batch)
            reward_logs.append(reward_log)

            K_train_reward_loop.set_postfix(K_rew=k_rew,
                                            reward_loss=reward_log["Training reward loss"],
                                            w=self.agent.reward.w.item())
        return reward_logs

    
    def train_policy(self):
        """
        Guided policy search or PG
        """
        print('\nTraining agent using sampled data from replay buffer...')
        train_policy_logs = []
        K_train_policy_loop = tqdm(range(self.params['num_policy_train_steps_per_iter']),
                                   desc="policy_update",
                                   leave=False)
        for k_ply in K_train_policy_loop:
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['train_batch_size'])
            policy_loss = self.agent.train_policy(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            train_policy_logs.append(policy_loss)
            K_train_policy_loop.set_postfix(K_ply=k_ply,
                                            policy_loss=policy_loss["Training Loss"],
                                            )
        return train_policy_logs

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, reward_logs, policy_logs):

        last_log = policy_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, _ = utils.sample_trajectories(
            self.env, eval_policy,
            self.params['eval_batch_size'],
            agent=self.agent,
            render=True
        )

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths is not None:
            eval_video_paths, _ = utils.sample_trajectories(self.env, eval_policy, self.agent, MAX_NVIDEO, render=True)

            # save train/eval videos
            print('\nSaving train and eval rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')

        #######################

        # save eval metrics
        # TODO: should parse the reward training loss and policy training loss
        # TODO: should add a visualization tool to check the trained reward function
        # Path(obs, image_obs, acs, log_probs, rewards, next_obs, terminals)
        if self.logmetrics:
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
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

