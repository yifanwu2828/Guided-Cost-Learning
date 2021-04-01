import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from matplotlib import pyplot as plt


class NavEnv(gym.Env):
    """
    2D continuous box environment for navigation
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, size=1, seed=1337, multiplayer=1):

        # the box size is (2*size, 2*size)
        self.size = size
        self.max_steps = 100
        self.action_dim = 2
        self.pos_dim = 2
        self.vel_dim = 2
        self.obs_dim = 64
        self.dt = 1e-1

        # States are 2D position
        # Actions are 2D velocity
        self.action_space = spaces.Box(
            low=-self.size,
            high=self.size,
            shape=(self.action_dim,),
            dtype="float32"
        )

        # Observations are the position and velocity
        self.observation_space = spaces.Box(
            low=-self.size,
            high=self.size,
            shape=(self.pos_dim + self.vel_dim,),
            dtype="float32"
        )

        # Window to use for human rendering mode
        self.window = None

        # Mission of the task
        self.mission: str = 'Navigate to the origin'

        # Initialize seed
        self.seed(seed=seed)

        # Initialize reward map
        self.init_reward()

        # Initialize the state
        self.reset()

    def step(self, action):

        self.step_count += 1
        self.update_states(action)

        # Observations are the position and velocity
        obs = np.concatenate((self.pos, self.vel))
        reward = self.eval_reward(self.pos)
        self.pos_list.append(self.pos.copy())

        done = False
        done_cond = abs(self.pos[0]) <= 1e-2 and abs(self.pos[1]) <= 1e-2
        terminate_cond = (self.step_count >= self.max_steps)
        if done_cond or terminate_cond:
            done = True

        return obs, reward, done, {}

    def preprocess_obs(self, agent_pos):
        """
        Add the agent position to the reward map
        """
        obs = self.reward_map.copy()
        agent_idx = ((agent_pos + self.size) / (2 * self.size) * (self.obs_dim - 1)).astype(int)

        # Set the agent to be red
        obs[agent_idx[0], agent_idx[1]] = np.array([255, 0, 0])
        return obs

    def update_states(self, action):
        """
        Update the position and velocity with a simple dynamics model
        x = x + x' * dt, x' = x' + u * dt
        Force the position and velocity within bounds
        """
        self.pos += self.vel * self.dt
        self.vel += action * self.dt
        self.pos[self.pos > self.size] = self.size
        self.pos[self.pos < -self.size] = -self.size
        self.vel[self.vel > self.size] = self.size
        self.vel[self.vel < -self.size] = -self.size

    def reset(self):
        """
        Randomly spawn a starting location
        """
        self.pos = np.random.uniform(
            low=-self.size,
            high=self.size,
            size=self.pos_dim
        )
        #        self.vel = np.random.uniform(
        #            low=-self.size,
        #            high=self.size,
        #            size=self.vel_dim
        #        )
        self.vel = np.zeros(self.vel_dim)

        # Keep track of the position in the current episode
        self.pos_list = [self.pos.copy()]

        # Step count since episode start
        self.step_count = 0

        #        # Observation is the reward map with the agent position
        #        obs = self.preprocess_obs(self.pos)
        # Observations are the position and velocity
        obs = np.concatenate((self.pos, self.vel))

        return obs

    def render(self, mode='human'):
        """
        Visualize the trajectory on the reward map
        """

        if mode == 'human' and not self.window:
            import gym_nav.window
            self.window = gym_nav.window.Window('gym_nav')
            self.window.show(block=False)

        img = self.preprocess_obs(self.pos)

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        elif mode == 'rgb_array':
            return img

    def close(self):
        if self.window:
            self.window.close()
        return

    def init_reward(self):
        """
        The reward is a mixture of Gaussians functions
        Highest at the origin and lowest at the four corners
        """

        self.mixtures = {
            'mu': [np.zeros(self.pos_dim),
                   np.array([self.size / 3, self.size / 3]),
                   np.array([self.size / 3, -self.size / 3]),
                   np.array([-self.size / 3, self.size / 3]),
                   np.array([-self.size / 3, -self.size / 3])],
            'std': [self.size,
                    self.size / 6,
                    self.size / 6,
                    self.size / 6,
                    self.size / 6],
            'A': [2, -1, -1, -1, -1]
        }

        # Increase obs_dim for higher resolution
        num = self.obs_dim
        x = np.linspace(-self.size, self.size, num=num)
        y = np.linspace(-self.size, self.size, num=num)
        # TODO: vectorized this computation
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                Z[i, j] = self.eval_gaussian(np.array([X[i, j], Y[i, j]]))
        # rescale values to [0, 255]
        self.reward_min, self.reward_max = np.min(Z), np.max(Z)
        Z = ((Z - self.reward_min) / (self.reward_max - self.reward_min) * 255).astype(np.uint8)
        # convert grayscale to rgb by stacking three channels
        self.reward_map = np.stack((Z, Z, Z), axis=-1)

    def eval_gaussian(self, x):
        """
        Evaluate the value of mixture of gaussian functions at location x
        """
        ans = 0
        for mu, std, A in zip(self.mixtures['mu'], self.mixtures['std'], self.mixtures['A']):
            ans += A * np.exp(-(x - mu).T @ (x - mu) / (2 * std ** 2))
        return ans

    def eval_reward(self, x):
        """
        Evaluate the reward at the current position x
        """
        reward = self.eval_gaussian(x)

        # shape reward to [-1, 1] to assist learning
        # reward = (reward - self.reward_min) / (self.reward_max - self.reward_min) * 2 - 1
        # shape reward to [-1, 0] to assist learning
        reward_std = (reward - self.reward_min) / (self.reward_max - self.reward_min)
        min_val, max_val = (-1, 0)
        scale_reward = reward_std * (max_val - min_val) + min_val
        return scale_reward
