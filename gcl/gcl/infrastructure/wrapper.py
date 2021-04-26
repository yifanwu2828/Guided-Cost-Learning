import numpy as np
import gym


class FixGoal(gym.Wrapper):
    def __init__(self, env, pos=(1.3040752, 0.74440193, 0.66095406)):
        super().__init__(env)
        self.env = env
        assert len(pos) == 3
        if not isinstance(pos, np.ndarray):
            pos = np.array(pos, dtype=np.float32)
        self.pos = pos

    def step(self, action):
        observation, _, done, info = self.env.step(action)

        achieved_goal = observation[3:6]
        reward = self.compute_reward(achieved_goal, self.env.goal)

        return observation, reward, done, info

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, goal, info=None):
        d = self.goal_distance(achieved_goal, goal)
        if self.env.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d


    def reset(self):
        obs = self.env.reset()
        self.env.goal[0] = self.pos[0]
        self.env.goal[1] = self.pos[1]
        self.env.goal[2] = self.pos[2]

        # this one do not work
        # self.env.goal = self.pos
        obs[0:3] = self.env.goal.copy()
        return obs