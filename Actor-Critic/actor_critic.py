import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description="actor-critic")
parser.add_argument('--env', type=str, default='CartPole-v1',
                    help='enviroment (default=CartPole-v1)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default=0.99)')
parser.add_argument('--seed', type=int, default=543,
                    help='random seed (default=543)')
parser.add_argument('--render', action = 'store_true',
                    help='render the enviroment')
parser.add_argument('--lr', type=float, default=1e-2, 
                    help='learning rate (default: 1e-2)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='interval between trainging status logs (default: 10)')
args = parser.parse_args()

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    '''
    implements both actor and critic in ond model
    '''
    def __init__(self, num_inputs, num_actions, 
                 hidden_size=128, learning_rate=3e-2):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # actor's layer
        self.action_head = nn.Linear(hidden_size, num_inputs)
        # critic's layer
        self.value_head = nn.Linear(hidden_size,1)
        #action & reward buffer
        self.saved_actions = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, x):
        '''
        forward of both actor and critic
        '''
        x = F.relu(self.linear1(x))

        # actor: choose action to take from state s_t
        # return probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluate being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t        
        return action_prob, state_values

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        print(state.shape)
        probs, state_value = self.forward(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        print("\n"*5)
        print(action.item())
        print("\n"*5)
        return action.item()
    
def finish_episode(policy,):
    R=0
    saved_actions = policy.saved_actions
    policy_loss = []
    value_loss = []
    returns = []

    #calculate the true value using rewards returned from env
    for r in policy.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0,R)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip (saved_actions, returns):
        advantage = R - value.item()

        # calculate actor(policy) loss
        policy_loss.append(-log_prob * advantage)
        # calculate critic(value) loss
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))
    
    policy.optimizer.zero_grad()
    # sum up all the values of policy_losses and value_losses
    loss = torch.cat(policy_loss).sum + torch.cat(value_loss).sum
    loss.backward()
    policy.optimizer.step()

    #reset rewards and action buffer
    del policy.rewards[:]
    del policy.saved_actions[:]

def main():
    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    continuous = isinstance(env.action_space,gym.spaces.box.Box)

    if not continuous:
        act_dim = env.action_space.n
    else:
        act_dim = env.action_space.shape[0]
    
    print("Dimension of obs_space",obs_dim)
    print("Dimension of act_space",act_dim)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    lr= args.lr
    policy = Policy(obs_dim, act_dim, hidden_size=128, learning_rate=lr)
    
    max_episode = 5000
    max_step = 10000
    all_rewards = []
    all_lengths = []

    running_reward = 10

    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0

        for t in range(1,max_step):
            # select action from policy
            action = policy.get_action(state)
            # take action
            state, reward, done , _ = env.step(action)

            if args.render:
                env.render()
            
            policy.rewards.append(reward)
            ep_reward += reward
            
            if done or t == max_step-1:
                all_rewards.append(ep_reward)
                all_lengths.append(t)
                break

        # update cumulative reward
        running_reward = 0.05* ep_reward + (1-0.05) * running_reward
        finish_episode(policy)
        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
                break
    
    plt.plot(all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('actor-critic10.png')

    plt.plot(all_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.savefig('actor-critic1.png')

if __name__ == "__main__":
    eps = np.finfo(np.float32).eps.item()
 
    main()