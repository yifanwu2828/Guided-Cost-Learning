import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



parser = argparse.ArgumentParser(description="REINFORCE")
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




class Policy(nn.Module):
    def __init__(self, num_inputs, num_actions,
                 hidden_size=128, learning_rate = 3e-4):
        super(Policy,self).__init__()
        self.num_actions = num_actions      
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.dropout = nn.Dropout(p=0.8)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self,x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear2(x)

        return F.softmax(x,dim=1)



    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

def finish_episode(policy,):
    reward = 0
    policy_loss = []
    returns = []
    # reverse order
    for r in policy.rewards[::-1]:
        reward = r + args.gamma * reward
        returns.insert(0, reward)
    #returns = torch.tensor(returns-np.mean(returns))
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, reward in zip(policy.saved_log_probs,returns):
        policy_loss.append(-log_prob * (reward))
    
    policy.optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    policy.optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    
    env = gym.make('CartPole-v1')
    print("Dimension of obs_space",env.observation_space.shape[0])
    print("Dimension of act_space",env.action_space.n)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    lr=args.lr
    policy = Policy(env.observation_space.shape[0], env.action_space.n,
                    hidden_size=128, learning_rate= lr)

    running_reward = 10
    max_episode = 3000
    max_step = 10000
    num_step=[]

    for i_episode in range(max_episode):
        ep_reward = 0
        state = env.reset()
        for t in range(max_step):
            action =policy.get_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward

            if done:
                num_step.append(t)
                break
        
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(policy)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
    plt.plot(num_step)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.savefig('REINFORCE1.png')

if  __name__ == "__main__":
    eps = np.finfo(np.float32).eps.item()
    main()