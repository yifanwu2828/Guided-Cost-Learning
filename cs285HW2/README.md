# UCB cs285 HW2

A re-implementation based on UCB cs285 HW2

The goal of this project is to perform policy gradient on a continuous environment through Pybullet

## Policy gradient
Reinforcement learning objective is to learn a optimal policy that maximize the objective function
J(theta) = E_{\tau ~ pi_{theta}(\tau)}[r(tau)]

where pi_{theta} is a probability distribution over the action space, conditioned on the state. In the agent environment loop, the agent samples and action a_t form pi_{theta}(*|s_t) and the environment responds with a reward(s_t,a_t)

### Disceret or Continous Action Space
A discrete flag is used when initiating MLPPolicyPG:

In a discrete action space, a categorical distribution is used to approximate pi_{theta}
if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)


In a continuous action space, the policy is approximated by gaussian N~(mu,sigma) with mean mu and std sigma: 
else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )
            
### Reducing Variance 
One of the disadvantages of policy gradient is that it has a high variance. To reduce variance "reward to go", denoted as rtg and baseline are introduced.

Reward to go inherent from Causality: policy at time t' cannot affect reward at time t when t<t'. Therefore, Q function is the total reward form taking a_t in s_t then follow the policy

subtracting a baseline from the policy gradient is unbiased in expectation. Although setting baseline equals to average reward is not the best baseline, it is good enough in practical. A very good choice of the baseline is  the value funcion, total reward expect to get if start at state s_t and follow the policy

The difference between rtg and baseline as a value function estimates how much better action a_it is than the average action take in s_it. In fact the difference between rtg and value funciton is so popular and it is called advantage function denoted as A^{pi}(s_t,a_t)

In practice, we do not know the true value of advantage. We have to estimate it. The better our estimate of the advantage, the lower our variance will be. The estimation doesn't necessarily produce unbiased estimates of advantage function. It is acceptable that we trade off enormous reduction in variance with slightly increase in bias. 

### Sample Result from performing Policy Gradient with estimated Advantage Function and a Value Function as A Baseline


