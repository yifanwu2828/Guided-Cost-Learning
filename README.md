# Model-based Inverse-Reinforcement-Learning


## Implementation of [Guided Cost Learning (GCL)](https://arxiv.org/pdf/1603.00448.pdf)
![GitHub repo size](https://img.shields.io/github/repo-size/yifanwu2828/Inverse-Reinforcement-Learning)
![GitHub contributors](https://img.shields.io/github/contributors/yifanwu2828/Inverse-Reinforcement-Learning)
![GitHub last commit](https://img.shields.io/github/last-commit/yifanwu2828/Inverse-Reinforcement-Learning)
## TODO
- [x] Set up 2D navigation environment
- [x] Add rgb_array render mode in 2D navigation environment
- [x] Set up RL agent (PPO, SAC) as expert policy
- [x] Implement GCL
- [x] Add visualization for learned reward during training
- [x] Check tensorboard video logger for visualizing reward map
- [x] Check expert policy and ensure it doesn't pass high-cost regions
- [x] Clear TODOs in code 

## Installation
1. Install Stable baselines 3
```bash
pip install stable-baselines3
```
2. Install gym_nav environment and gcl
```bash
pip install -e .
```

## Usage
1. Train an RL agent as the expert
```bash
cd scripts
python3 collect_demo.py
```
2. Run GCL
```bash
python3 run_exp.py
```
