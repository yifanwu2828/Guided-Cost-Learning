# Implementation of [Guided Cost Learning (GCL)](https://arxiv.org/pdf/1603.00448.pdf)
![GitHub repo size](https://img.shields.io/github/repo-size/yifanwu2828/Inverse-Reinforcement-Learning)
![GitHub contributors](https://img.shields.io/github/contributors/yifanwu2828/Inverse-Reinforcement-Learning)
![GitHub last commit](https://img.shields.io/github/last-commit/yifanwu2828/Inverse-Reinforcement-Learning)

## Installation
1. Install 3rd party packages
```bash
pip install -r requirements.txt
```
*Note you have to install mujoco for FetchReach-v1 Env

2. Install gym_nav environment and gcl
```bash
pip install -e .
```

## Usage
1. Train an RL agent as the expert
```bash
cd scripts
python3 collect_demo.py  # For 2D-Nav Env
python3 collect_fetch_demo.py  # For FetchReach-v1 Env 
```
2. Run GCL
```bash
cd scripts
python3 run_gcl_nav.py
python3 run_gcl_Fetch.py
```
