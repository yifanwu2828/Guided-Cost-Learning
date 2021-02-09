import argparse

from gcl_trainer import GCL_Trainer
from gcl.agents.gcl_agent import GCL_Agent

class IRL_Trainer():

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'output_size': params['output_size'],
            'learning_rate': params['learning_rate'],
        }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
            'reward_to_go': params['reward_to_go'],
            'nn_baseline': params['nn_baseline'],
        }

        train_args = {
            'num_policy_train_steps_per_iter': params['num_policy_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}
        
        self.params = params
        self.params['agent_class'] = GCL_Agent
        self.params['agent_params'] = agent_params

        ################
        ## IRL TRAINER
        ################

        self.gcl_trainer = GCL_Trainer(self.params)

    def run_training_loop(self):

        self.gcl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy=self.gcl_trainer.agent.actor,
            expert_data=self.params['expert_data'],
            expert_policy=self.params['expert_policy']

        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='NavEnv-v0')
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--expert_policy', type=str, default='ppo_nav_env')               # relative to where you're running this script from
    parser.add_argument('--expert_data', type=str, default='')            # relative to where you're running this script from
    parser.add_argument('--demo_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--train_demo_batch_size', type=int, default=10)
    parser.add_argument('--train_sample_batch_size', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=100)

    parser.add_argument('--discount', type=float, default=1.0)    
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--nn_baseline', action='store_true')
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')

    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--num_reward_train_steps_per_iter', type=int, default=10)
    parser.add_argument('--num_policy_train_steps_per_iter', type=int, default=10)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=20)

    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################
    # TODO




    ###################
    ### RUN TRAINING
    ###################

    trainer = IRL_Trainer(params)
    trainer.run_training_loop()

if __name__ == '__main__':
    main()