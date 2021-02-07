import argparse

from irl_trainer import IRL_Trainer
from agents.irl_agent import IRL_Agent

class IRL_Trainer():

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **train_args}
        
        self.params = params
        self.params['agent_class'] = IRL_Agent
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.irl_trainer = IRL_Trainer(self.params)

    def run_training_loop(self):

        self.irl_trainer.run_training_loop(
            self.params['n_iter']
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)    # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True)            # relative to where you're running this script from

    parser.add_argument('--env_name', type=str, default='NavEnv-v0')
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

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