import json
from core.agent import Agent
from core.feedback_model import FeedbackModel
from core.search_environment import SearchEnv
from core.simulation import Simulation

def main():
    with open('config/default_config.json') as f:
        config = json.load(f)

    feedback_model = FeedbackModel(config['models']['feedback_model_path'])
    env = SearchEnv(**config['simulation'], feedback_model=feedback_model)
    agents = [Agent(i, None, feedback_model) for i in range(config['simulation']['n_agents'])]
    simulation = Simulation(env, agents, config['simulation']['n_calls'])
    simulation.run()

if __name__ == "__main__":
    main()
