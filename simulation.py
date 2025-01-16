class Simulation:
    def __init__(self, env, agents, n_calls):
        self.env = env
        self.agents = agents
        self.n_calls = n_calls

    def run(self):
        for _ in range(self.n_calls):
            for agent in self.agents:
                obs = np.zeros(self.env.embedding_dim)
                action = agent.take_action(obs)
                self.env.step(action)
