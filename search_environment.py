import numpy as np
from gym import Env, spaces

class SearchEnv(Env):
    def __init__(self, n_agents, embedding_dim, discount_factor, bound_min, bound_max, feedback_model):
        super().__init__()
        self.n_agents = n_agents
        self.embedding_dim = embedding_dim
        self.discount_factor = discount_factor
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.feedback_model = feedback_model
        self.action_space = spaces.Box(low=self.bound_min, high=self.bound_max, shape=(self.embedding_dim,))
        self.tried_embeddings = [[] for _ in range(n_agents)]
        self.tried_scores = [[] for _ in range(n_agents)]

    def step(self, action):
        reward = self.feedback_model.score(action)
        self.tried_embeddings[0].append(action)
        self.tried_scores[0].append(reward)
        return np.zeros(self.embedding_dim), reward, False, {}
