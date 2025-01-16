class Agent:
    def __init__(self, id, rl_model, feedback_model, llm_model=None):
        self.id = id
        self.rl_model = rl_model
        self.feedback_model = feedback_model
        self.llm_model = llm_model
        self.embeddings = []
        self.scores = []

    def take_action(self, obs, use_llm=False):
        if use_llm and self.llm_model:
            return self.llm_model.generate_suggestion(obs)
        action, _ = self.rl_model.predict(obs)
        return action
