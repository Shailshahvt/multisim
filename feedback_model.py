import os
from tensorflow.keras import models, layers

class FeedbackModel:
    def __init__(self, model_path, embedding_dim=384):
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.model = self._load_or_build_model()

    def _load_or_build_model(self):
        if os.path.exists(self.model_path):
            return models.load_model(self.model_path)
        return self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.embedding_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def score(self, embedding):
        return self.model.predict(embedding.reshape(1, -1))[0][0]
