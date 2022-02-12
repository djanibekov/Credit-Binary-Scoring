import pickle
import numpy as np
import config


class MakePrediction:
    def __init__(self):
        self.loaded_model = pickle.load(open(fr'{config.PROJECT_DIR}\finalized_model.sav', 'rb'))

    def predict(self, X):
        return self.loaded_model.predict(np.array(X).reshape(-1, 1).T)

    def predict_score(self, X, y):
        return self.loaded_model.score(X, y)
