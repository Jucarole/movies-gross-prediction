import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import Utils


class Models:
    def __init__(self):
        self.reg = GradientBoostingRegressor(
            min_samples_split=4,
            min_samples_leaf=2,
            max_depth=5,
            max_features='auto',
            random_state=3
        )

        self.params = {
            'learning_rate' : [ 0.05, 0.1],
            'n_estimators' : range(50,301,50)
        }

    def grid_training(self, X, y):
        grid_reg = GridSearchCV(self.reg, self.params, cv=5).fit(X, y)
        
        best_score = grid_reg.best_score_
        best_params = grid_reg.best_params_
        best_model = grid_reg.best_estimator_

        return best_score, best_params, best_model