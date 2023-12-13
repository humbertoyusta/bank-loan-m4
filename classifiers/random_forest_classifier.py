
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifierDefault
from sklearn.pipeline import Pipeline
from transformers.preprocessor import Preprocessor
from sklearn.base import BaseEstimator

class RandomForestClassifier(BaseEstimator):
    def __init__(self, preprocessor_params={}, random_forest_params={}):
        self.preprocessor_params = preprocessor_params
        self.random_forest_params = random_forest_params
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.model = RandomForestClassifierDefault(**self.random_forest_params)

    def fit(self, X, y):
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('random_forest', self.model)
        ])
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_params(self, deep=True):
        return {"preprocessor_params": self.preprocessor_params,
                "random_forest_params": self.random_forest_params}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.model = RandomForestClassifier(**self.random_forest_params)
        return self
