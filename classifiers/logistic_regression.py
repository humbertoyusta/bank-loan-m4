
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers.preprocessor import Preprocessor
from sklearn.base import BaseEstimator

class LogisticRegressionClassifier(BaseEstimator):
    def __init__(self, preprocessor_params={}, logistic_regression_params={}):
        self.preprocessor_params = preprocessor_params
        self.logistic_regression_params = logistic_regression_params
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.model = LogisticRegression(**self.logistic_regression_params)

    def fit(self, X, y):
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('logistic_regression', self.model)
        ])
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_params(self, deep=True):
        return {"preprocessor_params": self.preprocessor_params,
                "logistic_regression_params": self.logistic_regression_params}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.model = LogisticRegression(**self.logistic_regression_params)
        return self
