
import yaml
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from transformers.preprocessor import Preprocessor
from sklearn.base import BaseEstimator

class KNNClassifier(BaseEstimator):
    def __init__(self, preprocessor_params={}, knn_params={}):
        self.preprocessor_params = preprocessor_params
        self.knn_params = knn_params
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.model = KNeighborsClassifier(**self.knn_params)

    def fit(self, X, y):
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('knn', self.model)
        ])
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_params(self, deep=True):
        return {"preprocessor_params": self.preprocessor_params,
                "knn_params": self.knn_params}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.model = KNeighborsClassifier(**self.knn_params)
        return self
