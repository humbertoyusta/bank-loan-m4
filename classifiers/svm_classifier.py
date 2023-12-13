
import yaml
import pandas as pd
from sklearn.svm import SVC as SVCDefault
from sklearn.pipeline import Pipeline
from transformers.preprocessor import Preprocessor
from sklearn.base import BaseEstimator

class SVMClassifier(BaseEstimator):
    def __init__(self, preprocessor_params={}, svm_params={}):
        self.preprocessor_params = preprocessor_params
        self.svm_params = svm_params
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.model = SVCDefault(**self.svm_params)

    def fit(self, X, y):
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('svm', self.model)
        ])
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_params(self, deep=True):
        return {"preprocessor_params": self.preprocessor_params,
                "svm_params": self.svm_params}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.model = SVCDefault(**self.svm_params)
        return self
