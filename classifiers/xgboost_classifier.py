
import yaml
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from transformers.preprocessor import Preprocessor
from sklearn.base import BaseEstimator

class XGBoostClassifier(BaseEstimator):
    def __init__(self, preprocessor_params={}, xgboost_params={}):
        self.preprocessor_params = preprocessor_params
        self.xgboost_params = xgboost_params
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.model = XGBClassifier(**self.xgboost_params)

    def fit(self, X, y):
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('xgboost', self.model)
        ])
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def get_params(self, deep=True):
        return {"preprocessor_params": self.preprocessor_params,
                "xgboost_params": self.xgboost_params}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.model = XGBClassifier(**self.xgboost_params)
        return self
