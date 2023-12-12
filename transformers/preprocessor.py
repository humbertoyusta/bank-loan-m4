
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, add_polynomial_features=False, degree=2):
        self.scaler = MinMaxScaler()
        self.add_polynomial_features = add_polynomial_features
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False) if self.add_polynomial_features else None

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)

        if self.add_polynomial_features:
            self.poly.fit(X_scaled)

        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)

        if self.add_polynomial_features:
            X_scaled = self.poly.transform(X_scaled)

        return X_scaled
