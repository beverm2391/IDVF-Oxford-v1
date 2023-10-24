from abc import ABC, abstractmethod
import numpy as np
from sklearn import linear_model

class Model(ABC):
    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.coefficients = None
        self.fit_intercept = fit_intercept  # Added

    def train(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched dimensions: X has {} rows but y has {} rows".format(X.shape[0], y.shape[0]))

        if self.fit_intercept:
            X = self._add_bias_term(X)  # Added

        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Model has not been trained yet.")

        if self.fit_intercept: 
            X = self._add_bias_term(X)  # Added

        return X @ self.coefficients

    def _add_bias_term(self, X):  # Added
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Added


class LassoRegression:
    def __init__(self, alpha, tol=1e-3):
        self.alpha = alpha
        self.tol = tol
        self.clf = linear_model.Lasso(alpha=alpha, tol=tol)

    def train(self, X, y):
        self.clf.fit(X,y)

    def predict(self, X):
        return self.clf.predict(X)