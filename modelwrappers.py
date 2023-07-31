"""
Library of model interfaces and wrappers for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Callable
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """
    Abstract class used for both Base ("Sibyl") and Meta ("Pythia") models.
    Modeled after the Nixtla library's model interface.
    """
    def __init__(self, scorer: Callable[[np.ndarray|pd.DataFrame, np.ndarray|pd.Series], float]):
        self.scorer = scorer

    def score(self, y: np.ndarray|pd.Series, X: np.ndarray|pd.DataFrame=None) -> float:
        """
        Score the model on the given data.

        @param y: Future values of time series of shape (t,).
        @param X: Exogenous variables of shape (t, n).
        @return: The score of the model on the given data.
        """
        return self.scorer(y, self.predict(len(y), X))

    def plot_prediction(self, y:np.ndarray|pd.Series, X:np.ndarray|pd.DataFrame=None):
        """
        Plot the actual and predicted values on a line chart.

        @param y: Future values of time series of shape (t,).
        @param X: Exogenous variables of shape (t, n).
        @return: The score of the model on the given data.
        """
        index = range(len(y)) if isinstance(y, np.ndarray) else y.index
        plt.plot(index, y, label='Actual')
        y_pred = self.predict(lookforward=len(y), X=X)
        plt.plot(index, y_pred, label='Model')
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    @abstractmethod
    def fit(self, y: np.ndarray|pd.Series, X: np.ndarray|pd.DataFrame=None) -> float:
        """
        Fit the selected model.

        @param y: Time series of shape (t,).
        @param X: Exogenous variables of shape (t, n).
        @return: The score on the training data.
        """
        pass

    @abstractmethod
    def predict(self, lookforward: int=1, X: np.ndarray|pd.DataFrame=None) -> np.ndarray:
        """
        Predict the next lookforward steps.

        @param lookforward: Number of steps to predict.
        @param X: Exogenous variables of shape (lookforward, n_x).
        @return: An array with the predicted values of shape (lookforward,).
        """
        pass


class StatsforecastWrapper(AbstractModel):
    """
    Wrapper for statsforecast models according to the AbstractModel interface.
    """
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def fit(self, y: np.ndarray|pd.Series, X: np.ndarray|pd.DataFrame=None) -> float:
        y_val = self._validate_input_array(y)
        X_val = None if X is None else self._validate_input_array(X)
        self.model.fit(y_val, X_val)
        return self.score(y_val, X_val)

    def predict(self, lookforward: int=1, X: np.ndarray|pd.DataFrame=None) -> np.ndarray:
        X_val = None if X is None else self._validate_input_array(X)
        return self.model.predict(h=lookforward, X=X_val)['mean']

    def _validate_input_array(self, X: np.ndarray|pd.DataFrame) -> np.ndarray:
        """
        Ensure the input data is a numpy array to be used by statsforecast models.

        @param X: Input data of shape (t, n).
        @return: The input data as a numpy array.
        """
        return X if isinstance(X, np.ndarray) else X.to_numpy()