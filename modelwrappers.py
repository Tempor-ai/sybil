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
    """
    def __init__(self, scorer: Callable[[np.ndarray|pd.DataFrame, np.ndarray|pd.Series], float]):
        self.scorer = scorer

    def score(self, X: np.ndarray|pd.DataFrame, y: np.ndarray|pd.Series) -> float:
        """
        Score the model on the given data.
        """
        return self.scorer(y, self.predict(X, len(y)))

    def plot_prediction(self, X:np.ndarray|pd.DataFrame, y:np.ndarray|pd.Series):
        """
        Plot the actual and predicted values.
        """
        index = range(len(y)) if isinstance(y, np.ndarray) else y.index
        plt.plot(index, y, label='Actual')
        y_pred = self.predict(X, lookforward=len(y))
        plt.plot(index, y_pred, label='Model')
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    @abstractmethod
    def fit(self, X: np.ndarray|pd.DataFrame, y: np.ndarray|pd.Series=None) -> float:
        """
        Fit the selected model.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray|pd.DataFrame, lookforward: int=1) -> np.ndarray:
        """
        Predict the next lookforward steps.
        """
        pass


class StatsforecastWrapper(AbstractModel):
    """
    Wrapper for statsforecast models according to the AbstractModel interface.
    """
    def __init__(self, model, *args, **kwargs):
        """
        Initialize the wrapper with a specific model type.
        """
        self.model = model
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray|pd.DataFrame, y: np.ndarray|pd.Series=None) -> float:
        """
        Fit the selected model.
        """
        self.model.fit(self._validate_input_array(X))

    def predict(self, X: np.ndarray|pd.DataFrame, lookforward: int=1) -> np.ndarray:
        """
        Predict the next lookforward steps.
        """
        return self.model.predict(h=lookforward)['mean']

    def _validate_input_array(self, X: np.ndarray|pd.DataFrame) -> np.ndarray:
        """
        Ensure the input data is a numpy array.
        """
        return X if isinstance(X, np.ndarray) else X.to_numpy()