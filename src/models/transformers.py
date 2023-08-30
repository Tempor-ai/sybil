"""
Library of transformers for data preprocessing
"""

import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod


class AbstractTransformer(ABC):
    """
    Abstract class used for defining Transformers interface.
    """

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.Series, pd.DataFrame]) -> None:
        """
        Fit the selected transformer.

        @param X: Generic input data of shape (t, n).
        @return: None
        """
        pass

    @abstractmethod
    def transform(self, X: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        Transforms the input data according to the transformer logic.

        @param X: Generic input data of shape (t, n).
        @return: Transformed data.
        """
        pass

    def fit_transform(self, X: Union[np.ndarray, pd.Series, pd.DataFrame]) -> None:
        """
        Fit and transform the input data according to the transformer logic.

        @param X: Generic input data of shape (t, n).
        @return: Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the input data according to the transformer logic.
        :param X: Generic input data of shape (t, n).
        :return: Data in the original format.
        """
        pass


class MinMaxScaler(AbstractTransformer):
    """
    MinMaxScaler transformer.
    :param min: Minimum value of the range.
    :param max: Maximum value of the range.
    """
    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max
        self.min_ = None
        self.max_ = None

    def fit(self, X: Union[np.ndarray, pd.Series, pd.DataFrame]) -> None:
        self.min_ = X.min()
        self.max_ = X.max()

    def transform(self, X: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        return (X - self.min_) / (self.max_ - self.min_) * (self.max - self.min) + self.min

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min) / (self.max - self.min) * (self.max_ - self.min_) + self.min_
