"""
Library of transformers for data preprocessing
"""

import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod


class AbstractPreprocessor(ABC):
    """
    Abstract class used for defining Preprocessor interface.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the selected processor.

        @param X: Generic input data of shape (t, n).
        @return: None
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data according to the preprocessor logic.

        @param X: Generic input data of shape (t, n).
        @return: Transformed data.
        """
        pass

    def fit_transform(self, X: pd.DataFrame) -> None:
        """
        Fit and transform the input data according to the preprocessor logic.

        @param X: Generic input data of shape (t, n).
        @return: Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the input data according to the preprocessor logic.
        :param X: Generic input data of shape (t, n).
        :return: Data in the original format.
        """
        pass


class MinMaxScaler(AbstractPreprocessor):
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

    def fit(self, X: pd.DataFrame) -> None:
        self.min_ = X.min()
        self.max_ = X.max()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self.min_) / (self.max_ - self.min_) * (self.max - self.min) + self.min

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self.min) / (self.max - self.min) * (self.max_ - self.min_) + self.min_


class SimpleImputer(AbstractPreprocessor):

    """SimpleImputer class to fill missing values.

    Args:
        strategy (str): The imputation strategy.
            - "mean" : Replace missing values using the mean
            - "median" : Replace missing values using the median
            - "most_frequent" : Replace missing values with the most frequent value
            - "constant" : Replace missing values with a constant value
    """

    def __init__(self, strategy="mean", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.mask = None

    def fit(self, X: pd.DataFrame) -> None:
        """Calculate the fill value based on the strategy."""
        if self.strategy == "mean":
            self.fill_value = X.mean()
        elif self.strategy == "median":
            self.fill_value = np.median(X)
        elif self.strategy == "constant":
            pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in X."""
        X_imputed = X.copy().values
        self.mask = np.isnan(X_imputed)
        for i in range(X_imputed.shape[1]):
            X_imputed[self.mask[:, i], i] = self.fill_value[i]
        return pd.DataFrame(X_imputed, index=X.index, columns=X.columns)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform X back to original with missing values."""
        X_missing = X.copy().values
        X_missing[self.mask] = np.nan
        return pd.DataFrame(X_missing, index=X.index, columns=X.columns)
