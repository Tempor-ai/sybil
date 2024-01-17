"""
Library of transformers for data preprocessing
"""

import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values


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
        self.fill_mask = None

    def fit(self, X: pd.DataFrame) -> None:
        """Calculate the fill value based on the strategy."""
        if self.strategy == "mean":
            self.fill_value = X.mean()
        elif self.strategy == "median":
            self.fill_value = X.median()
        elif self.strategy == "constant":
            pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in X."""
        X_imputed = X.copy()
        self.fill_mask = X_imputed.isna()
        for col in X_imputed.columns:
            X_imputed.loc[self.fill_mask[col], col] = self.fill_value[col]
        return X_imputed

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform X back to original with missing values."""
        X_missing = X.copy()
        X_missing[self.fill_mask] = np.nan
        return X_missing

class FillMissingValues(AbstractPreprocessor):
    """
    Wrapper for Darts fill_missing_values function.
    """

    def __init__(self, fill="auto"):
        self.fill = fill
        self.fill_mask = None

    def fit(self, X: pd.DataFrame) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fill_mask = X.isna()
        return fill_missing_values(
            series=TimeSeries.from_dataframe(X), fill=self.fill
        ).pd_dataframe()

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_missing = X.copy()
        X_missing[self.fill_mask] = np.nan
        return X_missing
