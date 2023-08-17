"""
Library of model interfaces and wrappers for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Callable
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from .ts_utils import get_seasonal_period, smape, mape
from sklearn.model_selection import train_test_split
from typing import Union, List


METRIC_TYPE = Callable[[np.ndarray, np.ndarray], float]
SCORERS_DICT = {'smape': smape, 'mape': mape}


class AbstractModel(ABC):
    """
    Abstract class used for both Base ("Sibyl") and Meta ("Pythia") models.

    @param type: The type of the model, also used to build a new model from the Factory.
    @param scorers: A list of scorers to use for evaluation.
    """
    def __init__(self, type: str, scorers: Union[METRIC_TYPE, List[METRIC_TYPE]]):
        self.type = type
        self.scorers = scorers if isinstance(scorers, list) else [scorers]

    def score(self, y: Union[np.ndarray, pd.Series], X: Union[np.ndarray, pd.DataFrame]=None) -> dict:
        """
        Score the model on the given data.

        @param y: Future values of time series of shape (t,).
        @param X: Future exogenous variables of shape (t, n).
        @return: A dictionary of scores.
        """
        y_pred = self.predict(len(y), X)
        return {scorer.__name__: scorer(y, y_pred) for scorer in self.scorers}

    def plot_prediction(self, y: Union[np.ndarray, pd.Series], X: Union[np.ndarray, pd.DataFrame]=None) -> None:
        """
        Plot the actual and predicted values on a line chart.

        @param y: Future values of time series of shape (t,).
        @param X: Exogenous variables of shape (t, n).
        @return: None
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

    # TODO: Need to integrate exogenous variables X
    def train(self, data: Union[np.ndarray,pd.DataFrame], test_size=0.1) -> dict:
        """
        Train a model on the given data.

        @param data: The data to train on.
        Leftmost column is the time column and the rightmost column is the value column.
        Any columns in between are exogenous variables.
        @param test_size: The ratio of data to use for testing.
        @return: A dictionary containing the trained model and any other information.
        """
        y = self._validate_input_array(data)[:, -1]
        y_train, y_test = train_test_split(y, test_size=test_size, shuffle=False)
        self.fit(y_train)
        scores = self.score(y_test)
        self.plot_prediction(y_test)
        return {'model': self,
                'type': self.type,
                'evaluation': scores}  # 'stats': {'season_length': season_length}

    @abstractmethod
    def fit(self, y: Union[np.ndarray, pd.Series], X: Union[np.ndarray,pd.DataFrame]=None) -> None:
        """
        Fit the selected model.

        @param y: Time series of shape (t,).
        @param X: Exogenous variables of shape (t, n).
        @return: None
        """
        pass

    @abstractmethod
    def predict(self, lookforward: int=1, X: Union[np.ndarray,pd.DataFrame]=None) -> np.ndarray:
        """
        Predict the next lookforward steps.

        @param lookforward: Number of steps to predict.
        @param X: Exogenous variables of shape (lookforward, n_x).
        @return: An array with the predicted values of shape (lookforward,).
        """
        pass

    def _validate_input_array(self, X: Union[np.ndarray,pd.DataFrame]) -> np.ndarray:
        """
        Ensure the input data is a numpy array to be used by statsforecast models.

        @param X: Input data of shape (t, n).
        @return: The input data as a numpy array.
        """
        return X if isinstance(X, np.ndarray) else X.to_numpy()

    @staticmethod
    def prepare_dataset(dataset: pd.DataFrame, time_col=0, value_col=-1) -> pd.DataFrame:
        """
        Prepare the dataset for training or prediction.

        :param dataset: Dataframe containing the dataset.
        :param time_col: Column name identifying the time component.
        :param value_col: Column name identifying the value component.
        :return: Dataframe with the time and value columns renamed to 'datetime' and 'value' respectively.
        """
        time_col_name = dataset.columns[time_col]
        value_col_name = dataset.columns[value_col]
        clean_dataset = dataset.rename(columns={time_col_name: 'datetime',
                                                value_col_name: 'value'})
        clean_dataset['datetime'] = pd.to_datetime(clean_dataset['datetime'])
        return clean_dataset.set_index('datetime')


class StatsforecastModel(AbstractModel):
    """
    Wrapper for statsforecast models according to the AbstractModel interface.
    """
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def fit(self, y: Union[np.ndarray,pd.Series], X: Union[np.ndarray, pd.DataFrame]=None) -> float:
        y_val = self._validate_input_array(y)
        X_val = None if X is None else self._validate_input_array(X)
        self.model.fit(y_val, X_val)

    def predict(self, lookforward: int=1, X: Union[np.ndarray, pd.DataFrame]=None) -> np.ndarray:
        X_val = None if X is None else self._validate_input_array(X)
        return self.model.predict(h=lookforward, X=X_val)['mean']


class ModelFactory():
    """
    Factory class for creating models.
    """
    @staticmethod
    def create_model(dataset: pd.DataFrame,
                     model_info) -> AbstractModel:
        """
        Create a model of the given type.

        @param dataset: A dataframe containing the dataset with the time column as the first column
        and the target column as the last column.
        @param model_info: A dictionary containing the model type and any other information.
        The format is the following:
            {
              "data": [
                [1, 10],
                [2, 5],
                [3, 7],
                [4, 1],
                [5, 10]
              ],
              "model": {
                "type": "meta-lr",
                "score": ["smape"],
                "param": {
                  "models": [
                    {
                      "type": "autoarima",
                      "score": ["smape"],
                      "param": {}
                    },
                    {
                      "type": "autotheta",
                      "score": ["smape"],
                      "param": {}
                    }
                  ]
                }
              }
            }
        @return: A model of the given type.
        """
        season_length = get_seasonal_period(dataset)
        scorers = [SCORERS_DICT[s] for s in model_info.score]
        if model_info.type == 'autotheta':
            from statsforecast.models import AutoTheta
            statsmodel = AutoTheta(season_length=season_length)
            return StatsforecastModel(model=statsmodel,
                                      scorers=scorers,
                                      type='autotheta')
        elif model_info.type == 'autoarima':
            from statsforecast.models import AutoARIMA
            statsmodel = AutoARIMA(season_length=season_length)
            return StatsforecastModel(model=statsmodel,
                                      scorers=scorers,
                                      type='autoarima')
        else:
            raise ValueError(f'Unknown model type: {model_info.score}')
