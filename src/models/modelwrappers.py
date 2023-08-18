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
from darts import TimeSeries
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
        Ensure the input data is a numpy array.

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


class DartsModel(AbstractModel):
    """
    Wrapper for Darts models according to the AbstractModel interface.
    """
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def fit(self, y: Union[np.ndarray,pd.Series], X: Union[np.ndarray,pd.DataFrame]=None) -> float:
        # TODO: Implement exogenous variables
        y_time_series = TimeSeries.from_values(y)
        self.model.fit(y_time_series)

    def predict(self, lookforward: int=1, X: Union[np.ndarray,pd.DataFrame]=None) -> np.ndarray:
        # TODO: Implement exogenous variables
        y_timeseries = self.model.predict(n=lookforward)
        return y_timeseries.values().ravel()


class MetaModelWA(AbstractModel):
    """
    MetaModel using the weighted average.
    """
    def __init__(self, models, *args, **kwargs):
        self.base_models = models
        self.models_weights = {}
        super().__init__(*args, **kwargs)

    def fit(self, y: Union[np.ndarray,pd.Series], X: Union[np.ndarray,pd.DataFrame]=None) -> float:
        # TODO: Implement exogenous variables
        y_base, y_meta = train_test_split(y, test_size=0.2, shuffle=False)
        main_scorer = self.scorers[0]
        base_scores = {}
        for model in self.base_models:
            print(f"Fitting base model: {model.type}")
            model.fit(y_base)
            y_pred = model.predict(len(y_meta))
            base_scores[model.type] = main_scorer(y_meta, y_pred)
            print(f"{main_scorer.__name__} test score: {base_scores[model.type]}")
            model.fit(y)
        total_score = sum(base_scores.values())
        self.models_weights = {model.type: base_scores[model.type] / total_score
                               for model in self.base_models}

    def predict(self, lookforward: int=1, X: Union[np.ndarray,pd.DataFrame]=None) -> np.ndarray:
        # TODO: Implement exogenous variables
        base_predictions = {model.type: model.predict(lookforward) for model in self.base_models}
        meta_predictions = sum([base_predictions[model.type] * self.models_weights[model.type]
                                for model in self.base_models])
        return meta_predictions


class ModelFactory():
    """
    Factory class for creating models.
    """
    @staticmethod
    def create_model(dataset: pd.DataFrame,
                     model_type: str = 'darts_autotheta',
                     model_params: dict = None,
                     scorers: Union[str, List[str]] = 'mape') -> AbstractModel:
        """
        Create a model of the given type.

        @param dataset: A dataframe containing the dataset with the time column as the first column
        and the target column as the last column.
        @param model_type: The type of the model to create. Defaults to 'darts_autotheta' if None.
        @param model_params: A dictionary containing the model parameters if necessary.
        @param scorers: A list of scorers to use for evaluation. Defaults to MAPE if None.
        @return: A model of the given type.
        """
        scorer_func = [SCORERS_DICT[s] for s in scorers] if isinstance(scorers, list) \
            else SCORERS_DICT[scorers]
        season_length = get_seasonal_period(dataset)
        if model_type == 'stats_autotheta':
            from statsforecast.models import AutoTheta
            model = StatsforecastModel(model=AutoTheta(season_length=season_length),
                                       scorers=scorer_func,
                                       type=model_type)
        elif model_type == 'stats_autoarima':
            from statsforecast.models import AutoARIMA
            model = StatsforecastModel(model=AutoARIMA(season_length=season_length),
                                       scorers=scorer_func,
                                       type=model_type)
        elif model_type == 'stats_autoets':
            from statsforecast.models import AutoETS
            model = StatsforecastModel(model=AutoETS(season_length=season_length),
                                       scorers=scorer_func,
                                       type=model_type)
        elif model_type == 'darts_autotheta':
            from darts.models import StatsForecastAutoTheta
            model = DartsModel(model=StatsForecastAutoTheta(season_length=season_length),
                               scorers=scorer_func,
                               type=model_type)
        elif model_type == 'darts_autoarima':
            from darts.models import StatsForecastAutoARIMA
            model = DartsModel(model=StatsForecastAutoARIMA(season_length=season_length),
                               scorers=scorer_func,
                               type=model_type)
        elif model_type == 'darts_autoets':
            from darts.models import StatsForecastAutoETS
            model = DartsModel(model=StatsForecastAutoETS(season_length=season_length),
                               scorers=scorer_func,
                               type=model_type)
        elif model_type == 'meta_wa':
            # TODO: Pass also base models parameters
            if model_params is None:
                base_models = [ModelFactory.create_model(dataset, model_type=m)
                               for m in ['darts_autoets', 'darts_autoarima', 'darts_autotheta']]
            else:
                base_models = [ModelFactory.create_model(dataset, base_model["type"])
                               for base_model in model_params['base_models']]
            model = MetaModelWA(models=base_models,
                                scorers=scorer_func,
                                type=model_type)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

        return model