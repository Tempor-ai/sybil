"""
Library of model interfaces and wrappers for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Callable
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from darts import TimeSeries
from typing import Union, List

METRIC_TYPE = Callable[[np.ndarray, np.ndarray], float]


class AbstractModel(ABC):
    """
    Abstract class used for both Base ("Sibyl") and Meta ("Pythia") models.

    @param type: The type of the model, also used to build a new model from the Factory.
    @param scorers: A list of scorers to use for evaluating metrics.
    """
    def __init__(self, type: str, scorers: Union[METRIC_TYPE, List[METRIC_TYPE]]):
        self.type = type
        self.scorers = scorers if isinstance(scorers, list) else [scorers]
        self.train_idx = None

    def score(self, y: pd.Series, X: pd.DataFrame=None) -> dict:
        """
        Score the model on the given data.

        @param y: Future values of time series of shape (t,).
        @param X: Future exogenous variables of shape (t, n).
        @return: A dictionary of scores.
        """
        y_pred = self.predict(lookforward=len(y), X=X)
        return {scorer.__name__: scorer(y, y_pred) for scorer in self.scorers}

    def plot_prediction(self, y: pd.Series, X: pd.DataFrame=None) -> None:
        """
        Plot the actual and predicted values on a line chart.

        @param y: Future values of time series of shape (t,).
        @param X: Exogenous variables of shape (t, n).
        @return: None
        """
        index = range(len(y)) if isinstance(y, np.ndarray) else y.index
        plt.plot(index, y, label='Actual')
        y_pred = self.predict(lookforward=len(y), X=X)
        plt.plot(index, y_pred, label=self.type)
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # TODO: Need to integrate exogenous variables X
    def train(self, data: pd.DataFrame, test_size=0.1) -> dict:
        """
        Train a model on the given data.

        @param data: The data to train on which should be Pandas DataFrame with a time index.
        The leftmost column is the target while any other columns are exogenous variables.
        @param test_size: The ratio of data to use for testing.
        @return: A dictionary containing the trained model and any other information.
        """
        # df = X.join(y)
        y = data.iloc[:, -1]
        if data.shape[1] > 1:  # Exogenous variables are present
            X = data.iloc[:, :-1]
            y_train, y_test, X_train, X_test = train_test_split(y, X,
                                                                test_size=test_size,
                                                                shuffle=False)
        else:
            X = None
            y_train, y_test = train_test_split(y,
                                               test_size=test_size,
                                               shuffle=False)
            X_train, X_test = None, None

        print(f"Training model {self.type} on {len(y_train)} samples. (TEST)")
        self._train(y=y_train, X=X_train)
        scores = self.score(y_test, X=X_test)

        self.train_idx = data.index
        print(f"Training model {self.type} on {len(y)} samples. (FULL)")
        self._train(y=y, X=X)  # Refit with full data

        return {'model': self,
                'type': self.type,
                'metrics': scores
                }

    def predict(self, lookforward: int = 1, X: pd.DataFrame = None) -> pd.Series:
        """
        Predict the next lookforward steps.

        @param lookforward: Number of steps to predict.
        @param X: Dataframe with exogenous variables of shape (lookforward, n_x).
        @return: A Pandas Series with the predicted values.
        """
        y_pred = self._predict(lookforward=lookforward, X=X)
        if self.train_idx is not None:
            timestep = self.train_idx[-1] - self.train_idx[-2]
            start = self.train_idx[-1] + timestep
            index = [start+i*timestep for i in range(0, lookforward)]
            y_pred = pd.Series(y_pred, index=index)
        return y_pred
    
    def isExternalModel(self):
        if self.type == 'NeuralProphet_model':
            return True
        else:
            return False

    @abstractmethod
    def _train(self, y: pd.Series, X: pd.DataFrame = None) -> None:
        """
        Internal method to train the model.

        @param y: Time series of shape (t,).
        @param X: Exogenous variables of shape (t, n).
        @return: None
        """
        pass

    @abstractmethod
    def _predict(self, lookforward: int = 1, X: pd.DataFrame = None) -> np.ndarray:
        """
        Internal method to predict the next lookforward steps.

        @param lookforward: Number of steps to predict.
        @param X: Exogenous variables of shape (lookforward, n_x).
        @return: An array with the predicted values of shape (lookforward,).
        """
        pass


class AbstractExternalModel(ABC):
    """
    Abstract class used for external models (etc. NeuralProphet_model).

    @param type: The type of the model, also used to build a new model from the Factory.
    @param scorers: A list of scorers to use for evaluating metrics.
    """
    # def __init__(self, type: str, scorers: Union[METRIC_TYPE, List[METRIC_TYPE]]):
    #     self.type = type
    #     self.scorers = scorers if isinstance(scorers, list) else [scorers]
    #     self.train_idx = None

    def __init__(self, neuralProphet_model, *args, **kwargs):
        self.neuralProphet_model = neuralProphet_model
        super().__init__(*args, **kwargs)

    def score(self, y: pd.Series, X: pd.DataFrame=None) -> dict:
        """
        Score the model on the given data.

        @param y: Future values of time series of shape (t,).
        @param X: Future exogenous variables of shape (t, n).
        @return: A dictionary of scores.
        """
        y_pred = self.predict(lookforward=len(y), X=X)
        return {scorer.__name__: scorer(y, y_pred) for scorer in self.scorers}

    def plot_prediction(self, y: pd.Series, X: pd.DataFrame=None) -> None:
        """
        Plot the actual and predicted values on a line chart.

        @param y: Future values of time series of shape (t,).
        @param X: Exogenous variables of shape (t, n).
        @return: None
        """
        index = range(len(y)) if isinstance(y, np.ndarray) else y.index
        plt.plot(index, y, label='Actual')
        y_pred = self.predict(lookforward=len(y), X=X)
        plt.plot(index, y_pred, label=self.type)
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # TODO: Need to integrate exogenous variables X
    def train(self, data: pd.DataFrame, test_size=0.1) -> dict:
        """
        format the data
        call the external model train method
        """
        y = data.iloc[:, -1]
        if data.shape[1] > 1:  # Exogenous variables are present
            X = data.iloc[:, :-1]
            y_train, y_test, X_train, X_test = train_test_split(y, X,
                                                                test_size=test_size,
                                                                shuffle=False)
        else:
            X = None
            y_train, y_test = train_test_split(y,
                                               test_size=test_size,
                                               shuffle=False)
            X_train, X_test = None, None

        print(f"Training model {self.type} on {len(y_train)} samples. (TEST)")
        self._train(y=y_train, X=X_train)
        scores = self.score(y_test, X=X_test)

        self.train_idx = data.index
        print(f"Training model {self.type} on {len(y)} samples. (FULL)")
        self._train(y=y, X=X)  # Refit with full data

        return {'model': self,
                'type': self.type,
                'metrics': scores
                }

    def predict(self, lookforward: int = 1, X: pd.DataFrame = None) -> pd.Series:
        """
        format the data
        call the external model predict method

        @param lookforward: Number of steps to predict.
        @param X: Dataframe with exogenous variables of shape (lookforward, n_x).
        @return: A Pandas Series with the predicted values.
        """
        y_pred = self._predict(lookforward=lookforward, X=X)
        if self.train_idx is not None:
            timestep = self.train_idx[-1] - self.train_idx[-2]
            start = self.train_idx[-1] + timestep
            index = [start+i*timestep for i in range(0, lookforward)]
            y_pred = pd.Series(y_pred, index=index)
        return y_pred

    @abstractmethod
    def _train(self, y: pd.Series, X: pd.DataFrame = None) -> None:
        """
        Internal method to train the model.

        @param y: Time series of shape (t,).
        @param X: Exogenous variables of shape (t, n).
        @return: None
        """
        pass

    @abstractmethod
    def _predict(self, lookforward: int = 1, X: pd.DataFrame = None) -> np.ndarray:
        """
        Internal method to predict the next lookforward steps.

        @param lookforward: Number of steps to predict.
        @param X: Exogenous variables of shape (lookforward, n_x).
        @return: An array with the predicted values of shape (lookforward,).
        """
        pass

class StatsforecastWrapper(AbstractModel):
    """
    Wrapper for statsforecast models according to the AbstractModel interface.
    """
    def __init__(self, stats_model, *args, **kwargs):
        self.stats_model = stats_model
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None) -> None:
        y_val = y.values
        X_val = None if X is None else X.values
        self.stats_model.fit(y=y_val, X=X_val)

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None) -> np.ndarray:
        X_val = None if X is None else X.values
        return self.stats_model.predict(h=lookforward, X=X_val)['mean']


class DartsWrapper(AbstractModel):
    """
    Wrapper for Darts models according to the AbstractModel interface.
    """
    def __init__(self, darts_model, *args, **kwargs):
        self.darts_model = darts_model
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None) -> None:
        y_time_series = TimeSeries.from_series(y)
        if X is not None and has_argument(self.darts_model.fit, 'future_covariates'):
            X_time_series = TimeSeries.from_dataframe(X)
            self.darts_model.fit(y_time_series, future_covariates=X_time_series)
        else:
            self.darts_model.fit(y_time_series)

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None) -> np.ndarray:
        if X is not None and has_argument(self.darts_model.fit, 'future_covariates'):
            X_ts = TimeSeries.from_dataframe(X)
            y_ts = self.darts_model.predict(n=lookforward, future_covariates=X_ts)
        else:
            y_ts = self.darts_model.predict(n=lookforward)
        return y_ts.values().ravel()

class NeuralProphetWrapper(AbstractModel):
    """
    Wrapper for neuralProphet models according to the AbstractExternalModel interface.
    """
    def __init__(self, neuralProphet_model, base_model_config, *args, **kwargs):
        self.neuralProphet_model = neuralProphet_model
        self.base_model_config = base_model_config
        super().__init__(*args, **kwargs)

    def _train(self, data: pd.DataFrame) -> None:
        self.neuralProphet_model.fit(data)

        
        # call model class train
        # def train(dataset, base_model_request):
        # # 
        # y_time_series = TimeSeries.from_series(y)
        # if X is not None and has_argument(self.neuralProphet_model.fit, 'future_covariates'):
        #     X_time_series = TimeSeries.from_dataframe(X)
        #     self.neuralProphet_model.fit(y_time_series, future_covariates=X_time_series)
        # else:
        # in TrainPayload
        # class TrainPayload(BaseModel):
        #     data: TrainDataPayload
        #     model: ModelPayload = Field(default_factory=ModelPayload)
        #     model_config = {"json_schema_extra": {"examples": [train_payload_example]}}

        # TrainDataPayload = List[List[PayloadValue]]
        # PayloadValue = str | int | float | bool



        # out TrainResponse
        # class TrainResponse(BaseModel):
        #     status: str
        #     metrics: dict[Any, Any]
        #     model: str
        # TrainResponse(
        #     status="ok",
        #     metrics={} if metrics is None else metrics.to_dict(),
        #     model=serialized,
        # )

        # format input into TrainPayload
        # api_json = {
        #     'data': train_data,
        #     'model': model_request  # (optional) can be commented out
        # }

        # model_request = {
        #     "params": {
        #     "changepoints_range": 0.2,
        #     "epochs": 2,
        #     "growth": "off"
        #     },
        #     "metrics": [],
        #     "type": "neuralprophet",
        # }

        # train_data
        # [['1949-01-01', 112.0],
        # ['1949-02-01', 118.0],
        # ['1949-03-01', 132.0],
        # ['1949-04-01', 129.0],
        # ['1949-05-01', 121.0],
        # ['1949-06-01', 135.0],
        # ['1949-07-01', 148.0],
        # ['1949-08-01', 148.0],
        # ['1949-09-01', 136.0],
        # ['1949-10-01', 119.0]]
        self.neuralProphet_model.fit(y_time_series)
        # return TrainResponse 

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None) -> np.ndarray:
        # if X is not None and has_argument(self.neuralProphet_model.fit, 'future_covariates'):
        #     X_ts = TimeSeries.from_dataframe(X)
        #     y_ts = self.neuralProphet_model.predict(n=lookforward, future_covariates=X_ts)
        # else:
        y_ts = self.neuralProphet_model.predict(n=lookforward)
        return y_ts.values().ravel()

# TODO implement the external model train and forcast
class MetaModelWA(AbstractModel):
    """
    MetaModel using the weighted average.
    """
    def __init__(self, models, *args, **kwargs):
        self.base_models = models
        self.models_weights = {}
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None) -> float:
        y_base, y_meta = train_test_split(y, test_size=0.2, shuffle=False)
        main_scorer = self.scorers[0]
        base_scores = {}
        for model in self.base_models:
            print(f"Fitting base model: {model.type}")
            # TODO - Need to add X to the train method
            model._train(y_base)
            y_pred = model.predict(len(y_meta))
            base_scores[model.type] = main_scorer(y_meta, y_pred)
            print(f"{main_scorer.__name__} test score: {base_scores[model.type]}")
            model._train(y)
            model.train_idx = y.index
        total_score = sum(base_scores.values())
        self.models_weights = {model.type: base_scores[model.type] / total_score
                               for model in self.base_models}

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None) -> np.ndarray:
        base_predictions = {model.type: model.predict(lookforward) for model in self.base_models}
        meta_predictions = sum([base_predictions[model.type] * self.models_weights[model.type]
                                for model in self.base_models])
        return meta_predictions


class MetaModelLR(AbstractModel):
    """
    MetaModel using Linear Regression to combine base models.
    """

    def __init__(self, models, *args, **kwargs):
        self.base_models = models
        self.regressor = LinearRegression()
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None) -> None:
        if X is None:
            y_base, y_meta = train_test_split(y, test_size=0.2, shuffle=False)
            X_base, X_meta = None, None
        else:
            y_base, y_meta, X_base, X_meta = train_test_split(y, X, test_size=0.2, shuffle=False)
        base_predictions = []

        df_base = X_base.join(y_base)
        df_meta = X_meta.join(y_meta)
        df_combined = X.join(y)
        for model in self.base_models:
            print(f"Fitting base model: {model.type}")
            if model.isExternalModel():
                model._train(df_base)
                y_pred = model.predict(lookforward=len(y_meta), X=X_meta)
                base_predictions.append(y_pred)
                model._train(df_combined)  # Refit with full data
            else:
                model._train(y_base, X=X_base)
                y_pred = model.predict(lookforward=len(y_meta), X=X_meta)
                base_predictions.append(y_pred)
                model._train(y, X=X)  # Refit with full data
            


        # Use linear regression to learn the weights
        base_predictions = np.column_stack(base_predictions)
        self.regressor.fit(base_predictions, y_meta)

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None) -> np.ndarray:
        base_predictions = [model.predict(lookforward, X) for model in self.base_models]
        X_meta = np.column_stack(base_predictions)
        meta_predictions = self.regressor.predict(X_meta)
        return meta_predictions.ravel()


def has_argument(func, arg_name):
    return arg_name in func.__code__.co_varnames
