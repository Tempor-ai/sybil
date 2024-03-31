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
    def __init__(self, type: str,  scorers: Union[METRIC_TYPE, List[METRIC_TYPE]], rnn_model:str = None, rnn_model_ckpt:str = None,):
        self.type = type
        self.scorers = scorers if isinstance(scorers, list) else [scorers]
        self.train_idx = None
        self.isExogenous = False
        self.rnn_model = rnn_model
        self.rnn_model_ckpt =rnn_model_ckpt

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
        X = data.iloc[:, :-1]
        if data.shape[1] > 1:  # Exogenous variables are present
            self.isExogenous = True
        else:
            self.isExogenous = False

        y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=test_size, shuffle=False)
        print(f"Training model {self.type} on {len(y_train)} samples. (TRAIN DATA)")
        self._train(y=y_train, X=X_train, isExogenous=self.isExogenous)
        scores = self.score(y_test, X=X_test)

        self.train_idx = data.index
        print(f"Training model {self.type} on {len(y)} samples. (FULL DATA)")
        self._train(y=y, X=X, isExogenous=self.isExogenous)  # Refit with full data

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
        # X = None if self.isExogenous is False else X
        y_pred = self._predict(lookforward=lookforward, X=X, isExogenous=self.isExogenous)
        if self.train_idx is not None:
            timestep = self.train_idx[-1] - self.train_idx[-2]
            start = self.train_idx[-1] + timestep
            index = [start+i*timestep for i in range(0, lookforward)]
            y_pred = pd.Series(y_pred, index=index)
        return y_pred
    
    def isExternalModel(self):
        if self.type == 'neuralprophet':
            return True
        else:
            return False

    @abstractmethod
    def _train(self, y: pd.Series, X: pd.DataFrame = None, isExogenous: bool = False) -> None:
        """
        Internal method to train the model.

        @param y: Time series of shape (t,).
        @param isExogenous: Boolean indicating whether exogenous variables are present.
        @param X: Exogenous variables of shape (t, n).
        @return: None
        """
        pass

    @abstractmethod
    def _predict(self, lookforward: int = 1, X: pd.DataFrame = None, isExogenous: bool = False) -> np.ndarray:
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
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None, isExogenous: bool = False) -> None:
        X = None if isExogenous is False else X
        y_val = y.values
        X_val = None if X is None else X.values
        self.model.fit(y=y_val, X=X_val)

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None, isExogenous: bool = False)-> np.ndarray:
        X = None if isExogenous is False else X
        X_val = None if X is None else X.values
        return self.model.predict(h=lookforward, X=X_val)['mean']


class DartsWrapper(AbstractModel):
    """
    Wrapper for Darts models according to the AbstractModel interface.
    """
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None, isExogenous: bool = False) -> None:
        X = None if isExogenous is False else X
        y_time_series = TimeSeries.from_series(y)
        if X is not None and has_argument(self.model.fit, 'future_covariates'):
            X_time_series = TimeSeries.from_dataframe(X)
            self.model.fit(y_time_series, future_covariates=X_time_series)
        else:
            self.model.fit(y_time_series)

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None, isExogenous: bool = False)-> np.ndarray:
        X = None if isExogenous is False else X
        if X is not None and has_argument(self.model.fit, 'future_covariates'):
            X_ts = TimeSeries.from_dataframe(X)
            y_ts = self.model.predict(n=lookforward, future_covariates=X_ts)
        else:
            y_ts = self.model.predict(n=lookforward)
        return y_ts.values().ravel()

class NeuralProphetWrapper(AbstractModel):
    """
    Wrapper for neuralProphet models according to the AbstractExternalModel interface.
    """
    def __init__(self, neuralProphet_model, base_model_config, *args, **kwargs):
        self.neuralProphet_model = neuralProphet_model
        self.base_model_config = base_model_config
        super().__init__(*args, **kwargs)

    def _train(self, data: pd.DataFrame, external_base_model_config) -> None:
        model = self.neuralProphet_model.fit(data, external_base_model_config)
        self.model = model

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None, isExogenous: bool = False)-> np.ndarray:
        y_ts = self.neuralProphet_model.predict(X, self.model)
        return y_ts

# TODO implement the external model train and forcast
class MetaModelWA(AbstractModel):
    """
    MetaModel using the weighted average.
    """
    def __init__(self, base_models, *args, **kwargs):
        self.base_models = base_models
        self.models_weights = {}
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None, isExogenous: bool = False) -> float:
        base_predictions = []
        main_scorer = self.scorers[0]
        base_scores = {}

        for model in self.base_models:
            y_base, y_meta, X_base, X_meta = train_test_split(y, X, test_size=0.2, shuffle=False)
            df_base = X_base.join(y_base)
            df_meta = X_meta.join(y_meta)
            df_combined = X.join(y)
            print(f"\nFitting base model: {model.type}")

            if model.isExternalModel():
                model._train(df_base, model.base_model_config)
                y_pred = model.predict(lookforward=len(y_meta), X=X_meta)
                base_scores[model.type] = main_scorer(y_meta, y_pred)
                print(f"{model.type} {main_scorer.__name__} test score: {base_scores[model.type]}")
                base_predictions.append(y_pred)
                model._train(df_combined, model.base_model_config)  # Refit with full data
                model.train_idx = y.index
            else:
                y_base, y_meta = train_test_split(y, test_size=0.2, shuffle=False)
                # TODO - Need to add X to the train method
                model._train(y_base)
                y_pred = model.predict(len(y_meta))
                base_scores[model.type] = main_scorer(y_meta, y_pred)
                print(f"{model.type} {main_scorer.__name__} test score: {base_scores[model.type]}")
                base_predictions.append(y_pred)
                model._train(y)
                model.train_idx = y.index
            
        total_score = sum(base_scores.values())
        self.models_weights = {model.type: base_scores[model.type] / total_score
                               for model in self.base_models}

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None, isExogenous: bool = False) -> np.ndarray:
        base_predictions = {}#convert 264 to block and add x variable initiation like train method
        for model in self.base_models:
            if model.isExternalModel():
                base_predictions[model.type] = model.predict(lookforward, X)
            else:
                base_predictions[model.type] = model.predict(lookforward)
             

        #base_predictions = {model.type: model.predict(lookforward) for model in self.base_models} converted to 261-265
        meta_predictions = sum([base_predictions[model.type] * self.models_weights[model.type]
                                for model in self.base_models])
        return meta_predictions


class MetaModelNaive(AbstractModel):
    """
    MetaModel using Naive ensemble. All base models are equally weighted
    """
    def __init__(self, base_models, *args, **kwargs):
        self.base_models = base_models
        self.models_weights = {}
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None) -> float:
        y_base, y_meta = train_test_split(y, test_size=0.2, shuffle=False)
        main_scorer = self.scorers[0]
        base_scores = {}
        for model in self.base_models:
            print(f"\nFitting base model: {model.type}")
            model._train(y_base)
            y_pred = model.predict(len(y_meta))
            base_scores[model.type] = main_scorer(y_meta, y_pred)
            print(f"{model.type} {main_scorer.__name__} test score: {base_scores[model.type]}")
            model._train(y)
            model.train_idx = y.index
        num_models = len(self.base_models)
        self.models_weights = {model.type: 1/num_models
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

    def __init__(self, base_models, *args, **kwargs):
        self.base_models = base_models
        self.regressor = LinearRegression()
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None, isExogenous: bool = False) -> None:
        base_predictions = []
        main_scorer = self.scorers[0]

        for model in self.base_models:
            y_base, y_meta, X_base, X_meta = train_test_split(y, X, test_size=0.2, shuffle=False)
            df_base = X_base.join(y_base)
            df_meta = X_meta.join(y_meta)
            df_combined = X.join(y)

            print(f"\nFitting base model: {model.type}")
            if model.isExternalModel():
                model._train(df_base, model.base_model_config)
                y_pred = model.predict(lookforward=len(y_meta), X=X_meta)
                base_predictions.append(y_pred)
                model._train(df_combined, model.base_model_config)  # Refit with full data
            else:
                model._train(y_base, X=X_base)
                y_pred = model.predict(lookforward=len(y_meta), X=X_meta)
                base_predictions.append(y_pred)
                test_score = main_scorer(y_meta, y_pred)
            print(f"{model.type} {main_scorer.__name__} test score: {test_score}")
            model._train(y, X=X)  # Refit with full data
            


        # Use linear regression to learn the weights
        base_predictions = np.column_stack(base_predictions)
        # base_predictions_df=pd.DataFrame(data=base_predictions, index=y_meta.index).dropna()
        # base_predictions_filtered = base_predictions_df.values
        # y_meta_filtered = y_meta[y_meta.index.isin(base_predictions_df.index)]
        self.regressor.fit(base_predictions, y_meta)

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None, isExogenous: bool = False) -> np.ndarray:
        base_predictions = [model.predict(lookforward, X) for model in self.base_models]
        base_predictions = np.column_stack(base_predictions)
        meta_predictions = self.regressor.predict(base_predictions)
        return meta_predictions.ravel()


def has_argument(func, arg_name):
    return arg_name in func.__code__.co_varnames
