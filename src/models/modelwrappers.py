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
import optuna
from darts.metrics import mse

METRIC_TYPE = Callable[[np.ndarray, np.ndarray], float]

# Defining an Epsilon for handling div/0 edge cases
epsilon=1e-10

class AbstractModel(ABC):
    """
    Abstract class used for both Base ("Sibyl") and Meta ("Pythia") models.

    @param type: The type of the model, also used to build a new model from the Factory.
    @param scorers: A list of scorers to use for evaluating metrics.
    """
    def __init__(self, type: str,  scorers: Union[METRIC_TYPE, List[METRIC_TYPE]], rnn_model:str = None, rnn_model_ckpt:str = None, isExogenous:bool = False):
        self.type = type
        self.scorers = scorers if isinstance(scorers, list) else [scorers]
        self.train_idx = None
        self.isExogenous = isExogenous
        self.rnn_model = rnn_model
        self.rnn_model_ckpt =rnn_model_ckpt

    def score(self, y: pd.Series, X: pd.DataFrame=None, y_train: pd.Series=None) -> dict:
        """
        Score the model on the given data.

        @param y: Future values of time series of shape (t,).
        @param X: Future exogenous variables of shape (t, n).
        @return: A dictionary of scores.
        """
        y_pred = self.predict(lookforward=len(y), X=X)
        return {scorer.__name__: scorer(y, y_pred, y_train) for scorer in self.scorers}

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

        y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=test_size, shuffle=False)
        print(f"Training model {self.type} on {len(y_train)} samples. (TRAIN DATA)")
        self._train(y=y_train, X=X_train)
        scores = self.score(y_test, X=X_test, y_train=y_train)

        self.train_idx = data.index
        print(f"Training model {self.type} on {len(y)} samples. (FULL DATA)")
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
        if self.type == 'neuralprophet':
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

class StatsforecastWrapper(AbstractModel):
    """
    Wrapper for statsforecast models according to the AbstractModel interface.
    """
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None) -> None:
        X = None if self.isExogenous is False else X
        y_val = y.values
        X_val = None if X is None else X.values
        self.model.fit(y=y_val, X=X_val)

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None)-> np.ndarray:
        X = None if self.isExogenous is False else X
        X_val = None if X is None else X.values
        return self.model.predict(h=lookforward, X=X_val)['mean']


class DartsWrapper(AbstractModel):
    """
    Wrapper for Darts models according to the AbstractModel interface.
    """
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame = None) -> None:

        param_grid = {
                'lags': [12, 24, 36],
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'lags_future_covariates': [[12], [24]]
            }

        if self.isExogenous and self.type == 'darts_autoarima':
            # print("Performing univariate analysis as model is auto_arima.")
            y_time_series = TimeSeries.from_series(y)
            self.model.fit(y_time_series)
        else:
            X = None if self.isExogenous is False else X
            y_time_series = TimeSeries.from_series(y)
            if X is not None and has_argument(self.model.fit, 'future_covariates'):
                X_time_series = TimeSeries.from_dataframe(X)
                if self.type == 'darts_lightgbm':
                    print("Entering Exo")
                    # print(X.shape)
                    # print(self.model)
                    # best_model = self.model.gridsearch(parameters=param_grid,series=y_time_series,forecast_horizon=10,verbose=-1,future_covariates=X_time_series)
                    
                    
                    print(self.model, " Self model")
                    # current_params = self.model.get_params()
                    # best_model, best_params,test = self.model.gridsearch(
                    # parameters=param_grid,
                    # series=y_time_series,
                    # # future_covariates=X_time_series,
                    # forecast_horizon=10
                    #     )
                    print("Before HP")
                    # print(self.model.lags_future_covariates,"Lags future")
                    # print(self.model.n_estimators,"Estimators")
                    # print(self.model.learning_rate," LR")
                    train_size = 0.8
                    train_points = int(train_size*len(y_time_series))
                    train_series = y_time_series[:train_points]
                    val_series = y_time_series[train_points:]
                    best_model, best_params = self.run_optuna(train_series, future_covariates=X_time_series,val_series=val_series,forecast_horizon=int(0.2*len(y_time_series)), n_trials=10)
                    print(best_model," Best model")
                    print(best_params)
                    # Set the best model
                    self.model = best_model
                    # self.model.model_params=best_params
                    print(" After HP")
                    # print(self.model.lags_future_covariates,"Lags future")
                    # print(self.model.n_estimators,"Estimators")
                    # print(self.model.learning_rate," LR")
                    # print("Best Parameters:", best_params)
                                
                    # self.model=best_model[0]
                    # updated_params = {**current_params, **best_model[0].get_params()}
                    # self.model = self.model.__class__(**updated_params)
                    # self.model.params.setdefault('lags_future_covariates', [0])
                    print(self.model," Self model after hp")
                self.model.fit(y_time_series, future_covariates=X_time_series)
                # self.model.fit(y_time_series)
                
                print(best_model)
            else:
                if self.type == 'darts_lightgbm':
                    X_time_series=None
                    print("Entering not Exo")
                    train_size = 0.8
                    train_points = int(train_size*len(y_time_series))
                    train_series = y_time_series[:train_points]
                    val_series = y_time_series[train_points:]
                    best_model, best_params = self.run_optuna(train_series, future_covariates=X_time_series,val_series=val_series,forecast_horizon=int(0.2*len(y_time_series)), n_trials=10)
                    self.model=best_model
                    print(best_model)
                    # self.model=best_model
                self.model.fit(y_time_series)

    def _predict(self, lookforward: int = 1, X: pd.DataFrame = None) -> np.ndarray:
        if self.isExogenous and self.type == 'darts_autoarima':
            # print("Performing univariate analysis as model is auto_arima.")
            y_ts = self.model.predict(n=lookforward)
        else:
            X = None if self.isExogenous is False else X
            if X is not None and has_argument(self.model.fit, 'future_covariates'):
                X_ts = TimeSeries.from_dataframe(X)
                y_ts = self.model.predict(n=lookforward, future_covariates=X_ts)
                # y_ts = self.model.predict(n=lookforward)
                
                print(self.model)
                print("Exo")
            else:
                y_ts = self.model.predict(n=lookforward)
                print(self.model)
                print("Not Exo")
        return y_ts.values().ravel()




    def objective(self, trial, series, past_covariates, future_covariates, val_series, forecast_horizon):
        # lags = trial.suggest_categorical('lags', [12, 24, 36])
        n_estimators = trial.suggest_categorical('n_estimators', [50, 100,200])
        learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
        lags_future_covariates = trial.suggest_categorical('lags_future_covariates', [[12], [24]])

        # self.model.lags = lags
        # self.model.lags_future_covariates = lags_future_covariates
        self.model.model.n_estimators = n_estimators
        self.model.model.learning_rate = learning_rate
        
        self.model.fit(series, past_covariates=past_covariates, future_covariates=future_covariates)

        val_prediction = self.model.predict(n=forecast_horizon, future_covariates=future_covariates)
        # Calculate MSE
        error = mse(val_series, val_prediction)
        return error

    def run_optuna(self, series, past_covariates=None, future_covariates=None, val_series=None, forecast_horizon=0, n_trials=100):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, series, past_covariates, future_covariates, val_series, forecast_horizon), n_trials=n_trials)

        best_params = study.best_params
        # self.model.lags = best_params['lags']
        # self.model.lags_future_covariates = best_params['lags_future_covariates']
        self.model.model.n_estimators = best_params['n_estimators']
        self.model.model.learning_rate = best_params['learning_rate']

        print("Best Parameters:", best_params)
        return self.model, best_params







class NeuralProphetWrapper(AbstractModel):
    """
    Wrapper for neuralProphet models according to the AbstractExternalModel interface.
    """
    def __init__(self, neuralProphet_model, base_model_config, *args, **kwargs):
        self.neuralProphet_model = neuralProphet_model
        self.base_model_config = base_model_config
        super().__init__(*args, **kwargs)

    def _train(self, data: pd.DataFrame, external_base_model_config) -> None:
        model = self.neuralProphet_model.fit(dataset=data, base_model_request=external_base_model_config)
        self.model = model

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None)-> np.ndarray:
        y_ts = self.neuralProphet_model.predict(data=X, model=self.model)
        return y_ts

class MetaModelWA(AbstractModel):
    """
    MetaModel using the weighted average.
    """
    def __init__(self, base_models, *args, **kwargs):
        self.base_models = base_models
        self.models_weights = {}
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None) -> float:
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
                base_scores[model.type] = main_scorer(y_meta, y_pred, y_base)
                print(f"{model.type} {main_scorer.__name__} test score: {base_scores[model.type]}")
                base_predictions.append(y_pred)
                model._train(df_combined, model.base_model_config)  # Refit with full data
                model.train_idx = y.index
            else:
                y_base, y_meta = train_test_split(y, test_size=0.2, shuffle=False)
                # TODO - Need to add X to the train method
                model._train(y_base,X=X_base)
                y_pred = model.predict(lookforward=len(y_meta),X=X_meta)
                base_scores[model.type] = main_scorer(y_meta, y_pred, y_base)
                print(f"{model.type} {main_scorer.__name__} test score: {base_scores[model.type]}")
                base_predictions.append(y_pred)
                model._train(y,X=X)
                model.train_idx = y.index

        # Inverting the scores for calculating weights
        for model in self.base_models:
            base_scores[model.type]=1/(base_scores[model.type]+epsilon)
            # print(model)
            # print(model.isExogenous)


        total_score = sum(base_scores.values())
        self.models_weights = {model.type: base_scores[model.type] / total_score
                               for model in self.base_models}
        # print(self.models_weights)
    def _predict(self, lookforward: int=1, X: pd.DataFrame=None) -> np.ndarray:
        base_predictions = {}#convert 264 to block and add x variable initiation like train method
        for model in self.base_models:
            if model.isExternalModel():
                base_predictions[model.type] = model.predict(lookforward, X)
            else:
                base_predictions[model.type] = model.predict(lookforward,X)


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
                base_scores[model.type] = main_scorer(y_meta, y_pred, y_base)
                print(f"{model.type} {main_scorer.__name__} test score: {base_scores[model.type]}")
                base_predictions.append(y_pred)
                model._train(df_combined, model.base_model_config)  # Refit with full data
                model.train_idx = y.index
            else:
                y_base, y_meta = train_test_split(y, test_size=0.2, shuffle=False)
                model._train(y_base,X=X_base)
                y_pred = model.predict(len(y_meta),X=X_meta)
                base_scores[model.type] = main_scorer(y_meta, y_pred, y_base)
                print(f"{model.type} {main_scorer.__name__} test score: {base_scores[model.type]}")
                base_predictions.append(y_pred)
                model._train(y,X=X)
                model.train_idx = y.index

        num_models = len(self.base_models)
        self.models_weights = {model.type: 1/num_models
                               for model in self.base_models}

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None) -> np.ndarray:
        base_predictions = {}#convert 264 to block and add x variable initiation like train method
        for model in self.base_models:
            if model.isExternalModel():
                base_predictions[model.type] = model.predict(lookforward, X)
            else:
                base_predictions[model.type] = model.predict(lookforward,X)


        #base_predictions = {model.type: model.predict(lookforward) for model in self.base_models} converted to 261-265
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

    def _train(self, y: pd.Series, X: pd.DataFrame=None) -> None:
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
                test_score = main_scorer(y_meta, y_pred, y_base)
                print(f"{model.type} {main_scorer.__name__} test score: {test_score}")
                model._train(df_combined, model.base_model_config)  # Refit with full data

            else:
                model._train(y_base, X=X_base)
                y_pred = model.predict(lookforward=len(y_meta), X=X_meta)
                base_predictions.append(y_pred)
                test_score = main_scorer(y_meta, y_pred, y_base)
                print(f"{model.type} {main_scorer.__name__} test score: {test_score}")
                model._train(y, X=X)  # Refit with full data



        # Use linear regression to learn the weights
        base_predictions = np.column_stack(base_predictions)
        # base_predictions_df=pd.DataFrame(data=base_predictions, index=y_meta.index).dropna()
        # base_predictions_filtered = base_predictions_df.values
        # y_meta_filtered = y_meta[y_meta.index.isin(base_predictions_df.index)]
        self.regressor.fit(base_predictions, y_meta)

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None) -> np.ndarray:
        base_predictions = [model.predict(lookforward, X) for model in self.base_models]
        base_predictions = np.column_stack(base_predictions)
        meta_predictions = self.regressor.predict(base_predictions)
        return meta_predictions.ravel()


def has_argument(func, arg_name):
    return arg_name in func.__code__.co_varnames