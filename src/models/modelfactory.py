"""
Module to create prediction pipelines.
"""

import pandas as pd
from typing import Union, List
from .ts_utils import get_seasonal_period, smape, mape
from .preprocessor import MinMaxScaler, SimpleImputer
from .modelwrappers import AbstractModel, StatsforecastWrapper, DartsWrapper, MetaModelWA, MetaModelLR
from .pipeline import Pipeline

SCORERS_DICT = {'smape': smape, 'mape': mape}
DEFAULT_BASE_MODELS = [{'type': 'darts_lightgbm'},
                       {'type': 'darts_autoets'},
                       {'type': 'darts_autoarima'},
                       {'type': 'darts_autotheta'},
                       {'type': 'stats_autotheta'}]
DEFAULT_CFG = {'type': 'meta_lr',
               'score': ['smape', 'mape'],
               'params': {
                   'preprocessors': [
                       {'type': 'simpleimputer', 'params': {'strategy': 'mean'}},
                       {'type': 'minmaxscaler'}
                   ],
                   'base_models': DEFAULT_BASE_MODELS}
               }


class ModelFactory:
    """
    Factory class for creating models.
    """

    @staticmethod
    def _get_model_class(type: str):
        """
        Helper method to import and return the class of a model based on its type.

        :param type: The type of model to instantiate.
        :return: An class of the specified model.
        """
        models = {
            'stats_autotheta': ('statsforecast.models', 'AutoTheta'),
            'stats_autoarima': ('statsforecast.models', 'AutoARIMA'),
            'stats_autoets': ('statsforecast.models', 'AutoETS'),
            'darts_autotheta': ('darts.models', 'StatsForecastAutoTheta'),
            'darts_autoarima': ('darts.models', 'StatsForecastAutoARIMA'),
            'darts_autoets': ('darts.models', 'StatsForecastAutoETS'),
            'darts_lightgbm': ('darts.models.forecasting.lgbm', 'LightGBMModel')
        }

        module_name, class_name = models[type]
        return getattr(__import__(module_name, fromlist=[class_name]), class_name)


    @staticmethod
    def create_model(dataset: pd.DataFrame,
                     type: str = DEFAULT_CFG['type'],
                     score: Union[str, List[str]] = DEFAULT_CFG['score'],
                     params: dict = DEFAULT_CFG['params']) -> AbstractModel:
        """
        Create a model of the given type.

        @param dataset: A dataframe containing the dataset with the time column as the first column
        and the target column as the last column.
        @param type: The type of the model to create. Defaults to 'darts_autotheta'.
        @param score: A list of scorers to use for evaluation. Defaults to 'mape'.
        @param params: A dictionary containing the model parameters if necessary.
        @return: A model of the given type.
        """

        scorer_funcs = [SCORERS_DICT[s] for s in score] if (isinstance(score, list) and len(score) > 0) \
            else DEFAULT_CFG['score']
        season_length = get_seasonal_period(dataset["value"])

        if type in ('stats_autotheta', 'stats_autoarima', 'stats_autoets'):
            model_class = ModelFactory._get_model_class(type)
            model_instance = model_class(season_length=season_length)
            predictor = StatsforecastWrapper(stats_model=model_instance, type=type, scorers=scorer_funcs)
        elif type in ('darts_autotheta', 'darts_autoarima', 'darts_autoets'):
            model_class = ModelFactory._get_model_class(type)
            model_instance = model_class(season_length=season_length)
            predictor = DartsWrapper(darts_model=model_instance, type=type, scorers=scorer_funcs)
        elif type == 'darts_lightgbm':
            model_class = ModelFactory._get_model_class(type)
            lags = max(season_length, 1)
            model_instance = model_class(lags=lags,
                                         #lags_future_covariates=(0, lags)  TODO: Need to fix use of covariates lags
                                         )
            predictor = DartsWrapper(darts_model=model_instance, type=type, scorers=scorer_funcs)
        elif "meta_" in type:
            base_models_kwargs = DEFAULT_BASE_MODELS if not params else params['base_models']
            base_models = [ModelFactory.create_model(dataset, **model_kwargs) for model_kwargs in base_models_kwargs]
            ModelClass = MetaModelWA if type == 'meta_wa' else MetaModelLR
            predictor = ModelClass(models=base_models, type=type, scorers=scorer_funcs)
        else:
            raise ValueError(f'Unknown model type: {type}')

        if params and 'preprocessors' in params:
            preprocessors = [ModelFactory._create_preprocessor(preprocessor) for preprocessor in params['preprocessors']]
            return Pipeline(processors=preprocessors, model=predictor, type=type, scorers=scorer_funcs)
        return predictor

    @staticmethod
    def _create_preprocessor(preprocessor):
        """
        Helper method to instantiate a preprocessor based on its type.

        :param preprocessor: Dictionary with type and optional parameters for the preprocessor.
        :return: An instance of the specified preprocessor.
        """
        preprocessors_map = {
            'minmaxscaler': MinMaxScaler,
            'simpleimputer': SimpleImputer
        }
        type = preprocessor.get('type')
        kwargs = preprocessor.get('param', {})
        if type in preprocessors_map:
            return preprocessors_map.get(type)(**kwargs)
        else:
            raise ValueError(f'Unknown preprocessor type: {type}')

    @staticmethod
    def prepare_dataset(dataset: pd.DataFrame, time_col=0, value_col=-1) -> pd.DataFrame:
        """
        Prepare the dataset for training or prediction.

        :param dataset: Dataframe containing the dataset.
        :param time_col: Index or name of the column identifying the time component. Defaults to 0.
        :param value_col: Index or name of the column identifying the value component. Defaults to -1.
        :return: Dataframe with the time and value columns renamed to 'datetime' and 'value' respectively.
        """
        clean_dataset = dataset.rename(columns={dataset.columns[time_col]: 'datetime', dataset.columns[value_col]: 'value'})
        if clean_dataset['datetime'].dtype == object:
            clean_dataset['datetime'] = pd.to_datetime(clean_dataset['datetime'], infer_datetime_format=True)
        clean_dataset = clean_dataset.set_index("datetime").astype(float)
        return clean_dataset
