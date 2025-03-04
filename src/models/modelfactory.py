"""
Module to create prediction pipelines.
"""

import os
import uuid
import pandas as pd
import blosc
import base64
import pickle
from typing import Union, List
from .ts_utils import get_seasonal_period, smape, mape, mase
from .preprocessor import MinMaxScaler, SimpleImputer, DartsImputer
from .modelwrappers import AbstractModel, StatsforecastWrapper, DartsWrapper, NeuralProphetWrapper, DeepSYBILWrapper, \
                           MetaModelWA, MetaModelLR, MetaModelNaive
from .pipeline import ExternalPipeline, Pipeline

SCORERS_DICT = {'smape': smape, 'mape': mape, 'mase': mase}
DEFAULT_BASE_MODELS = [
    {'type': 'darts_autotheta'},
    {'type': 'darts_autoarima'},
    {'type': 'darts_autoets'},
    {'type': 'darts_naive'},
    {'type': 'darts_seasonalnaive'}
]
META_PREPROCESSORS = [
    {'type': 'dartsimputer'},
    {'type': 'minmaxscaler'}
]
DEFAULT_NP_BASE_MODELS = {
    "params": {
      "changepoints_range": 0.2,
      "epochs": 2,
      "growth": "off"
    },
    "metrics": [],
    "type": "neuralprophet",
}
DEFAULT_DSYBIL_BASE_MODELS = {
    'type': 'meta_wa',  # 'meta_naive', 'meta_wa'
    'scorers': ['mase', 'smape'],
    'params': {
        'preprocessors': [
            # {'type': 'dartsimputer'},
            # {'type': 'simpleimputer', 'params': {'strategy': 'mean'}},
            {'type': 'minmaxscaler'},
        ],
        'base_models': [
            # {'type': 'darts_rnn',
            #  'params': {
            #      'model': 'LSTM',
            #      'hidden_dim': 10,
            #      'n_rnn_layers': 3
            # }},
            # {'type': 'darts_tcn',
            #  'params': {
            #      'output_chunk_length': 52,
            #      'input_chunk_length': 104,                 
            #      'n_epochs': 20,
            # }},
            # {'type': 'darts_rnn',
            #  'params': {
            #      'model': 'LSTM',
            #      'hidden_dim': 10,
            #      'n_rnn_layers': 3
            # }},
            {'type': 'darts_nlinear'},
            {'type': 'darts_dlinear'},
            # {'type': 'darts_blockrnn'},
            # {'type': 'darts_tsmixer',
            #  'params': {
            #      'output_chunk_length': 52,
            #      'input_chunk_length': 104,                 
            #      'n_epochs': 20,
            # }},
            #  {'type': 'darts_tide'}
        ],
    },
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
        :return: A class of the specified model.
        """
        models = {
            'stats_autotheta': ('statsforecast.models', 'AutoTheta'),
            'stats_autoarima': ('statsforecast.models', 'AutoARIMA'),
            'stats_autoets': ('statsforecast.models', 'AutoETS'),
            'darts_autotheta': ('darts.models', 'StatsForecastAutoTheta'),
            'darts_autoarima': ('darts.models', 'StatsForecastAutoARIMA'),
            'darts_autoets': ('darts.models', 'StatsForecastAutoETS'),
            'darts_lightgbm': ('darts.models.forecasting.lgbm', 'LightGBMModel'),
            'darts_naive': ('darts.models', 'NaiveMovingAverage'),
            'darts_seasonalnaive': ('darts.models', 'NaiveSeasonal'),
            'darts_linearregression': ('darts.models', 'LinearRegressionModel'),
            'darts_tbats': ('darts.models', 'TBATS'),
            'neuralprophet': ('models.external.rest_models', 'OnboardNeuralProphet'),
            'darts_autoces': ('darts.models', 'StatsForecastAutoCES'),
            'darts_kalman': ('darts.models', 'KalmanForecaster'),
            'darts_catboost': ('darts.models', 'CatBoostModel'),
            'deepsybil': ('models.external.rest_models', 'DeepSYBIL'),
        }

        module_name, class_name = models[type]
        modelClass = getattr(__import__(module_name, fromlist=[class_name]), class_name)
        
        return modelClass

    @staticmethod
    def _create_meta_model(params: dict, type: str, dataset: pd.DataFrame, scorer_funcs: List[str], is_exogenous: bool):
        """
        Helper method to create the predictor for a meta model.
        """
        base_models_kwargs = params.get('base_models', DEFAULT_BASE_MODELS)
        params['base_models'] = [ModelFactory.create_model(dataset, **kws)
                                    for kws in base_models_kwargs]
        ModelClass = MetaModelWA if type == 'meta_wa' else (MetaModelNaive if type == 'meta_naive' else MetaModelLR)
        if params.get('preprocessors') is None:
            # If not custom preprocessors, use default META_PREPROCESSORS
            
            predictor = ModelClass(type=type, scorers=scorer_funcs, **params, is_exogenous=is_exogenous)
            params.setdefault('preprocessors', META_PREPROCESSORS)
        else:
            # Remove custom preprocessors for ModelClass, then add back
            preprocessors = params['preprocessors'].copy()
            del params['preprocessors']
            
            predictor = ModelClass(type=type, scorers=scorer_funcs, **params, is_exogenous=is_exogenous)
            params.setdefault('preprocessors', preprocessors)
        return params, predictor

    @staticmethod
    def _create_base_model(params: dict, type: str, dataset: pd.DataFrame, scorer_funcs: List[str], season_length:int, is_exogenous: bool, external_params: dict = None):
        """
        Helper method to create the predictor for a base model.
        """
        if type in ('stats_autotheta', 'stats_autoarima', 'stats_autoets',
                    'darts_autotheta', 'darts_autoarima', 'darts_autoets', 'darts_autoces'):
            params.setdefault('season_length', season_length)
        if type in ('darts_lightgbm','darts_catboost'):
            if len(dataset.columns)>1 and 'lags_future_covariates' not in params:
                params.setdefault('lags_future_covariates', [0])
            if 'lags' not in params:
                params.setdefault('lags', season_length)
        if type == 'darts_seasonalnaive':
            params.setdefault('K', season_length)
        if type == 'darts_naive':
            params.setdefault('input_chunk_length', 1)
        if type == 'darts_linearregression':
            params.setdefault('lags', season_length)
        if type == 'darts_tbats':
            params.setdefault('seasonal_periods', [season_length])
        if type == 'darts_kalman':
            params.setdefault('dim_x', season_length)

        predictor = None
        if type.startswith('stats_') or type.startswith('darts_'):
            model_class = ModelFactory._get_model_class(type)
            wrapper_class = StatsforecastWrapper if type.startswith('stats_') else DartsWrapper
            model_instance = model_class(**params)
            predictor = wrapper_class(
                model=model_instance,
                type=type,
                scorers=scorer_funcs,
                is_exogenous=is_exogenous
            )    
        elif type in 'neuralprophet': 
            if external_params is None:
                base_model_config = DEFAULT_NP_BASE_MODELS
            else:
                base_model_config = external_params
                model_instance = ModelFactory._get_model_class(type=type) # not needed for season_length setted to auto in neuralprophet project, we can add the attribute when neuralprophet expose the config to the user.
                predictor = NeuralProphetWrapper(
                    neuralprophet_model=model_instance,
                    type=type,
                    scorers=scorer_funcs,
                    base_model_config=base_model_config,
                    is_exogenous=is_exogenous
                )
        elif type == 'deepsybil':  # TO-DO
            if external_params is None:
                base_model_config = DEFAULT_DSYBIL_BASE_MODELS  # TO-DO
            else:
                base_model_config = external_params
                model_instance = ModelFactory._get_model_class(type=type)
                predictor = DeepSYBILWrapper(
                    deepsybil_model=model_instance,
                    type=type,
                    scorers=scorer_funcs,
                    base_model_config=base_model_config,
                )
        else:
            raise ValueError(f'Unknown model type: {type}')

        return params, predictor

    @staticmethod
    def _create_preprocessor(preprocessor: dict):
        """
        Helper method to instantiate a preprocessor based on its type.

        :param preprocessor: Dictionary with type and optional parameters for the preprocessor.
        :return: An instance of the specified preprocessor.
        """
        preprocessors_map = {'minmaxscaler': MinMaxScaler,
                             'simpleimputer': SimpleImputer,
                             'dartsimputer': DartsImputer}
        type = preprocessor.get('type')
        kwargs = preprocessor.get('params', {})
        if type in preprocessors_map:
            return preprocessors_map.get(type)(**kwargs)
        else:
            raise ValueError(f'Unknown preprocessor type: {type}')

    @staticmethod
    def prepare_dataset(dataset: pd.DataFrame, time_col=0) -> pd.DataFrame:
        """
        Prepare the dataset for training or prediction.

        :param dataset: Dataframe containing the dataset.
        :param time_col: Index or name of the column identifying the time component. Defaults to 0.
        :return: Dataframe with the time column formatted to datetime, set as index and the
        other columns cast as floats.
        """
        clean_dataset = dataset.copy()
        time_col_name = clean_dataset.columns[time_col]
        if clean_dataset[time_col_name].dtype == object:
            clean_dataset[time_col_name] = pd.to_datetime(clean_dataset[time_col_name],
                                                          infer_datetime_format=True)
        clean_dataset = clean_dataset.set_index(time_col_name).astype(float)
        return clean_dataset
    
    @staticmethod
    def save(model) -> str:
        output_model = base64.b64encode(blosc.compress(pickle.dumps(model)))
        
        return output_model
    
    @staticmethod
    def load(modelStr):
        model = pickle.loads(blosc.decompress(base64.b64decode(modelStr)))
        return model
    
    @staticmethod
    def create_model(dataset: pd.DataFrame,
                     type: str = 'meta_wa',
                     scorers: Union[str, List[str]] = None,
                     params: dict = None, external_params: dict = None) -> AbstractModel:
        """
        Create a model of the given type.

        @param dataset: A dataframe containing the dataset with the time column as the first column
        and the target column as the last column.
        @param type: The type of the model to create. Defaults to 'darts_autotheta'.
        @param scorers: A list of scorers to use for evaluating metrics. Defaults to 'mape'.
        @param params: A dictionary containing the model parameters if necessary.
        @return: A model of the given type.
        """

        if scorers is None: scorers = ['mase', 'smape']
        if params is None: params = {}

        if dataset.shape[1] > 1:  # Exogenous variables are present
            is_exogenous = True
        else:
            is_exogenous = False

        scorer_funcs = [SCORERS_DICT[s] for s in scorers]
        season_length = max(get_seasonal_period(dataset.iloc[:, -1]), 1)

        if type.startswith('meta_'):
            # Meta model case
            params, predictor = ModelFactory._create_meta_model(params,type, dataset, scorer_funcs, is_exogenous)
        else:
            # Base model case
            params, predictor = ModelFactory._create_base_model(params, type, dataset, scorer_funcs, season_length, is_exogenous, external_params)
        
        # Base model case
        if params and 'preprocessors' in params:
            # predictor = ModelFactory._create_preprocessor_pipeline(params, scorer_funcs, predictor)
            preprocessors = [ModelFactory._create_preprocessor(pp_name)
                             for pp_name in params['preprocessors']]
            if predictor.isExternalModel():
                return ExternalPipeline(processors=preprocessors, model=predictor)
            else:
                return Pipeline(processors=preprocessors, model=predictor, type=type, scorers=scorer_funcs)
        return predictor