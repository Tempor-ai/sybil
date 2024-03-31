"""
Module to create prediction pipelines.
"""

import os
import uuid
import pandas as pd
import blosc
import base64
import pickle
from darts.models.forecasting.rnn_model import RNNModel
from typing import Union, List
from .ts_utils import get_seasonal_period, smape, mape
from .preprocessor import MinMaxScaler, SimpleImputer, DartsImputer
from .modelwrappers import AbstractModel, StatsforecastWrapper, DartsWrapper,NeuralProphetWrapper, MetaModelWA, MetaModelLR, MetaModelNaive
from .pipeline import ExternalPipeline, Pipeline

SCORERS_DICT = {'smape': smape, 'mape': mape}
META_BASE_MODELS = [
    {'type': 'darts_rnn'},
    {'type': 'darts_lightgbm'},  # TODO: Need to fix use of covariates lags
    {'type': 'darts_autotheta'},
    {'type': 'darts_autoarima'},
    {'type': 'darts_autoets'},
    {'type': 'darts_naive'},
    {'type': 'darts_seasonalnaive'},
    {'type': 'darts_linearregression'},
    # {'type': 'stats_autotheta'},
    # {'type': 'stats_autoarima'},
    # {'type': 'stats_autoets'},
    {'type': 'neuralprophet'}
]
META_PREPROCESSORS = [
    {'type': 'dartsimputer'},
    # {'type': 'simpleimputer', 'params': {'strategy': 'mean'}},
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
            'darts_rnn': ('darts.models.forecasting.rnn_model', 'RNNModel'),
            'darts_naive': ('darts.models', 'NaiveMovingAverage'),
            'darts_seasonalnaive': ('darts.models', 'NaiveSeasonal'),
            'darts_linearregression': ('darts.models', 'LinearRegressionModel'),
            'darts_tbats': ('darts.models', 'TBATS'),
            'neuralprophet': ('models.external.onboard_neuralprophet', 'OnboardNeuralProphet')
        }

        module_name, class_name = models[type]
        return getattr(__import__(module_name, fromlist=[class_name]), class_name)

    @staticmethod
    def create_model(dataset: pd.DataFrame,
                     type: str = 'meta_lr',
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

        if scorers is None: scorers = ['smape', 'mape']
        if params is None: params = {}

        scorer_funcs = [SCORERS_DICT[s] for s in scorers]
        season_length = max(get_seasonal_period(dataset.iloc[:, -1]), 1)

        if type.startswith('meta_'):
            # Meta model case
            base_models_kwargs = params.get('base_models', META_BASE_MODELS)
            params['base_models'] = [ModelFactory.create_model(dataset, **kws)
                                     for kws in base_models_kwargs]
            ModelClass = MetaModelWA if type == 'meta_wa' else (MetaModelNaive if type == 'meta_naive' else MetaModelLR)
            if params.get('preprocessors') is None:
                # If not custom preprocessors, use default META_PREPROCESSORS
                predictor = ModelClass(type=type, scorers=scorer_funcs, **params)
                params.setdefault('preprocessors', META_PREPROCESSORS)
            else:
                # QUICK FIX: Remove custom preprocessors for ModelClass, then add back
                # TO-DO: move preprocessors outside of create_model() and params 
                preprocessors = params['preprocessors'].copy()
                del params['preprocessors']
                predictor = ModelClass(type=type, scorers=scorer_funcs, **params)
                params.setdefault('preprocessors', preprocessors)
        else:
            # Base model case
            if type in ('stats_autotheta', 'stats_autoarima', 'stats_autoets',
                        'darts_autotheta', 'darts_autoarima', 'darts_autoets'):
                params.setdefault('season_length', season_length)
            if type == 'darts_lightgbm':
                if len(dataset.columns)>1 and 'lags_future_covariates' not in params:
                    params.setdefault('lags_future_covariates', [0])
                if 'lags' not in params:
                    params.setdefault('lags', season_length)
            if type == 'darts_rnn':
                params.setdefault('input_chunk_length', season_length)
            if type == 'darts_seasonalnaive':
                params.setdefault('K', season_length)
            if type == 'darts_naive':
                params.setdefault('input_chunk_length', 1)
            if type == 'darts_linearregression':
                params.setdefault('lags', season_length)
            if type == 'darts_tbat':
                params.setdefault('seasonal_periods', [season_length])

            model_class = ModelFactory._get_model_class(type)
            wrapper_class = StatsforecastWrapper if type.startswith('stats_') else DartsWrapper
            model_instance = model_class(**params)
            predictor = wrapper_class(model=model_instance, type=type, scorers=scorer_funcs)
            
            if type == 'darts_rnn':
                predictor = wrapper_class(model=model_instance, type=type, rnn_model="",rnn_model_ckpt="", scorers=scorer_funcs)

            if type == 'neuralprophet': 
                if external_params is None:
                    base_model_config = DEFAULT_NP_BASE_MODELS
                else:
                    base_model_config = external_params
                model_instance = ModelFactory._get_model_class(type=type) # not needed for season_length setted to auto in neuralprophet project, we can add the attribute when neuralprophet expose the config to the user.
                predictor = NeuralProphetWrapper(neuralProphet_model=model_instance, type=type, scorers=scorer_funcs, base_model_config=base_model_config)

        if params and 'preprocessors' in params:
            preprocessors = [ModelFactory._create_preprocessor(pp_name)
                             for pp_name in params['preprocessors']]
            if predictor.isExternalModel():
                return ExternalPipeline(processors=preprocessors, model=predictor)
            else:
                return Pipeline(processors=preprocessors, model=predictor, type=type, scorers=scorer_funcs)
        return predictor

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
        for item in model.model.base_models:
            if item.type == 'darts_rnn':
                uid = str(uuid.uuid4())
                item.model.save(uid)
                rnn_model = base64.b64encode(blosc.compress(open(uid, "rb").read()))
                rnn_model_ckpt = base64.b64encode(blosc.compress(open(uid+".ckpt", "rb").read()))
                item.rnn_model = rnn_model
                item.rnn_model_ckpt = rnn_model_ckpt
                os.remove(uid)
                os.remove(uid+".ckpt")
        output_model = base64.b64encode(blosc.compress(pickle.dumps(model)))
        
        return output_model
    
    @staticmethod
    def load(modelStr):
        model = pickle.loads(blosc.decompress(base64.b64decode(modelStr)))
        for item in model.model.base_models:
            if item.type == 'darts_rnn':
                rnn_model = item.rnn_model
                rnn_model_ckpt = item.rnn_model_ckpt
                uid = str(uuid.uuid4())
                open(uid, "wb").write(blosc.decompress(base64.b64decode(rnn_model)))
                open(uid+".ckpt", "wb").write(blosc.decompress(base64.b64decode(rnn_model_ckpt)))
                rnnload = RNNModel.load(uid)
                item.model = rnnload
                os.remove(uid)
                os.remove(uid+".ckpt")
        return model