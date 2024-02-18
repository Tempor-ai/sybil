"""
Module to create prediction pipelines.
"""

import pandas as pd
from typing import Union, List
from .ts_utils import get_seasonal_period, smape, mape
from .preprocessor import MinMaxScaler, SimpleImputer
from .modelwrappers import AbstractModel, StatsforecastWrapper, DartsWrapper,NeuralProphetWrapper, MetaModelWA, MetaModelLR
from .pipeline import Pipeline

SCORERS_DICT = {'smape': smape, 'mape': mape}
DEFAULT_CFG = {'type': 'meta_lr',
               'score': ['smape', 'mape'],
               'params': {
                   'preprocessors': [
                       {'type': 'simpleimputer', 'params': {'strategy': 'mean'}},
                       {'type': 'minmaxscaler'}
                   ],
                   'base_models': [
                       {'type': 'darts_autoets'},
                       {'type': 'darts_autoarima'},
                       {'type': 'darts_autotheta'},
                       {'type': 'stats_autotheta'},
                       {'type': 'stats_autotheta'},
                       {'type': 'neuralprophet'}]}
               }
DEFAULT_BASE_MODELS = [{'type': 'darts_autoets'}, {'type': 'darts_autoarima'}, {'type': 'darts_autotheta'}]

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
    def _get_model_instance(type: str, season_length: int):
        """
        Helper method to instantiate a model based on its type and season_length.

        :param type: The type of model to instantiate.
        :param season_length: Seasonal period of the dataset.
        :return: An instance of the specified model.
        """
        models = {
            'stats_autotheta': ('statsforecast.models', 'AutoTheta'),
            'stats_autoarima': ('statsforecast.models', 'AutoARIMA'),
            'stats_autoets': ('statsforecast.models', 'AutoETS'),
            'darts_autotheta': ('darts.models', 'StatsForecastAutoTheta'),
            'darts_autoarima': ('darts.models', 'StatsForecastAutoARIMA'),
            'darts_autoets': ('darts.models', 'StatsForecastAutoETS'),
            'neuralprophet': ('models.external.onboard_neuralprophet', 'OnboardNeuralProphet')
        }

        module_name, class_name = models[type]
        ModelClass = getattr(__import__(module_name, fromlist=[class_name]), class_name)

        if type in ('neuralprophet'):
            return ModelClass()
        else:
            return ModelClass(season_length=season_length)

    @staticmethod
    def create_model(dataset: pd.DataFrame,
                     type: str = DEFAULT_CFG['type'],
                     scorers: Union[str, List[str]] = DEFAULT_CFG['score'],
                     params: dict = DEFAULT_CFG['params'], external_params: dict = None) -> AbstractModel:
        """
        Create a model of the given type.

        @param dataset: A dataframe containing the dataset with the time column as the first column
        and the target column as the last column.
        @param type: The type of the model to create. Defaults to 'darts_autotheta'.
        @param scorers: A list of scorers to use for evaluating metrics. Defaults to 'mape'.
        @param params: A dictionary containing the model parameters if necessary.
        @return: A model of the given type.
        """

        scorer_funcs = [SCORERS_DICT[s] for s in scorers] if (isinstance(scorers, list) and len(scorers) > 0) \
            else DEFAULT_CFG['score']
        season_length = get_seasonal_period(dataset.iloc[:, -1])

        if type in ('stats_autotheta', 'stats_autoarima', 'stats_autoets'):
            model_instance = ModelFactory._get_model_instance(type, season_length)
            predictor = StatsforecastWrapper(stats_model=model_instance, type=type, scorers=scorer_funcs)
        elif type in ('darts_autotheta', 'darts_autoarima', 'darts_autoets'):
            model_instance = ModelFactory._get_model_instance(type, season_length)
            predictor = DartsWrapper(darts_model=model_instance, type=type, scorers=scorer_funcs)
        elif type in ('neuralprophet'): #TODO add default value for the external_params ?
            season_length = None
            if external_params is None:
                base_model_config = DEFAULT_NP_BASE_MODELS
            else:
                base_model_config = external_params
            model_instance = ModelFactory._get_model_instance(type=type, season_length=season_length) # not needed for season_length setted to auto in neuralprophet project, we can add the attribute when neuralprophet expose the config to the user.
            predictor = NeuralProphetWrapper(neuralProphet_model=model_instance, type=type, scorers=scorer_funcs, base_model_config=base_model_config)
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
