"""
Module to create prediction pipelines.
"""

import pandas as pd
from typing import Union, List

from .ts_utils import get_seasonal_period, smape, mape
from .transformers import MinMaxScaler
from .modelwrappers import AbstractModel, StatsforecastWrapper, DartsWrapper, MetaModelWA, MetaModelLR
from .pipeline import Pipeline


SCORERS_DICT = {'smape': smape, 'mape': mape}
DEFAULT_BASE_MODELS = [{'type': 'darts_autoets'},
                       {'type': 'darts_autoarima'},
                       {'type': 'darts_autotheta'}]


class ModelFactory():
    """
    Factory class for creating models.
    """
    @staticmethod
    def create_model(dataset: pd.DataFrame,
                     type: str = 'darts_autotheta',
                     model_params: dict = None,
                     scorers: Union[str, List[str]] = 'mape') -> AbstractModel:
        """
        Create a model of the given type.

        @param dataset: A dataframe containing the dataset with the time column as the first column
        and the target column as the last column.
        @param type: The type of the model to create. Defaults to 'darts_autotheta' if None.
        @param model_params: A dictionary containing the model parameters if necessary.
        @param scorers: A list of scorers to use for evaluation. Defaults to MAPE if None.
        @return: A model of the given type.
        """
        scorer_func = [SCORERS_DICT[s] for s in scorers] if isinstance(scorers, list) \
            else SCORERS_DICT[scorers]
        season_length = get_seasonal_period(dataset["value"])
        if type == 'stats_autotheta':
            from statsforecast.models import AutoTheta
            model = StatsforecastWrapper(model=AutoTheta(season_length=season_length),
                                         scorers=scorer_func,
                                         type=type)
        elif type == 'stats_autoarima':
            from statsforecast.models import AutoARIMA
            model = StatsforecastWrapper(model=AutoARIMA(season_length=season_length),
                                         scorers=scorer_func,
                                         type=type)
        elif type == 'stats_autoets':
            from statsforecast.models import AutoETS
            model = StatsforecastWrapper(model=AutoETS(season_length=season_length),
                                         scorers=scorer_func,
                                         type=type)
        elif type == 'darts_autotheta':
            from darts.models import StatsForecastAutoTheta
            model = DartsWrapper(model=StatsForecastAutoTheta(season_length=season_length),
                                 scorers=scorer_func,
                                 type=type)
        elif type == 'darts_autoarima':
            from darts.models import StatsForecastAutoARIMA
            model = DartsWrapper(model=StatsForecastAutoARIMA(season_length=season_length),
                                 scorers=scorer_func,
                                 type=type)
        elif type == 'darts_autoets':
            from darts.models import StatsForecastAutoETS
            model = DartsWrapper(model=StatsForecastAutoETS(season_length=season_length),
                                 scorers=scorer_func,
                                 type=type)
        elif "meta_" in type:
            # TODO: Pass also base models parameters
            base_models_kwargs = DEFAULT_BASE_MODELS if model_params is None else model_params['base_models']
            base_models = [ModelFactory.create_model(dataset, **model_kwargs)
                           for model_kwargs in base_models_kwargs]
            if type == 'meta_wa':
                model = MetaModelWA(models=base_models,
                                    scorers=scorer_func,
                                    type=type)
            elif type == 'meta_lr':
                model = MetaModelLR(models=base_models,
                                    scorers=scorer_func,
                                    type=type)
        elif "test" in type:
            base_model = ModelFactory.create_model(dataset)
            transformers = [MinMaxScaler()]
            model = Pipeline(transformers=transformers,
                             model=base_model,
                             scorers=scorer_func,
                             type=type)
        else:
            raise ValueError(f'Unknown model type: {type}')

        return model

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
        return clean_dataset
