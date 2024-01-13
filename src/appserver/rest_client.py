import json
import yaml
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import APIRouter
from pydantic import BaseModel

import pickle
import pandas as pd
from models.modelfactory import ModelFactory
from typing import Union, List
import blosc
import base64
import logging
from fastapi.encoders import jsonable_encoder

class rest_client:
    def run(self):
        # Add your training logic here
        file_name = 'https://github.com/ourownstory/neuralprophet-data/raw/main/datasets/air_passengers.csv'

        train_df = pd.read_csv(file_name)

        time_col = train_df.columns[0]
        target_col = train_df.columns[-1]
        train_df[time_col] = train_df[time_col].astype(str)
        train_df[target_col] = train_df[target_col].astype(float)

        # from the dataset in routes.py, same as train_data, no need to read again from the file. read directly from the reqeust
        train_data = [] 
        for value in train_df.values:
            train_data.append(list(value))

        # from sybil request ?
        model_request2 = {
            "params": {
            "changepoints_range": 0.2,
            "epochs": 2,
            "growth": "off"
            },
            "metrics": [],
            "type": "neuralprophet",
        }

        # for now use this model requs
        model_request = {
            "type": "neuralprophet",
            "params": {
                "changepoints_range": 0.2,
                "epochs": 2,
                "growth": "off",
                "metrics": [],
            }
        }
            
# class ModelPayload(BaseModel):
#     type: Literal["neuralprophet"] | None = "neuralprophet"
#     metrics: List[Literal["mae", "rmse", "mse"]] = ["mae", "rmse"]
#     params: ParametersPayload = Field(default_factory=ParametersPayload)
        



# class ParametersPayload(BaseModel):
#     # Trend
#     growth: np_types.GrowthMode = "linear"
#     changepoints: Optional[list] = None
#     n_changepoints: int = 10
#     changepoints_range: float = 0.8
#     trend_reg: float = 0
#     trend_reg_threshold: Optional[Union[bool, float]] = False
#     trend_global_local: TrendGlobalLocal = "global"

#     # Seasonality
#     yearly_seasonality: np_types.SeasonalityArgument = "auto"
#     weekly_seasonality: np_types.SeasonalityArgument = "auto"
#     daily_seasonality: np_types.SeasonalityArgument = "auto"
#     seasonality_mode: np_types.SeasonalityMode = "additive"
#     seasonality_reg: float = 0
#     season_global_local: np_types.SeasonGlobalLocalMode = "global"

#     # Forecasts
#     n_forecasts: int = 1
#     n_lags: int = 0

#     # Auto regression
#     ar_layers: Optional[list] = []
#     ar_reg: Optional[float] = None

#     lagged_reg_layers: Optional[list] = []

#     # Training
#     learning_rate: Optional[float] = None
#     epochs: Optional[int] = None
#     batch_size: Optional[int] = None
#     loss_func: str = "Huber"
#     optimizer: str = "AdamW"
#     newer_samples_weight: float = 2
#     newer_samples_start: float = 0.0

#     # Probabilistic forecasting
#     quantiles: List[float] = []

#     # Impute values
#     impute_missing: bool = True
#     impute_linear: int = 10
#     impute_rolling: int = 10
#     drop_missing: bool = False

#     # Metrics
#     collect_metrics: Union[bool, list, dict] = True

#     # Normalization
#     normalize: np_types.NormalizeMode = "auto"
#     global_normalization: bool = False
#     global_time_normalization: bool = True
#     unknown_data_normalization: bool = False

#     # Extra training parameters
#     accelerator: Optional[str] = None
#     trainer_config: dict = {}
#     prediction_frequency: Optional[dict] = None

#     # Regressors
#     lagged_regressors: Optional[List[LaggedRegressor]] = []
#     future_regressors: Optional[List[FutureRegressor]] = []

#     # Events
#     country_holidays: Optional[List[CountryHoliday]] = []
#     events: Optional[List[Event]] = []

# model_request = {
#     'type': 'meta_wa',  # 'meta_wa'
#     'scorers': ['smape', 'mape'],
#     'params': {
#         'preprocessors': [
#             {'type': 'simpleimputer', 'params': {'strategy': 'mean'}},
#             {'type': 'minmaxscaler'},
#         ],
#         'base_models': [
#             {'type': 'darts_autoets'},
#             {'type': 'darts_autoarima'},
#             {'type': 'darts_autotheta'},
#             {'type': 'stats_autotheta'},
#             {'type': 'neuralprophet', 'params': {'changepoints_range':0.2, 'epochs':2, 'growth':'off'}, 'metrics':[] }
#         ],
#     },
# }
        


        api_json = {
            'data': train_data,
            'model': model_request  # (optional) can be commented out
        }

        # from config file
        # URL to our SYBIL AWS service
        protocol = "http"
        host = "localhost"
        port = 8001
        endpoint = 'train'

        url = '%s://%s:%s/%s' % (protocol, host, str(port), endpoint)

        response = requests.post(url, json=api_json)
        print(response)
            
    def forecast(self):
        # Add your forecasting logic here
        pass


