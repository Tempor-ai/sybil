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

logger = logging.getLogger(__name__)
router = APIRouter()

DATASET_VALUE = Union[str, int, float]


class Parameters(BaseModel):
    preprocessors: Union[List['Preprocessor'], None] = None
    base_models: Union[List['Model'], None] = None


class Preprocessor(BaseModel):
    type: str
    params: Union[Parameters, None] = None


class Model(BaseModel):
    type: str
    scorers: Union[List[str], None] = None
    # params: Union[Parameters, None] = None
    external_params: Union[dict, None] = None
    params: dict

class TrainRequest(BaseModel):
    data: List[List[DATASET_VALUE]]
    model: Union[Model, None] = None


class Metric(BaseModel):
    type: str
    value: float

class TrainResponse(BaseModel):
    model: str
    type: str
    metrics: Union[List[Metric], None] = None


@router.get('/')
def index():
    pass


@router.post('/train')
async def train(train_request: TrainRequest):
    logger.debug("TrainRequest: %s", train_request)
    dataset = ModelFactory.prepare_dataset(pd.DataFrame(train_request.data))

    # Get optional user specs
    model_info = train_request.model

    # If user did not pass in the model spec
    if model_info is None:
        # Create model objects with the ModelFactory defaults
        model = ModelFactory.create_model(dataset)
    else:
        model_info_json = jsonable_encoder(model_info)
        # Create model objects from the spec user passed in
        model = ModelFactory.create_model(dataset=dataset, **model_info_json)

    # Train model
    training_info = model.train(dataset)

    # Serialize, compress, and finally encode model in base64 ASCII, so it can be sent in JSON
    # output_model = base64.b64encode(blosc.compress(pickle.dumps(training_info['model'])))

    output_model = ModelFactory.save(training_info['model'])

    
    # Build metrics JSON for response
    metrics = []
    for metric_type in training_info["metrics"]:
        metrics.append(Metric(type=metric_type,
                              value=training_info["metrics"][metric_type]
                              ))
    if len(metrics) == 0:
        metrics = None

    # There is dynamacism in the metrics field
    return TrainResponse(model=output_model,
                         type=training_info["type"],
                         metrics=metrics
                         )


class ForecastRequest(BaseModel):
    model: str
    data: Union[List[DATASET_VALUE], List[List[DATASET_VALUE]]]


class ForecastResponse(BaseModel):
    data: List[List[DATASET_VALUE]]


@router.post('/forecast')
async def forecast(forecast_request: ForecastRequest):

    logger.debug("ForecastRequest: %s", forecast_request)

    # Decode model from base64 ASCII, decompress, and final deserialize
    # model = pickle.loads(blosc.decompress(base64.b64decode(forecast_request.model)))
    model = ModelFactory.load(forecast_request.model)
    # TODO Model currently does not support dates, array is converted into number of steps
    if isinstance(forecast_request.data[0], list):  # Forecast request has exogenous variables
        dataset = ModelFactory.prepare_dataset(pd.DataFrame(forecast_request.data))
    else:
        dataset = pd.DataFrame(forecast_request.data)
        dataset[0] = pd.to_datetime(dataset[0])
        dataset.set_index(0, inplace=True)
        
    num_steps = len(forecast_request.data)
    output = model.predict(lookforward=num_steps, X=dataset).reset_index()
    output['index'] = output['index'].apply(lambda x:x.isoformat())
    output = output.values.tolist()

    return ForecastResponse(data=output)

