from fastapi import APIRouter
from pydantic import BaseModel

import pickle
import pandas as pd
from .models.modelwrappers import AbstractModel
from .models.modelfactory import ModelFactory
from typing import Union, List
import blosc
import base64
import numpy as np
import logging
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger(__name__)
router = APIRouter()


class Parameters(BaseModel):
    base_models: Union[List['Model'], None] = None


class Model(BaseModel):
    type: str
    scorers: Union[List[str], None] = None
    params: Union[Parameters, None] = None


class TrainRequest(BaseModel):
    data: List[List[Union[str, int, float]]]
    model: Union[Model, None] = None


class Evaluation(BaseModel):
    type: str
    value: float


class TrainResponse(BaseModel):
    model: str
    type: str
    evaluation: Union[List[Evaluation], None] = None


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
        model = ModelFactory.create_model(dataset=dataset,
                                          type=model_info_json["type"],
                                          scorers=model_info_json["scorers"],
                                          params=model_info_json["params"])      

    # Train model
    training_info = model.train(dataset)

    # Serialize, compress, and finally encode model in base64 ASCII, so it can be sent in JSON
    output_model = base64.b64encode(blosc.compress(pickle.dumps(training_info['model'])))

    # Build evaluation JSON for response
    evaluation = []
    for evaluation_type in training_info["evaluation"]:
        evaluation.append(Evaluation(type=evaluation_type,
                                     value=training_info["evaluation"][evaluation_type]
                                     ))
    if len(evaluation) == 0:
        evaluation = None

    # There is dynamacism in the evaluation field
    return TrainResponse(model=output_model,
                         type=training_info["type"],
                         evaluation=evaluation
                         )


class ForecastRequest(BaseModel):
    model: str
    predicts: List[Union[int, str]]


class ForecastResponse(BaseModel):
    data: List[List[Union[int, str, float]]]


@router.post('/forecast')
async def forecast(forecast_request: ForecastRequest):

    logger.debug("ForecastRequest: %s", forecast_request)

    # Decode model from base64 ASCII, decompress, and final deserialize
    model = pickle.loads(blosc.decompress(base64.b64decode(forecast_request.model)))

    # TODO Model currently does not support dates, array is converted into number of steps
    num_steps = len(forecast_request.predicts)

    output = model.predict(lookforward=num_steps).reset_index()
    output['index'] = output['index'].apply(lambda x:x.isoformat())
    output = output.values.tolist()
    print(output)

    return ForecastResponse(data=output)
