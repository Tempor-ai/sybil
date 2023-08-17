from fastapi import APIRouter
from pydantic import BaseModel

import pickle
import pandas as pd
from models.modelwrappers import AbstractModel, ModelFactory
from typing import Union, List
import blosc
import base64
import numpy as np
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# Train response
class Model(BaseModel):
    type: str
    score: list
    param: None


class Parameters(BaseModel):
    models: List[Model]


class Model(BaseModel):
    type: str
    score: List[str]
    param: Union[Parameters, None] = None


class TrainRequest(BaseModel):
    data: List[List[Union[str, int, float]]]
    model: Union[Model, None] = None


class Evaluation(BaseModel):
    smape: float
    mape: float


class TrainResponse(BaseModel):
    model: str
    type: str
    evaluation: Evaluation


@router.get('/')
def index():
    pass


@router.post('/train')
async def train(train_request: TrainRequest):

    dataset = AbstractModel.prepare_dataset(pd.DataFrame(train_request.data))
    model_info = train_request.model
    model = ModelFactory.create_model(dataset, model_info=model_info)

    # Train model
    training_info = model.train(dataset)
    output_model = base64.b64encode(blosc.compress(pickle.dumps(training_info['model'])))

    # There is dynamacism in the evaluation field
    return TrainResponse(model=output_model,
                         type=training_info["type"],
                         evaluation=Evaluation(mape=training_info["evaluation"]["mape"],
                                               smape=training_info["evaluation"]["smape"]))


class ForecastRequest(BaseModel):
    model: str
    predicts: List[Union[int, str]]


class ForecastResponse(BaseModel):
    data: List[List[Union[int, str, float]]]


@router.post('/forecast')
async def forecast(forecast_request: ForecastRequest):

    model = pickle.loads(blosc.decompress(base64.b64decode(forecast_request.model)))
    output = model.predict(np.array(forecast_request.predicts))

    logger.error(output)
