from fastapi import APIRouter
from pydantic import BaseModel

import pickle
import pandas as pd
from models.modelwrappers import AbstractModel, ModelFactory
from typing import Union, List

router = APIRouter()

class Model(BaseModel):
    type: str
    score: list
    param: None

class Parameters(BaseModel):
    models: List[Model]


class Model(BaseModel):
    type: str
    score: list
    param: Parameters


class TrainRequest(BaseModel):
    data: str #TODO Convert this to list of lists
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

    dataset = AbstractModel.prepare_dataset(pd.read_json(train_request.data))
    model_info = train_request.model
    model = ModelFactory.create_model(dataset, model_info=model_info)

    # Train model
    training_info = model.train(dataset)

    # There is dynamacism in the evaluation field
    return TrainResponse(model=pickle.dumps(training_info['model']),
                         type=training_info["type"],
                         evaluation=Evaluation(mape=training_info["evaluation"]["mape"],
                                               smape=training_info["evaluation"]["smap"]))


@router.post('/forecast')
async def forecast():
    pass