
"""
Routes class for fast api
"""
import logging
import pandas as pd
from typing import Union, List
from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
import yaml
import requests

from models.modelfactory import ModelFactory


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

class Metric(BaseModel):
    type: str
    value: float

class TrainRequest(BaseModel):
    data: List[List[DATASET_VALUE]]
    model: Union[Model, None] = None

class TrainResponse(BaseModel):
    model: str
    type: str
    metrics: Union[List[Metric], None] = None

class ForecastRequest(BaseModel):
    model: str
    data: Union[List[DATASET_VALUE], List[List[DATASET_VALUE]]]

class ForecastResponse(BaseModel):
    data: List[List[DATASET_VALUE]]


#####################
# Routes section
#####################
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

## MAUQ API CALL
    yact=training_info['yact']
    ypred=training_info['ypred']
    mauq_df =pd.DataFrame({
    'y_act': yact.values,
    'y_pred': ypred.values
    })

    api_json = {
        'data': mauq_df.values.tolist(),
        'problem_type': 'regression',
        'confidence_level': 0.7,
        'output_type': 'data'
    }
    print(api_json)

    # with open('mauq_url.yaml', 'r') as file:
    #     url_dict = yaml.safe_load(file)
    # URL to our MAUQ AWS service
    protocol = 'http'
    host = 'localhost'
    port = 8001
    endpoint = 'quantify-uncertainty'

    url = '%s://%s:%s/%s' % (protocol, host, str(port), endpoint)
    response = requests.post(url, json=api_json)
    print(response.json())
    

    # There is dynamacism in the metrics field
    return TrainResponse(model=output_model,
                         type=training_info["type"],
                         metrics=metrics
                         )

@router.post('/forecast')
async def forecast(forecast_request: ForecastRequest):

    logger.debug("ForecastRequest: %s", forecast_request)

    # Decode model from base64 ASCII, decompress, and final deserialize
    # model = pickle.loads(blosc.decompress(base64.b64decode(forecast_request.model)))
    model = ModelFactory.load(forecast_request.model)
    if isinstance(forecast_request.data[0], list):  # Forecast request has exogenous variables
        dataset = ModelFactory.prepare_dataset(pd.DataFrame(forecast_request.data))
        dates = []
        for items in forecast_request.data:
            dates.append(items[0])
    else:
        dataset = pd.DataFrame(forecast_request.data)
        dataset[0] = pd.to_datetime(dataset[0])
        dataset.set_index(0, inplace=True)
        dates = forecast_request.data
    num_steps = len(forecast_request.data)
    output = model.predict(lookforward=num_steps, X=dataset).reset_index()
    output['index'] = dates

    output = output.values.tolist()
    

    return ForecastResponse(data=output)

