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
    def train(dataset, base_model_request):
        # Add your training logic here
        # from config file
        # URL to our SYBIL AWS service
        protocol = "http"
        host = "localhost"
        port = 8001
        endpoint = 'train'

        url = '%s://%s:%s/%s' % (protocol, host, str(port), endpoint)

        train_data = []
        dataset.index = dataset.index.strftime('%Y-%m-%d')
        dataset.reset_index(inplace=True)
        for value in dataset.values:
            train_data.append(list(value))

        api_json = {
            'data': train_data,
            'model': base_model_request  # (optional) can be commented out
        }
        response = requests.post(url, json=api_json)
        print(response)
        return response.json().get('model')
            
    def forecast(dataset, model):
        # Add your forecasting logic here

        dataset.index = dataset.index.strftime('%Y-%m-%d')
        data = dataset.reset_index()

        api_json = {
            'data': data.values.tolist(),
            'model': model
        }

        # from config file
        # URL to our SYBIL AWS service
        protocol = "http"
        host = "localhost"
        port = 8001
        endpoint = 'forecast'

        url = '%s://%s:%s/%s' % (protocol, host, str(port), endpoint)
        
        response = requests.post(url, json=api_json)
        new = response.json()
        return response.json().get('forecast')  


