import json
import yaml
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import APIRouter
from pydantic import BaseModel

import pickle
import pandas as pd
from models.modelfactory import ModelFactory
from typing import Union, List
import os
import blosc
import base64
import logging
from fastapi.encoders import jsonable_encoder

class rest_client:
    def train(dataset, base_model_request):
        # Add your training logic here
        # from config file
        filePath = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        config_file=os.path.join(filePath, 'config.yml') 

        with open(config_file, 'r') as file:
            url_dict = yaml.safe_load(file)

        protocol = url_dict['protocol']
        host = url_dict['host']
        port = url_dict['port']
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
        filePath = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        config_file=os.path.join(filePath, 'config.yml') 

        with open(config_file, 'r') as file:
            url_dict = yaml.safe_load(file)

        protocol = url_dict['protocol']
        host = url_dict['host']
        port = url_dict['port']
        endpoint = 'forecast'

        url = '%s://%s:%s/%s' % (protocol, host, str(port), endpoint)
        
        response = requests.post(url, json=api_json)
        new = response.json()
        new.get('forecast')

        #check of response is the same length as the input data
        forecast_yhat = response.json().get('forecast')['yhat1']
        forecast_ds = response.json().get('forecast')['ds']
        forecast_yhat_np=np.asarray(list(forecast_yhat.values()))
        forecast_ds_np=np.asarray(list(forecast_ds.values()))
        forecast_df = pd.DataFrame(data = forecast_yhat_np, index=forecast_ds_np, columns = ['prediction']) 
        forecast_df_index= pd.to_datetime(forecast_df.index, infer_datetime_format=True).strftime('%Y-%m-%d')
        forecast_df.set_index(forecast_df_index, inplace=True)
        forecast_response_json = response.json().get('forecast')
        if (forecast_df.shape[0] != dataset.shape[0]):
            # pad the forecast result if forecast_yhat_df or forecast_ds_df has less rows than the input dataset
            merged_df = dataset.join(forecast_df)['prediction'].fillna(forecast_yhat_np[0])
            merged_df.index = merged_df.index.set_names([0])
            merged_df = merged_df.reset_index(0)
            merged_df[0] = pd.to_datetime(merged_df[0], infer_datetime_format=True)
            merged_df[0] = merged_df[0].apply(lambda x:x.isoformat())
            forecast_response_json['yhat1'] = merged_df['prediction'].to_dict()
            forecast_response_json['ds'] = merged_df[0].to_dict()

        return forecast_response_json 