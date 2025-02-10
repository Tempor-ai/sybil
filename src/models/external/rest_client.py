"""
Library of restful client for external models apis
"""
import os
import yaml
import requests
import pandas as pd
from io import StringIO

class rest_client:
    def train(dataset, base_model_request):
        # read connection details from config file
        filePath = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        config_file=os.path.join(filePath, 'config.yml') 

        with open(config_file, 'r') as file:
            url_dict = yaml.safe_load(file)

        protocol = url_dict['protocol']
        host = url_dict['host']
        port = url_dict['port']
        endpoint = 'train'

        url = '%s://%s:%s/%s' % (protocol, host, str(port), endpoint)

        # format the dataset
        dataset.index = dataset.index.strftime('%Y-%m-%d')
        dataset.reset_index(inplace=True)

        dataset.rename(columns={dataset.columns[0]: 'ds', dataset.columns[-1]: 'y'}, inplace=True)
        cols =  dataset.columns.tolist()
        col_rename=cols[:1] + cols[-1:] + cols[1:-1] 
        dataset = dataset[col_rename]

        # create the request json
        train_data = []
        for value in dataset.values:
            train_data.append(list(value))

        api_json = {
            'data': train_data,
            'model': base_model_request  # (optional) can be commented out
        }

        # send the request
        response = requests.post(url, json=api_json)
        return response.json().get('model')
            
    def forecast(dataset, model):
        # read connection details from config file
        filePath = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        config_file=os.path.join(filePath, 'config.yml') 

        with open(config_file, 'r') as file:
            url_dict = yaml.safe_load(file)

        protocol = url_dict['protocol']
        host = url_dict['host']
        port = url_dict['port']
        endpoint = 'forecast'

        url = '%s://%s:%s/%s' % (protocol, host, str(port), endpoint)

        # format the dataset
        dataset.index = dataset.index.strftime('%Y-%m-%d')
        data = dataset.reset_index()

        data.rename(columns={data.columns[0]: 'ds'}, inplace=True)
        data["y"] = 1
        
        cols =  data.columns.tolist()
        col_rename=cols[:1] + cols[-1:] + cols[1:-1] 
        data = data[col_rename]

        # create the request json
        api_json = {
            'data': data.values.tolist(),
            'model': model
        }

        # send the request
        response = requests.post(url, json=api_json)

        #parse the response
        forecast_json_out = response.json()
        forecast_data = forecast_json_out['forecast']
        forecast_df = pd.DataFrame(forecast_data)
        # forecast_df = pd.read_csv(request_data, sep=",")
        forecast_df = forecast_df.rename(columns={'ds': 'time'})

        #check of response is the same length as the input data
        dataset.index = pd.to_datetime(dataset.index, infer_datetime_format=True)
        forecast_df['time']= pd.to_datetime(forecast_df.time, infer_datetime_format=True)
        forecast_df.set_index('time', inplace=True)
        merged_df = dataset.join(forecast_df)
        if (forecast_df.shape[0] != dataset.shape[0]):
            merged_df = merged_df[['yhat1']].fillna(forecast_df['yhat1'].iloc[0])
        merged_df.reset_index(inplace=True)
        return merged_df['yhat1'].to_numpy()