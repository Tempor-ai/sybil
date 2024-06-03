class OnboardNeuralProphet:
    def fit(dataset, base_model_request):
        import requests
        from models.external.rest_client import rest_client
        # Add your training logic here
        # from the dataset in routes.py, same as train_data, no need to read again from the file. read directly from the reqeust
        # from sybil request

        model = rest_client.train(dataset, base_model_request)
        return model

    def predict(data, model):
        import requests
        import pandas as pd
        import numpy as np
        from models.external.rest_client import rest_client
        
        response = rest_client.forecast(data, model)
        # forecast_ds = response['ds']
        # forecast_yhat = response['yhat1']
        # print(response)
        # pred = np.asarray(list(response))
        # print(pred)
        return response

