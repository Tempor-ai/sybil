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
        forecast_ds = response['ds']
        forecast_yhat = response['yhat1']

        pred = np.asarray(list(forecast_yhat.values()))

        # forecast_df = pd.DataFrame(
        #     data={
        #         '0': forecast_ds,
        #         '': forecast_yhat,
        #     }
        # )
        # forecast_df.set_index('0', inplace=True)
        # forecast_df.index= pd.to_datetime(forecast_df.index)
        # forecast_df.index = forecast_df.index.strftime('%Y-%m-%d')
        # data_orig = pd.DataFrame(index=data.index)

        # final_pred = pd.merge(data_orig, forecast_df, left_index=True, right_index=True, how='outer')

        # pred = final_pred.values.tolist()
        return pred

