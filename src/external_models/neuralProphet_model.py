class neuralProphet_model:
    def train(dataset, base_model_request):
        # Add your training logic here
        # from the dataset in routes.py, same as train_data, no need to read again from the file. read directly from the reqeust
        # from sybil request
        model_base_model_requestrequest = {'type': 'neuralprophet', 'params': {'changepoints_range':0.2, 'epochs':2, 'growth':'off'}, 'metrics':[] }
        model_base_model_requestrequest = base_model_request
        api_json = {
            'data': dataset,
            'model': model_base_model_requestrequest  # (optional) can be commented out
        }

        # from config file
        # URL to our SYBIL AWS service
        protocol = "http"
        host = "localhost"
        port = 8001
        endpoint = 'train'

        url = '%s://%s:%s/%s' % (protocol, host, str(port), endpoint)

        response = requests.post(url, json=api_json)
        print(response)

    def forecast(self):
        # Add your forecasting logic here
        pass
