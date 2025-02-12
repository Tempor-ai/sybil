"""
Library of external models class
"""
from .rest_clients import neuralprophet_rest_client, deepsybil_rest_client

class OnboardNeuralProphet:
    def fit(dataset, base_model_request):
        # from sybil.src.models.external.rest_clients import neuralprophet_rest_client
        model = neuralprophet_rest_client.train(dataset, base_model_request)
        return model

    def predict(data, model):
        # from sybil.src.models.external.rest_clients import neuralprophet_rest_client
        response = neuralprophet_rest_client.forecast(data, model)
        return response


class DeepSYBIL:
    def fit(dataset, base_model_request):
        # from sybil.src.models.external.rest_clients import deepsybil_rest_client
        model = deepsybil_rest_client.train(dataset, base_model_request)
        return model

    def predict(data, model):
        # from sybil.src.models.external.rest_clients import deepsybil_rest_client
        response = deepsybil_rest_client.forecast(data, model)
        return response
