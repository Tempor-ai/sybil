"""
Library of external models class
"""
class OnboardNeuralProphet:
    def fit(dataset, base_model_request):
        from models.external.rest_client import rest_client
        model = rest_client.train(dataset, base_model_request)
        return model

    def predict(data, model):
        from models.external.rest_client import rest_client
        response = rest_client.forecast(data, model)
        return response

