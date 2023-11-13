from sybil_pb2 import TrainRequest, TrainResponse, Metric, ForecastRequest, ForecastResponse, ScalarValueList, ScalarValue
import sybil_pb2_grpc
import grpc
from concurrent import futures
import pandas as pd
import pickle
import blosc
import base64
from models.modelfactory import ModelFactory

class SybilService(sybil_pb2_grpc.SybilServicer):

    def Train(self, request, context):
        # Convert the request data to a DataFrame
        data = [[scalar_value.value for scalar_value in scalar_value_list.values] 
                for scalar_value_list in request.data]

        # Convert the list of lists to a DataFrame
        dataset = pd.DataFrame(data)

        # Prepare the model (similar to the existing logic in your FastAPI app)
        # If model_info is None, use ModelFactory defaults
        model_info = request.model
        if not model_info:
            model = ModelFactory.create_model(dataset)
        else:
            model = ModelFactory.create_model(dataset=dataset, **model_info)

        # Train the model
        training_info = model.train(dataset)

        # Serialize, compress, and encode the model
        output_model = base64.b64encode(blosc.compress(pickle.dumps(training_info['model']))).decode('utf-8')

        # Prepare metrics
        metrics = [Metric(type=metric_type, value=value) for metric_type, value in training_info["metrics"].items()]

        # Return the response
        return TrainResponse(model=output_model, type=training_info["type"], metrics=metrics)
    
    def Forecast(self, request, context):
        # Decode the model from the request
        model = pickle.loads(blosc.decompress(base64.b64decode(request.model)))

        # Convert request data to DataFrame
        forecast_data = [[scalar_value.value for scalar_value in scalar_value_list.values] 
                         for scalar_value_list in request.data]
        dataset = pd.DataFrame(forecast_data)

        # Perform the forecast (based on your existing logic)
        output = model.predict(lookforward=len(forecast_data), X=dataset).reset_index()
        output['index'] = output['index'].apply(lambda x: x.isoformat())
        forecast_output = output.values.tolist()

        # Convert forecast output to gRPC format
        response_data = [ScalarValueList(values=[ScalarValue(string_value=str(item)) for item in row]) 
                         for row in forecast_output]

        # Return the response
        return ForecastResponse(data=response_data)
