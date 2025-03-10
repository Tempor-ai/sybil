from sybil_pb2 import TrainRequest, TrainResponse, Metric, ForecastRequest, ForecastResponse, ScalarValueList, ScalarValue
import sybil_pb2_grpc
import grpc
from concurrent import futures
import pandas as pd
import pickle
import blosc
import base64
from models.modelfactory import ModelFactory
import time
import logging
import argparse
from concurrent import futures
import google.protobuf.json_format
import json

## Host and port on which the server listens ##
parser = argparse.ArgumentParser(description='')
parser.add_argument("--host", type=str, default="127.0.0.1",  help= "host" )
parser.add_argument("--port", type=int, default=7000,  help= "port" )
args = parser.parse_args()


def prepare_grpc_req(request):
    # Convert the request data to a DataFrame
    data = [[getattr(scalar_value, scalar_value.WhichOneof('value')) for scalar_value in scalar_value_list.values]
            for scalar_value_list in request.data]
    # Prepare the model (similar to the existing logic in your FastAPI app)
    # If model_info is None, use ModelFactory defaults
    model_info = google.protobuf.json_format.MessageToJson(request.model_info, use_integers_for_enums=False, including_default_value_fields=False, preserving_proto_field_name=True)
    model_info = json.loads(model_info)

    return data, model_info


def prepare_jsonstr_req(request):
    # JSON Schema Validation needed here

    request_dict = json.loads(request.json)

    model = None
    
    if "model" in request_dict:
        model = request_dict["model"]


    return request_dict["data"], model


def prepare_grpc_req_forecast(request):
    # Convert the request data to a DataFrame
    data = [[getattr(scalar_value, scalar_value.WhichOneof('value')) for scalar_value in scalar_value_list.values]
            for scalar_value_list in request.data]

    # If model_info is None, use ModelFactory defaults
    model = request.model

    return data, model


def prepare_jsonstr_req_forecast(request):
    # JSON Schema Validation needed here

    request_dict = json.loads(request.json)

    return request_dict["data"], request_dict["model"]


class SybilService(sybil_pb2_grpc.SybilServicer):

    def Train(self, request, context):

        data = None
        model_info = None

        if request.json is None:
            data, model_info = prepare_grpc_req(request)
        else:
            data, model_info = prepare_jsonstr_req(request)

        # Convert the list of lists to a DataFrame
        dataset = ModelFactory.prepare_dataset(pd.DataFrame(data))

        if not model_info:
            model = ModelFactory.create_model(dataset)
        else:
            model = ModelFactory.create_model(dataset=dataset, **model_info)

        # Train the model
        training_info = model.train(dataset)

        # Serialize, compress, and encode the model
        #output_model = base64.b64encode(blosc.compress(pickle.dumps(training_info['model']))).decode('utf-8')
        output_model = base64.b64encode(blosc.compress(pickle.dumps(training_info['model'])))

        # Prepare metrics
        metrics = [Metric(type=metric_type, value=value) for metric_type, value in training_info["metrics"].items()]

        # Return the response
        return TrainResponse(model=output_model, type=training_info["type"], metrics=metrics)
    
    def Forecast(self, request, context):

        forecast_data = None
        model = None

        if request.json is None:
            forecast_data, model = prepare_grpc_req_forecast(request)
        else:
            forecast_data, model = prepare_jsonstr_req_forecast(request)

        # Decode the model from the request
        model = pickle.loads(blosc.decompress(base64.b64decode(model)))


		# TODO Model currently does not support dates, array is converted into number of steps
        if isinstance(forecast_data[0], list):  # Forecast request has exogenous variables
            dataset = ModelFactory.prepare_dataset(pd.DataFrame(forecast_data))
        else:
            dataset = None
        num_steps = len(forecast_data)


        # Convert request data to DataFrame
        #dataset = ModelFactory.prepare_dataset(pd.DataFrame(forecast_data))

        # Perform the forecast (based on your existing logic)
        output = model.predict(lookforward=num_steps, X=dataset).reset_index()
        #output['index'] = output['index'].apply(lambda x: x.isoformat())
        forecast_output = output.values.tolist()

        # Convert forecast output to gRPC format
        response_data = [ScalarValueList(values=[ScalarValue(string_value=str(item)) for item in row]) 
                         for row in forecast_output]

        # Return the response
        return ForecastResponse(data=response_data)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sybil_pb2_grpc.add_SybilServicer_to_server(SybilService(), server)
    server.add_insecure_port('{}:{}'.format(args.host, args.port))
    server.start()
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    print('Server start')

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    logging.basicConfig()
    serve()
