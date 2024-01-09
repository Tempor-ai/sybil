import os
import sys
import time
import random
import logging
import argparse
import grpc
import sybil_pb2
import sybil_pb2_grpc

# Parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1", type=str, help="host")
parser.add_argument("--port", default=8020, type=int, help="port")
args = parser.parse_args()

def SybilClient(stub):
    # Example usage of the Train method
    train_request = sybil_pb2.TrainRequest(json='{"data":[[1,1],[2,2],[3,3]], "model": {"type": "darts_autotheta", "scorers": ["smape"]}}')

    start_time = time.time()
    train_response = stub.Train(train_request)
    response_time = time.time() - start_time

    print('\n########################################################################################\n')
    print("{:.3f}s\nModel: {}\nType: {}\nMetrics: {}".format(response_time, train_response.model, train_response.type, train_response.metrics))
    print('\n########################################################################################\n')

    exit(1)
    # Example usage of the Forecast method
    forecast_request = sybil_pb2.ForecastRequest(
        model=train_response.model,  # Base64 encoded string of your model
        data=[sybil_pb2.ScalarValueList(values=[sybil_pb2.ScalarValue(int_value=16)]),
            sybil_pb2.ScalarValueList(values=[sybil_pb2.ScalarValue(int_value=17)]),
            sybil_pb2.ScalarValueList(values=[sybil_pb2.ScalarValue(int_value=18)]),
            sybil_pb2.ScalarValueList(values=[sybil_pb2.ScalarValue(int_value=19)]),
            sybil_pb2.ScalarValueList(values=[sybil_pb2.ScalarValue(int_value=20)]),
            sybil_pb2.ScalarValueList(values=[sybil_pb2.ScalarValue(int_value=21)]),
            sybil_pb2.ScalarValueList(values=[sybil_pb2.ScalarValue(int_value=22)]),
            sybil_pb2.ScalarValueList(values=[sybil_pb2.ScalarValue(int_value=23)]),
            sybil_pb2.ScalarValueList(values=[sybil_pb2.ScalarValue(int_value=24)]),
            sybil_pb2.ScalarValueList(values=[sybil_pb2.ScalarValue(int_value=25)])]  # Populate with ScalarValueList for forecast
    )

    start_time = time.time()
    forecast_response = stub.Forecast(forecast_request)
    response_time = time.time() - start_time

    print('\n########################################################################################\n')
    print("{:.3f}s\nForecast Data: {}".format(response_time, forecast_response.data))
    print('\n########################################################################################\n')

def run():
    with grpc.insecure_channel('{}:{}'.format(args.host, args.port)) as channel:
        stub = sybil_pb2_grpc.SybilStub(channel)
        SybilClient(stub)

if __name__ == '__main__':
    logging.basicConfig()
    run()
