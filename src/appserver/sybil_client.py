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
parser.add_argument("--port", default=7000, type=int, help="port")
args = parser.parse_args()

def SybilClient(stub):

    with open("/home/ubuntu/temporAI/sybil/examples/json/api_call_example_tc2_train_request.json", "r") as request_fh:
        # Example usage of the Train method
        train_request = sybil_pb2.TrainRequest(json=request_fh.read())

    start_time = time.time()
    train_response = stub.Train(train_request)
    response_time = time.time() - start_time

    print('\n########################################################################################\n')
    print("{:.3f}s\nModel: {}\nType: {}\nMetrics: {}".format(response_time, train_response.model, train_response.type, train_response.metrics))
    print('\n########################################################################################\n')

    with open("/home/ubuntu/temporAI/sybil/examples/json/api_call_example_tc1_forecast_request.json", "r") as request_fh:
        # Example usage of the Forecast method
        forecast_request = sybil_pb2.ForecastRequest(json=request_fh.read())

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
