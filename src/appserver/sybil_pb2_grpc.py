# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import sybil_pb2 as sybil__pb2


class SybilStub(object):
    """The service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Train = channel.unary_unary(
                '/Sybil/Train',
                request_serializer=sybil__pb2.TrainRequest.SerializeToString,
                response_deserializer=sybil__pb2.TrainResponse.FromString,
                )
        self.Forecast = channel.unary_unary(
                '/Sybil/Forecast',
                request_serializer=sybil__pb2.ForecastRequest.SerializeToString,
                response_deserializer=sybil__pb2.ForecastResponse.FromString,
                )


class SybilServicer(object):
    """The service definition.
    """

    def Train(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Forecast(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SybilServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Train': grpc.unary_unary_rpc_method_handler(
                    servicer.Train,
                    request_deserializer=sybil__pb2.TrainRequest.FromString,
                    response_serializer=sybil__pb2.TrainResponse.SerializeToString,
            ),
            'Forecast': grpc.unary_unary_rpc_method_handler(
                    servicer.Forecast,
                    request_deserializer=sybil__pb2.ForecastRequest.FromString,
                    response_serializer=sybil__pb2.ForecastResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Sybil', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Sybil(object):
    """The service definition.
    """

    @staticmethod
    def Train(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Sybil/Train',
            sybil__pb2.TrainRequest.SerializeToString,
            sybil__pb2.TrainResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Forecast(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Sybil/Forecast',
            sybil__pb2.ForecastRequest.SerializeToString,
            sybil__pb2.ForecastResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)